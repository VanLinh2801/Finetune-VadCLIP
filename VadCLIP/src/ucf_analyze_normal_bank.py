import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import CLIPVAD
from ucf_evaluate import UCFRelativeTestDataset
from utils import tools
from utils.tools import get_batch_mask


FRAMES_PER_SNIPPET = 16
LABEL_MAP = {
    "Normal": "Normal",
    "Abuse": "Abuse",
    "Arrest": "Arrest",
    "Arson": "Arson",
    "Assault": "Assault",
    "Burglary": "Burglary",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "RoadAccidents": "RoadAccidents",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Shoplifting": "Shoplifting",
    "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}
GROUPS = ["true_high", "false_high", "missed_anomaly", "true_low"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a balanced Normal visual memory bank without training."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snippets-per-chunk", default=8, type=int)
    parser.add_argument("--max-snippets-per-video", default=80, type=int)
    parser.add_argument("--heldout-ratio", default=0.2, type=float)
    parser.add_argument("--old-style-prototypes", default=32, type=int)
    parser.add_argument("--expanded-prototypes", default=256, type=int)
    parser.add_argument("--rare-exemplars", default=2048, type=int)
    parser.add_argument("--max-exemplars-per-video", default=8, type=int)
    parser.add_argument("--similarity-neighbors", default=5, type=int)
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
    parser.add_argument("--rescore-gammas", default="0,0.25,0.5,1")
    parser.add_argument("--normal-batch-size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    return parser.parse_args()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(
            dict.fromkeys(key for row in rows for key in row.keys())
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(roc_auc_score(labels, scores) * 100)


def safe_ap(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(average_precision_score(labels, scores) * 100)


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def parse_gammas(value):
    gammas = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not gammas:
        raise ValueError("--rescore-gammas must contain at least one value")
    return gammas


def as_text(value):
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def load_state_dict(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    return payload


def build_model(args, device):
    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )
    model.load_state_dict(load_state_dict(args.model_path, device), strict=True)
    model.to(device).eval()
    return model


class NormalTrainDataset(Dataset):
    def __init__(self, train_list, feature_root, visual_length):
        self.feature_root = Path(feature_root)
        self.visual_length = int(visual_length)
        with Path(train_list).open("r", encoding="utf-8-sig") as handle:
            self.rows = [
                row for row in csv.DictReader(handle) if row["label"].lower() == "normal"
            ]
        missing = [
            row["path"]
            for row in self.rows
            if not (self.feature_root / row["path"]).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} Normal feature files. First: {missing[:5]}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = np.load(self.feature_root / row["path"])
        feature, length = tools.process_feat(feature, self.visual_length)
        return torch.tensor(feature), int(length), row["video_id"], row["path"]


def evenly_spaced_indices(length, count):
    count = min(max(1, int(count)), max(1, int(length)))
    return np.unique(np.linspace(0, int(length) - 1, count, dtype=np.int64))


def collect_normal_candidates(model, dataloader, args, device):
    feature_parts = []
    video_parts = []
    chunk_parts = []
    index_parts = []
    with torch.no_grad():
        for batch_index, item in enumerate(dataloader):
            visual, lengths, video_ids, chunk_paths = item
            visual = visual.to(device, non_blocking=True)
            lengths = lengths.to(torch.int64)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            final_visual = model.encode_video(visual, padding_mask, lengths)
            classifier_input = final_visual + model.mlp2(final_visual)
            for item_index, length in enumerate(lengths.tolist()):
                indices = evenly_spaced_indices(length, args.snippets_per_chunk)
                feature_parts.append(
                    classifier_input[item_index, indices]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .numpy()
                )
                video_parts.extend([str(video_ids[item_index])] * len(indices))
                chunk_parts.extend([str(chunk_paths[item_index])] * len(indices))
                index_parts.extend(indices.tolist())
            if (batch_index + 1) % 100 == 0:
                print(f"  Normal feature batches processed: {batch_index + 1}/{len(dataloader)}")

    features = np.concatenate(feature_parts).astype(np.float32)
    video_ids = np.asarray(video_parts, dtype=object)
    chunk_paths = np.asarray(chunk_parts, dtype=object)
    snippet_indices = np.asarray(index_parts, dtype=np.int64)

    kept = []
    per_video = Counter()
    for index, video_id in enumerate(video_ids):
        if per_video[video_id] >= args.max_snippets_per_video:
            continue
        kept.append(index)
        per_video[video_id] += 1
    kept = np.asarray(kept, dtype=np.int64)
    return {
        "features": normalize(features[kept]),
        "video_id": video_ids[kept],
        "chunk_path": chunk_paths[kept],
        "snippet_index": snippet_indices[kept],
    }


def split_normal_videos(video_ids, heldout_ratio, seed):
    unique = np.asarray(sorted(set(video_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    heldout_count = max(1, int(round(len(unique) * heldout_ratio)))
    heldout = set(unique[:heldout_count].tolist())
    development = set(unique[heldout_count:].tolist())
    if development & heldout:
        raise AssertionError("Development and held-out video IDs overlap")
    return development, heldout


def fit_centroids(features, count, seed):
    count = min(int(count), len(features))
    estimator = MiniBatchKMeans(
        n_clusters=count,
        random_state=seed,
        batch_size=2048,
        n_init=3,
    )
    estimator.fit(features)
    return normalize(estimator.cluster_centers_), float(estimator.inertia_)


def max_similarity(features, bank, chunk_size=2048):
    output = []
    for start in range(0, len(features), chunk_size):
        output.append((features[start : start + chunk_size] @ bank.T).max(axis=1))
    return np.concatenate(output)


def select_rare_exemplars(features, video_ids, centroids, count, per_video_cap):
    similarity = max_similarity(features, centroids)
    order = np.argsort(similarity)
    selected = []
    selected_per_video = Counter()
    for index in order:
        video_id = str(video_ids[index])
        if selected_per_video[video_id] >= per_video_cap:
            continue
        selected.append(int(index))
        selected_per_video[video_id] += 1
        if len(selected) >= count:
            break
    if len(selected) < min(count, len(features)):
        raise ValueError(
            f"Could select only {len(selected)}/{count} rare exemplars with the per-video cap"
        )
    selected = np.asarray(selected, dtype=np.int64)
    return features[selected], selected, similarity[selected]


def build_banks(features, video_ids, args, seed):
    old_centroids, old_inertia = fit_centroids(
        features, args.old_style_prototypes, seed
    )
    expanded_centroids, expanded_inertia = fit_centroids(
        features, args.expanded_prototypes, seed + 1
    )
    exemplars, exemplar_indices, exemplar_centroid_similarity = select_rare_exemplars(
        features,
        video_ids,
        expanded_centroids,
        args.rare_exemplars,
        args.max_exemplars_per_video,
    )
    exemplar_video_counts = Counter(video_ids[exemplar_indices].tolist())
    max_exemplars_from_one_video = max(exemplar_video_counts.values(), default=0)
    if max_exemplars_from_one_video > args.max_exemplars_per_video:
        raise AssertionError("Rare-exemplar per-video cap was exceeded")
    banks = {
        "old_style": old_centroids,
        "expanded": expanded_centroids,
        "hybrid": normalize(np.concatenate([expanded_centroids, exemplars], axis=0)),
    }
    metadata = {
        "old_style": {
            "centroids": len(old_centroids),
            "exemplars": 0,
            "inertia": old_inertia,
        },
        "expanded": {
            "centroids": len(expanded_centroids),
            "exemplars": 0,
            "inertia": expanded_inertia,
        },
        "hybrid": {
            "centroids": len(expanded_centroids),
            "exemplars": len(exemplars),
            "inertia": expanded_inertia,
            "exemplar_indices": exemplar_indices,
            "exemplar_centroid_similarity": exemplar_centroid_similarity,
            "max_exemplars_from_one_video": max_exemplars_from_one_video,
        },
    }
    return banks, metadata


def mean_topk_similarity(features, bank, neighbors, chunk_size=2048):
    neighbors = min(max(1, int(neighbors)), len(bank))
    output = []
    for start in range(0, len(features), chunk_size):
        similarity = features[start : start + chunk_size] @ bank.T
        if neighbors == 1:
            values = similarity.max(axis=1)
        else:
            values = np.partition(similarity, -neighbors, axis=1)[:, -neighbors:].mean(axis=1)
        output.append(values.astype(np.float32))
    result = np.concatenate(output)
    if not np.isfinite(result).all() or result.min() < -1.0001 or result.max() > 1.0001:
        raise ValueError("Normal similarity contains invalid values")
    return result


def describe(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def coverage_diagnostics(bank_name, bank, development, heldout, heldout_video_ids, args):
    development_similarity = mean_topk_similarity(
        development, bank, args.similarity_neighbors
    )
    heldout_similarity = mean_topk_similarity(heldout, bank, args.similarity_neighbors)
    threshold = float(np.percentile(development_similarity, 5))
    summary = {
        "bank": bank_name,
        "bank_size": len(bank),
        "development_features": len(development),
        "heldout_features": len(heldout),
        "development_p05_threshold": threshold,
        "heldout_coverage": float(np.mean(heldout_similarity >= threshold)),
    }
    summary.update({f"development_{key}": value for key, value in describe(development_similarity).items()})
    summary.update({f"heldout_{key}": value for key, value in describe(heldout_similarity).items()})

    video_rows = []
    for video_id in sorted(set(heldout_video_ids)):
        mask = heldout_video_ids == video_id
        values = heldout_similarity[mask]
        row = {
            "bank": bank_name,
            "video_id": video_id,
            "coverage": float(np.mean(values >= threshold)),
        }
        row.update(describe(values))
        video_rows.append(row)
    return summary, video_rows


def snippet_ground_truth(gt_frames, snippet_count):
    expected = snippet_count * FRAMES_PER_SNIPPET
    gt_frames = np.asarray(gt_frames[:expected], dtype=np.int64)
    if len(gt_frames) < expected:
        gt_frames = np.pad(gt_frames, (0, expected - len(gt_frames)))
    reshaped = gt_frames.reshape(snippet_count, FRAMES_PER_SNIPPET)
    return (reshaped.max(axis=1) > 0).astype(np.int64), reshaped.mean(axis=1)


def topk_mask(scores, ratio):
    count = max(1, min(len(scores), int(np.ceil(len(scores) * ratio))))
    indices = np.argsort(-np.asarray(scores), kind="stable")[:count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[indices] = True
    return mask


def assign_groups(gt_any, high_mask):
    output = np.full(len(gt_any), "true_low", dtype=object)
    output[(gt_any == 1) & high_mask] = "true_high"
    output[(gt_any == 1) & ~high_mask] = "missed_anomaly"
    output[(gt_any == 0) & high_mask] = "false_high"
    return output


def make_lengths(length, visual_length):
    remaining = int(length)
    lengths = torch.zeros(int(length / visual_length) + 1)
    for index in range(int(length / visual_length) + 1):
        if index == 0 and remaining < visual_length:
            lengths[index] = remaining
        elif index == 0 and remaining > visual_length:
            lengths[index] = visual_length
            remaining -= visual_length
        elif remaining > visual_length:
            lengths[index] = visual_length
            remaining -= visual_length
        else:
            lengths[index] = remaining
    return lengths.to(torch.int64)


def run_test_inference(model, dataloader, gt, args, device):
    features = []
    scores = []
    groups = []
    gt_any_parts = []
    gt_fraction_parts = []
    video_ids = []
    class_names = []
    snippet_indices = []
    gt_cursor = 0
    with torch.no_grad():
        for batch_index, item in enumerate(dataloader):
            visual = item[0].squeeze(0)
            class_name = as_text(item[1])
            video_id = as_text(item[3])
            snippet_count = int(item[2])
            if snippet_count < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = make_lengths(snippet_count, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            final_visual = model.encode_video(visual, padding_mask, lengths)
            classifier_input = final_visual + model.mlp2(final_visual)
            logits = model.classifier(classifier_input)
            valid_features = classifier_input.reshape(-1, classifier_input.shape[-1])[:snippet_count]
            valid_scores = torch.sigmoid(logits.reshape(-1)[:snippet_count])

            frame_count = snippet_count * FRAMES_PER_SNIPPET
            gt_frames = gt[gt_cursor : gt_cursor + frame_count]
            gt_cursor += frame_count
            current_gt_any, current_gt_fraction = snippet_ground_truth(
                gt_frames, snippet_count
            )
            current_scores = valid_scores.cpu().numpy()
            current_groups = assign_groups(
                current_gt_any, topk_mask(current_scores, args.top_k_ratio)
            )

            features.append(valid_features.cpu().to(torch.float16).numpy())
            scores.append(current_scores)
            groups.append(current_groups)
            gt_any_parts.append(current_gt_any)
            gt_fraction_parts.append(current_gt_fraction)
            video_ids.extend([video_id] * snippet_count)
            class_names.extend([class_name] * snippet_count)
            snippet_indices.extend(range(snippet_count))
            if (batch_index + 1) % 50 == 0:
                print(f"  Test videos processed: {batch_index + 1}/{len(dataloader)}")

    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT has {len(gt)}")
    return {
        "features": normalize(np.concatenate(features).astype(np.float32)),
        "classifier_score": np.concatenate(scores).astype(np.float32),
        "group": np.concatenate(groups),
        "gt_any": np.concatenate(gt_any_parts),
        "gt_fraction": np.concatenate(gt_fraction_parts),
        "video_id": np.asarray(video_ids, dtype=object),
        "class_name": np.asarray(class_names, dtype=object),
        "snippet_index": np.asarray(snippet_indices, dtype=np.int64),
    }


def group_masks(metadata):
    return {
        "true_high": metadata["group"] == "true_high",
        "missed_anomaly": metadata["group"] == "missed_anomaly",
        "true_low": metadata["group"] == "true_low",
        "false_high_abnormal_video": (metadata["group"] == "false_high")
        & (metadata["class_name"] != "Normal"),
        "false_high_normal_video": (metadata["group"] == "false_high")
        & (metadata["class_name"] == "Normal"),
    }


def add_comparison(rows, bank_name, comparison, positive_name, negative_name, similarity, masks):
    positive = masks[positive_name]
    negative = masks[negative_name]
    pair = positive | negative
    labels = positive[pair].astype(np.int64)
    rows.append(
        {
            "bank": bank_name,
            "comparison": comparison,
            "positive_class": positive_name,
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "auc": safe_auc(labels, similarity[pair]),
            "ap": safe_ap(labels, similarity[pair]),
            "positive_similarity_mean": float(similarity[positive].mean()),
            "negative_similarity_mean": float(similarity[negative].mean()),
            "mean_gap_positive_minus_negative": float(
                similarity[positive].mean() - similarity[negative].mean()
            ),
        }
    )


def efficacy_diagnostics(bank_name, similarity, metadata):
    masks = group_masks(metadata)
    group_rows = []
    for group_name, mask in masks.items():
        row = {"bank": bank_name, "scope": group_name}
        row.update(describe(similarity[mask]))
        group_rows.append(row)

    separation_rows = []
    add_comparison(
        separation_rows,
        bank_name,
        "false_high_abnormal_vs_missed_anomaly",
        "false_high_abnormal_video",
        "missed_anomaly",
        similarity,
        masks,
    )
    add_comparison(
        separation_rows,
        bank_name,
        "false_high_abnormal_vs_true_high",
        "false_high_abnormal_video",
        "true_high",
        similarity,
        masks,
    )
    normal_mask = metadata["gt_any"] == 0
    anomaly_mask = metadata["gt_any"] == 1
    pair = normal_mask | anomaly_mask
    separation_rows.append(
        {
            "bank": bank_name,
            "comparison": "all_normal_vs_all_anomaly",
            "positive_class": "normal_snippet",
            "positive_count": int(normal_mask.sum()),
            "negative_count": int(anomaly_mask.sum()),
            "auc": safe_auc(normal_mask[pair].astype(np.int64), similarity[pair]),
            "ap": safe_ap(normal_mask[pair].astype(np.int64), similarity[pair]),
            "positive_similarity_mean": float(similarity[normal_mask].mean()),
            "negative_similarity_mean": float(similarity[anomaly_mask].mean()),
            "mean_gap_positive_minus_negative": float(
                similarity[normal_mask].mean() - similarity[anomaly_mask].mean()
            ),
        }
    )

    class_rows = []
    for class_name in sorted(set(metadata["class_name"])):
        class_mask = metadata["class_name"] == class_name
        for group_name, mask in masks.items():
            current = class_mask & mask
            if not current.any():
                continue
            row = {"bank": bank_name, "class_name": class_name, "scope": group_name}
            row.update(describe(similarity[current]))
            class_rows.append(row)

        false_mask = class_mask & masks["false_high_abnormal_video"]
        missed_mask = class_mask & masks["missed_anomaly"]
        if false_mask.any() and missed_mask.any():
            pair_mask = false_mask | missed_mask
            labels = false_mask[pair_mask].astype(np.int64)
            class_rows.append(
                {
                    "bank": bank_name,
                    "class_name": class_name,
                    "scope": "false_high_vs_missed_separation",
                    "count": int(pair_mask.sum()),
                    "mean": np.nan,
                    "std": np.nan,
                    "p10": np.nan,
                    "p50": np.nan,
                    "p90": np.nan,
                    "auc_false_high_as_more_normal": safe_auc(
                        labels, similarity[pair_mask]
                    ),
                    "mean_gap_false_high_minus_missed": float(
                        similarity[false_mask].mean() - similarity[missed_mask].mean()
                    ),
                }
            )
    return group_rows, separation_rows, class_rows


def percentile_normality(similarity, calibration_values):
    reference = np.sort(np.asarray(calibration_values, dtype=np.float32))
    return np.searchsorted(reference, similarity, side="right").astype(np.float32) / max(
        1, len(reference)
    )


def rescore_diagnostics(bank_name, similarity, calibration_values, metadata, gt, gammas):
    probability = np.clip(metadata["classifier_score"], 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability))
    normality = percentile_normality(similarity, calibration_values)
    rows = []
    for gamma in gammas:
        adjusted = 1 / (1 + np.exp(-(logits - gamma * normality)))
        frame_scores = np.repeat(adjusted, FRAMES_PER_SNIPPET)[: len(gt)]
        rows.append(
            {
                "bank": bank_name,
                "gamma": gamma,
                "frame_auc": safe_auc(gt, frame_scores),
                "frame_ap": safe_ap(gt, frame_scores),
                "score_mean": float(adjusted.mean()),
                "normality_mean": float(normality.mean()),
            }
        )
    baseline = next((row for row in rows if row["gamma"] == 0), rows[0])
    for row in rows:
        row["auc_delta_vs_gamma_0"] = row["frame_auc"] - baseline["frame_auc"]
        row["ap_delta_vs_gamma_0"] = row["frame_ap"] - baseline["frame_ap"]
    return rows


def print_bank_summary(bank_name, coverage, separation, rescoring):
    false_missed = next(
        row
        for row in separation
        if row["comparison"] == "false_high_abnormal_vs_missed_anomaly"
    )
    false_true = next(
        row
        for row in separation
        if row["comparison"] == "false_high_abnormal_vs_true_high"
    )
    best_rescore = max(rescoring, key=lambda row: row["frame_auc"])
    print("\n" + "-" * 80)
    print(f"Bank: {bank_name} | size={coverage.get('bank_size', 'reference')}")
    if "heldout_coverage" in coverage:
        print(
            f"  Held-out Normal coverage: {coverage['heldout_coverage'] * 100:.2f}% "
            f"| threshold={coverage['development_p05_threshold']:.4f}"
        )
    print(
        "  False-high vs missed anomaly: "
        f"AUC={false_missed['auc']:.2f} | "
        f"gap={false_missed['mean_gap_positive_minus_negative']:+.4f}"
    )
    print(
        "  False-high vs true-high:      "
        f"AUC={false_true['auc']:.2f} | "
        f"gap={false_true['mean_gap_positive_minus_negative']:+.4f}"
    )
    print(
        f"  Best diagnostic rescore: gamma={best_rescore['gamma']:.2f} | "
        f"AUC={best_rescore['frame_auc']:.2f} "
        f"({best_rescore['auc_delta_vs_gamma_0']:+.2f}) | "
        f"AP={best_rescore['frame_ap']:.2f} "
        f"({best_rescore['ap_delta_vs_gamma_0']:+.2f})"
    )


def main():
    args = parse_args()
    if args.snippets_per_chunk <= 0 or args.max_snippets_per_video <= 0:
        raise ValueError("Snippet sampling limits must be positive")
    if not 0 < args.heldout_ratio < 1:
        raise ValueError("--heldout-ratio must be in (0, 1)")
    gammas = parse_gammas(args.rescore_gammas)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Comprehensive Normal visual bank diagnostics")
    print("Diagnostic only: no training and no loss changes")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Feature root: {args.feature_root}")
    model = build_model(args, device)

    normal_dataset = NormalTrainDataset(
        args.train_list, args.feature_root, args.visual_length
    )
    normal_unique_videos = sorted({row["video_id"] for row in normal_dataset.rows})
    print(
        f"Normal train source: rows={len(normal_dataset)} "
        f"| unique_videos={len(normal_unique_videos)}"
    )
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.normal_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    candidates = collect_normal_candidates(model, normal_loader, args, device)
    counts = Counter(candidates["video_id"].tolist())
    print(
        f"Balanced candidates: {len(candidates['features'])} "
        f"| unique_videos={len(counts)} | max_per_video={max(counts.values())}"
    )
    if len(normal_dataset) != 8000 or len(normal_unique_videos) != 800:
        print(
            "WARNING: expected 8000 Normal rows and 800 videos, "
            f"found {len(normal_dataset)} rows and {len(normal_unique_videos)} videos."
        )

    development_videos, heldout_videos = split_normal_videos(
        candidates["video_id"], args.heldout_ratio, args.seed
    )
    development_mask = np.isin(candidates["video_id"], list(development_videos))
    heldout_mask = np.isin(candidates["video_id"], list(heldout_videos))
    development_features = candidates["features"][development_mask]
    heldout_features = candidates["features"][heldout_mask]
    heldout_video_ids = candidates["video_id"][heldout_mask]
    print(
        f"Development split: {len(development_videos)} videos, "
        f"{len(development_features)} features"
    )
    print(
        f"Held-out split: {len(heldout_videos)} videos, "
        f"{len(heldout_features)} features"
    )

    print("\nFitting development banks...")
    development_banks, development_metadata = build_banks(
        development_features,
        candidates["video_id"][development_mask],
        args,
        args.seed,
    )
    coverage_rows = []
    coverage_video_rows = []
    for bank_name, bank in development_banks.items():
        summary, video_rows = coverage_diagnostics(
            bank_name,
            bank,
            development_features,
            heldout_features,
            heldout_video_ids,
            args,
        )
        coverage_rows.append(summary)
        coverage_video_rows.extend(video_rows)

    print("\nFitting final banks from all Normal training videos...")
    final_banks, final_metadata = build_banks(
        candidates["features"], candidates["video_id"], args, args.seed + 100
    )
    expected_sizes = {
        "old_style": args.old_style_prototypes,
        "expanded": args.expanded_prototypes,
        "hybrid": args.expanded_prototypes + args.rare_exemplars,
    }
    for bank_name, expected in expected_sizes.items():
        if len(final_banks[bank_name]) != expected:
            raise ValueError(
                f"{bank_name} bank has {len(final_banks[bank_name])}, expected {expected}"
            )

    np.savez_compressed(
        output_dir / "normal_visual_banks.npz",
        **{name: bank.astype(np.float32) for name, bank in final_banks.items()},
    )

    test_dataset = UCFRelativeTestDataset(
        args.visual_length, args.test_list, args.feature_root
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    gt = np.load(args.gt_path)
    print("\nRunning baseline inference on the test set...")
    test_metadata = run_test_inference(model, test_loader, gt, args, device)

    group_rows = []
    separation_rows = []
    class_rows = []
    rescore_rows = []
    calibration_rows = []
    acceptance = {}
    for bank_name, bank in final_banks.items():
        test_similarity = mean_topk_similarity(
            test_metadata["features"], bank, args.similarity_neighbors
        )
        calibration_similarity = mean_topk_similarity(
            heldout_features, bank, args.similarity_neighbors
        )
        calibration_row = {"bank": bank_name}
        calibration_row.update(describe(calibration_similarity))
        calibration_rows.append(calibration_row)

        current_groups, current_separation, current_classes = efficacy_diagnostics(
            bank_name, test_similarity, test_metadata
        )
        current_rescore = rescore_diagnostics(
            bank_name,
            test_similarity,
            calibration_similarity,
            test_metadata,
            gt,
            gammas,
        )
        group_rows.extend(current_groups)
        separation_rows.extend(current_separation)
        class_rows.extend(current_classes)
        rescore_rows.extend(current_rescore)
        coverage = next(row for row in coverage_rows if row["bank"] == bank_name)
        print_bank_summary(bank_name, coverage, current_separation, current_rescore)

        false_missed = next(
            row
            for row in current_separation
            if row["comparison"] == "false_high_abnormal_vs_missed_anomaly"
        )
        best_rescore = max(current_rescore, key=lambda row: row["frame_auc"])
        baseline_rescore = next(
            (row for row in current_rescore if row["gamma"] == 0), current_rescore[0]
        )
        valid_class_rows = [
            row
            for row in current_classes
            if row["scope"] == "false_high_vs_missed_separation"
        ]
        positive_classes = sum(
            row["mean_gap_false_high_minus_missed"] > 0 for row in valid_class_rows
        )
        acceptance[bank_name] = {
            "heldout_coverage_at_development_p05": coverage["heldout_coverage"],
            "false_high_vs_missed_auc": false_missed["auc"],
            "false_high_minus_missed_similarity_gap": false_missed[
                "mean_gap_positive_minus_negative"
            ],
            "classes_with_positive_false_high_minus_missed_gap": positive_classes,
            "classes_evaluated": len(valid_class_rows),
            "best_rescore_gamma": best_rescore["gamma"],
            "best_rescore_auc_delta": best_rescore["frame_auc"]
            - baseline_rescore["frame_auc"],
            "best_rescore_ap_delta": best_rescore["frame_ap"]
            - baseline_rescore["frame_ap"],
            "passes_initial_gate": bool(
                coverage["heldout_coverage"] >= 0.95
                and false_missed["auc"] >= 60
                and false_missed["mean_gap_positive_minus_negative"] > 0
                and positive_classes > len(valid_class_rows) / 2
                and best_rescore["frame_auc"] - baseline_rescore["frame_auc"] >= 0.2
                and best_rescore["frame_ap"] - baseline_rescore["frame_ap"] >= -0.5
            ),
        }

    # Reproduce the old analyzer's max-to-32-centroid similarity for reference only.
    old_reference_similarity = mean_topk_similarity(
        test_metadata["features"], final_banks["old_style"], 1
    )
    old_groups, old_separation, old_classes = efficacy_diagnostics(
        "old_style_top1_reference", old_reference_similarity, test_metadata
    )
    old_calibration = mean_topk_similarity(heldout_features, final_banks["old_style"], 1)
    old_rescore = rescore_diagnostics(
        "old_style_top1_reference",
        old_reference_similarity,
        old_calibration,
        test_metadata,
        gt,
        gammas,
    )
    group_rows.extend(old_groups)
    separation_rows.extend(old_separation)
    class_rows.extend(old_classes)
    rescore_rows.extend(old_rescore)
    print_bank_summary(
        "old_style_top1_reference", {"bank_size": 32}, old_separation, old_rescore
    )

    bank_metadata_rows = []
    for split_name, metadata, feature_count, video_count in [
        (
            "development",
            development_metadata,
            len(development_features),
            len(development_videos),
        ),
        ("final", final_metadata, len(candidates["features"]), len(counts)),
    ]:
        for bank_name, values in metadata.items():
            bank_metadata_rows.append(
                {
                    "split": split_name,
                    "bank": bank_name,
                    "source_rows": len(normal_dataset),
                    "source_unique_videos": len(normal_unique_videos),
                    "bank_video_count": video_count,
                    "candidate_feature_count": feature_count,
                    "centroid_count": values["centroids"],
                    "exemplar_count": values["exemplars"],
                    "bank_size": values["centroids"] + values["exemplars"],
                    "kmeans_inertia": values["inertia"],
                    "max_exemplars_from_one_video": values.get(
                        "max_exemplars_from_one_video", 0
                    ),
                }
            )

    exemplar_rows = []
    exemplar_indices = final_metadata["hybrid"]["exemplar_indices"]
    exemplar_similarity = final_metadata["hybrid"]["exemplar_centroid_similarity"]
    for rank, (index, similarity) in enumerate(
        zip(exemplar_indices, exemplar_similarity), start=1
    ):
        exemplar_rows.append(
            {
                "rank": rank,
                "video_id": candidates["video_id"][index],
                "chunk_path": candidates["chunk_path"][index],
                "snippet_index": int(candidates["snippet_index"][index]),
                "nearest_centroid_similarity": float(similarity),
            }
        )

    write_csv(output_dir / "bank_metadata.csv", bank_metadata_rows)
    write_csv(output_dir / "coverage_summary.csv", coverage_rows)
    write_csv(output_dir / "coverage_per_video.csv", coverage_video_rows)
    write_csv(output_dir / "calibration_summary.csv", calibration_rows)
    write_csv(output_dir / "test_group_similarity.csv", group_rows)
    write_csv(output_dir / "test_per_class_similarity.csv", class_rows)
    write_csv(output_dir / "separation_metrics.csv", separation_rows)
    write_csv(output_dir / "rescoring_metrics.csv", rescore_rows)
    write_csv(output_dir / "hybrid_rare_exemplars.csv", exemplar_rows)
    with (output_dir / "acceptance_report.json").open("w", encoding="utf-8") as handle:
        json.dump(acceptance, handle, ensure_ascii=False, indent=2)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    print("\nAcceptance report")
    for bank_name, values in acceptance.items():
        print(
            f"  {bank_name}: {'PASS' if values['passes_initial_gate'] else 'FAIL'} "
            f"| positive_classes={values['classes_with_positive_false_high_minus_missed_gap']}"
            f"/{values['classes_evaluated']}"
        )
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
