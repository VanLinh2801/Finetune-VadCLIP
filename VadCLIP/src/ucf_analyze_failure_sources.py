import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

from ucf_analyze_score_separation import (
    UCFRelativeTestDataset,
    get_prompt_text_for_config,
    get_prompt_weights_for_config,
    load_model,
    make_lengths,
)
from ucf_analyze_visual_features import (
    CLASS_NAMES,
    FRAMES_PER_SNIPPET,
    GROUPS,
    assign_groups,
    capture_forward,
    flatten_valid,
    normalize,
    selected_mask,
    snippet_gt,
    write_csv,
)
from utils import tools
from utils.tools import get_batch_mask


BANK_STAGES = ["raw_clip", "final_visual", "classifier_input"]


def parse_float_list(value):
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one float value.")
    return values


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(roc_auc_score(labels, scores) * 100)


def safe_ap(labels, scores):
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(average_precision_score(labels, scores) * 100)


class NormalTrainDataset(Dataset):
    def __init__(self, train_list, feature_root, visual_length, max_rows, seed):
        self.feature_root = Path(feature_root)
        self.visual_length = visual_length
        with open(train_list, "r", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row["label"] == "Normal"]
        if max_rows > 0 and len(rows) > max_rows:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(rows), max_rows, replace=False))
            rows = [rows[index] for index in selected]
        self.rows = rows
        missing = [row["path"] for row in rows if not (self.feature_root / row["path"]).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} sampled Normal feature files. First: {missing[:5]}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = np.load(self.feature_root / row["path"])
        feature, length = tools.process_feat(feature, self.visual_length)
        return torch.tensor(feature), int(length), row["video_id"]


def sample_four_groups(groups, max_per_group, seed, candidate_mask=None):
    rng = np.random.default_rng(seed)
    output = []
    if candidate_mask is None:
        candidate_mask = np.ones(len(groups), dtype=bool)
    for group in GROUPS:
        indices = np.flatnonzero((groups == group) & candidate_mask)
        if indices.size > max_per_group:
            indices = rng.choice(indices, max_per_group, replace=False)
        output.extend(indices.tolist())
    return np.asarray(sorted(output), dtype=np.int64)


def context_text_base(model, prompt_text, model_type):
    if model_type == "multi_prompt":
        return model.encode_textprompt_groups(prompt_text)
    if model_type == "prompt_replacement":
        return model.encode_textprompt_groups(prompt_text)
    return model.encode_textprompt(prompt_text)


def counterfactual_logits2(
    model,
    final_visual,
    logits1,
    text_base,
    prompt_weights,
    model_type,
    gamma,
):
    visual_context = logits1.permute(0, 2, 1) @ final_visual
    visual_context = F.normalize(visual_context, dim=-1)
    visual_norm = F.normalize(final_visual, dim=-1)

    if model_type == "multi_prompt":
        context = visual_context.unsqueeze(2).expand(
            visual_context.shape[0],
            text_base.shape[0],
            text_base.shape[1],
            visual_context.shape[-1],
        )
        text_features = text_base.unsqueeze(0).expand(
            visual_context.shape[0], text_base.shape[0], text_base.shape[1], text_base.shape[2]
        )
        text_features = text_features + gamma * context
        text_features = text_features + model.mlp1(text_features)
        prompt_logits = torch.einsum(
            "btd,bckd->btck",
            visual_norm,
            F.normalize(text_features, dim=-1).type(visual_norm.dtype),
        ) / 0.07
        if prompt_weights is None:
            return torch.logsumexp(prompt_logits, dim=-1) - math.log(prompt_logits.shape[-1])
        weights = prompt_weights.to(prompt_logits.device, dtype=prompt_logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.logsumexp(
            prompt_logits + weights.clamp_min(1e-8).log().view(1, 1, *weights.shape),
            dim=-1,
        )

    context = visual_context.expand(
        visual_context.shape[0], text_base.shape[0], visual_context.shape[-1]
    )
    text_features = text_base.unsqueeze(0).expand(
        visual_context.shape[0], text_base.shape[0], text_base.shape[1]
    )
    text_features = text_features + gamma * context
    text_features = text_features + model.mlp1(text_features)
    return visual_norm @ F.normalize(text_features, dim=-1).permute(0, 2, 1).type(
        visual_norm.dtype
    ) / 0.07


def run_test_inference(model, dataloader, args, config, gt, device):
    model_type = config.get("model_type", "baseline")
    prompt_text = get_prompt_text_for_config(config)
    prompt_weights = get_prompt_weights_for_config(config, device)
    with torch.no_grad():
        text_base = context_text_base(model, prompt_text, model_type)

    records = []
    gt_cursor = 0
    gamma_one_max_error = 0.0
    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            class_name = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            video_id = item[3][0] if isinstance(item[3], (list, tuple)) else item[3]
            snippet_count = int(item[2])
            if snippet_count < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = make_lengths(snippet_count, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            output = capture_forward(
                model,
                visual,
                padding_mask,
                prompt_text,
                lengths,
                prompt_weights,
                model_type,
            )
            final_visual = output["final_visual"]
            classifier_input = final_visual + model.mlp2(final_visual)
            logits1 = output["logits1"]
            classifier_score = torch.sigmoid(flatten_valid(logits1, snippet_count).squeeze(-1))
            original_logits2 = flatten_valid(output["logits2"], snippet_count)

            gamma_logits = {}
            for gamma in args.context_gammas:
                logits2 = counterfactual_logits2(
                    model,
                    final_visual,
                    logits1,
                    text_base,
                    prompt_weights,
                    model_type,
                    gamma,
                )
                gamma_logits[gamma] = flatten_valid(logits2, snippet_count).cpu().numpy()
            if 1.0 in gamma_logits:
                error = np.max(
                    np.abs(gamma_logits[1.0] - original_logits2.cpu().numpy())
                )
                gamma_one_max_error = max(gamma_one_max_error, float(error))

            frame_count = snippet_count * FRAMES_PER_SNIPPET
            gt_frames = np.asarray(gt[gt_cursor : gt_cursor + frame_count], dtype=np.int64)
            gt_cursor += frame_count
            gt_any, gt_fraction = snippet_gt(gt_frames, snippet_count)
            classifier_numpy = classifier_score.cpu().numpy()
            group = assign_groups(gt_any, selected_mask(classifier_numpy, args.top_k_ratio))
            records.append(
                {
                    "video_id": str(video_id),
                    "class_name": str(class_name),
                    "snippet_count": snippet_count,
                    "gt_frames": gt_frames,
                    "gt_any": gt_any,
                    "gt_fraction": gt_fraction,
                    "group": group,
                    "classifier_score": classifier_numpy,
                    "classifier_logit": flatten_valid(logits1, snippet_count).squeeze(-1).cpu().numpy(),
                    "context_logits": gamma_logits,
                    "features": {
                        "raw_clip": flatten_valid(output["raw_clip"], snippet_count)
                        .cpu()
                        .to(torch.float16)
                        .numpy(),
                        "final_visual": flatten_valid(final_visual, snippet_count)
                        .cpu()
                        .to(torch.float16)
                        .numpy(),
                        "classifier_input": flatten_valid(classifier_input, snippet_count)
                        .cpu()
                        .to(torch.float16)
                        .numpy(),
                    },
                }
            )
    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)} frames.")
    return records, gamma_one_max_error


def flatten_test_records(records):
    metadata = {
        "video_id": np.concatenate(
            [np.repeat(record["video_id"], record["snippet_count"]) for record in records]
        ),
        "class_name": np.concatenate(
            [np.repeat(record["class_name"], record["snippet_count"]) for record in records]
        ),
        "snippet_index": np.concatenate(
            [np.arange(record["snippet_count"]) for record in records]
        ),
        "gt_any": np.concatenate([record["gt_any"] for record in records]),
        "group": np.concatenate([record["group"] for record in records]),
        "classifier_score": np.concatenate([record["classifier_score"] for record in records]),
        "classifier_logit": np.concatenate([record["classifier_logit"] for record in records]),
    }
    features = {
        stage: np.concatenate([record["features"][stage] for record in records]).astype(np.float32)
        for stage in BANK_STAGES
    }
    return metadata, features


def grouped_probe_and_head(model_name, stage, features, metadata, indices):
    x = normalize(features[indices])
    y = metadata["gt_any"][indices]
    video_ids = metadata["video_id"][indices]
    head_scores = metadata["classifier_score"][indices]
    probe_auc = []
    head_auc = []
    splitter = GroupKFold(n_splits=3)
    for train_indices, test_indices in splitter.split(x, y, video_ids):
        if np.unique(y[train_indices]).size < 2 or np.unique(y[test_indices]).size < 2:
            continue
        probe = LogisticRegression(
            max_iter=400, class_weight="balanced", solver="liblinear"
        )
        probe.fit(x[train_indices], y[train_indices])
        probe_auc.append(
            roc_auc_score(y[test_indices], probe.predict_proba(x[test_indices])[:, 1]) * 100
        )
        head_auc.append(roc_auc_score(y[test_indices], head_scores[test_indices]) * 100)
    return {
        "model": model_name,
        "stage": stage,
        "sample_count": len(indices),
        "folds": len(probe_auc),
        "probe_auc_mean": float(np.mean(probe_auc)) if probe_auc else np.nan,
        "probe_auc_std": float(np.std(probe_auc)) if probe_auc else np.nan,
        "same_fold_classifier_auc_mean": float(np.mean(head_auc)) if head_auc else np.nan,
        "probe_minus_classifier_auc": float(np.mean(probe_auc) - np.mean(head_auc))
        if probe_auc and head_auc
        else np.nan,
    }


def knn_binary_purity(features, labels, indices, neighbors):
    x = normalize(features[indices])
    y = labels[indices]
    count = min(neighbors + 1, len(indices))
    nearest = NearestNeighbors(n_neighbors=count, metric="cosine").fit(x)
    neighbor_indices = nearest.kneighbors(x, return_distance=False)[:, 1:]
    return float(np.mean(y[neighbor_indices] == y[:, None]))


def classifier_diagnostics(model_name, features, metadata, args):
    head_rows = []
    scopes = {
        "all_test": np.ones(len(metadata["group"]), dtype=bool),
        "abnormal_videos_only": metadata["class_name"] != "Normal",
    }
    for scope_index, (scope, candidate_mask) in enumerate(scopes.items()):
        indices = sample_four_groups(
            metadata["group"],
            args.max_samples_per_group,
            args.seed + scope_index,
            candidate_mask,
        )
        for stage in ["final_visual", "classifier_input"]:
            row = grouped_probe_and_head(model_name, stage, features[stage], metadata, indices)
            row["scope"] = scope
            row["knn_anomaly_normal_purity"] = knn_binary_purity(
                features[stage], metadata["gt_any"], indices, args.knn_neighbors
            )
            head_rows.append(row)

    z = normalize(features["final_visual"])
    h = normalize(features["classifier_input"])
    same_snippet_cosine = np.sum(z * h, axis=1)
    group_rows = []
    for group in GROUPS:
        mask = metadata["group"] == group
        group_rows.append(
            {
                "model": model_name,
                "group": group,
                "count": int(mask.sum()),
                "final_to_classifier_input_cosine_mean": float(same_snippet_cosine[mask].mean()),
                "classifier_input_norm_mean": float(
                    np.linalg.norm(features["classifier_input"][mask], axis=1).mean()
                ),
                "classifier_logit_mean": float(metadata["classifier_logit"][mask].mean()),
                "classifier_score_mean": float(metadata["classifier_score"][mask].mean()),
            }
        )
    return head_rows, group_rows


def collect_normal_train_features(model, dataloader, args, device):
    collected = {stage: [] for stage in BANK_STAGES}
    with torch.no_grad():
        for visual, lengths, _video_ids in dataloader:
            visual = visual.to(device, non_blocking=True)
            lengths = lengths.to(torch.int64)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            final_visual = model.encode_video(visual, padding_mask, lengths)
            classifier_input = final_visual + model.mlp2(final_visual)
            stage_tensors = {
                "raw_clip": visual,
                "final_visual": final_visual,
                "classifier_input": classifier_input,
            }
            for item_index, length in enumerate(lengths.tolist()):
                for stage in BANK_STAGES:
                    collected[stage].append(
                        stage_tensors[stage][item_index, :length]
                        .detach()
                        .cpu()
                        .to(torch.float16)
                        .numpy()
                    )
    return {
        stage: np.concatenate(collected[stage]).astype(np.float32) for stage in BANK_STAGES
    }


def fit_normal_bank(features, prototype_count, max_features, seed):
    rng = np.random.default_rng(seed)
    features = normalize(features)
    if max_features > 0 and len(features) > max_features:
        indices = rng.choice(len(features), max_features, replace=False)
        features = features[indices]
    cluster_count = min(prototype_count, len(features))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        batch_size=2048,
        n_init=3,
    )
    kmeans.fit(features)
    prototypes = normalize(kmeans.cluster_centers_)
    return prototypes, len(features), float(kmeans.inertia_)


def max_prototype_similarity(features, prototypes, chunk_size=4096):
    features = normalize(features)
    output = []
    for start in range(0, len(features), chunk_size):
        output.append((features[start : start + chunk_size] @ prototypes.T).max(axis=1))
    return np.concatenate(output)


def normal_bank_diagnostics(model_name, features, metadata, banks):
    group_rows = []
    auc_rows = []
    class_rows = []
    for stage, prototypes in banks.items():
        similarity = max_prototype_similarity(features[stage], prototypes)
        scopes = {
            "true_high": metadata["group"] == "true_high",
            "missed_anomaly": metadata["group"] == "missed_anomaly",
            "true_low": metadata["group"] == "true_low",
            "false_high_abnormal_video": (metadata["group"] == "false_high")
            & (metadata["class_name"] != "Normal"),
            "false_high_normal_video": (metadata["group"] == "false_high")
            & (metadata["class_name"] == "Normal"),
        }
        for scope, mask in scopes.items():
            values = similarity[mask]
            group_rows.append(
                {
                    "model": model_name,
                    "bank_stage": stage,
                    "scope": scope,
                    "count": int(mask.sum()),
                    "normal_similarity_mean": float(values.mean()),
                    "normal_similarity_p10": float(np.percentile(values, 10)),
                    "normal_similarity_p50": float(np.percentile(values, 50)),
                    "normal_similarity_p90": float(np.percentile(values, 90)),
                }
            )

        false_mask = scopes["false_high_abnormal_video"]
        missed_mask = scopes["missed_anomaly"]
        pair_mask = false_mask | missed_mask
        pair_labels = false_mask[pair_mask].astype(np.int64)
        auc_rows.append(
            {
                "model": model_name,
                "bank_stage": stage,
                "comparison": "false_high_abnormal_vs_missed_anomaly",
                "positive_class": "false_high_abnormal_video",
                "auc": safe_auc(pair_labels, similarity[pair_mask]),
                "ap": safe_ap(pair_labels, similarity[pair_mask]),
                "mean_gap_false_high_minus_missed": float(
                    similarity[false_mask].mean() - similarity[missed_mask].mean()
                ),
            }
        )
        anomaly_mask = np.isin(metadata["group"], ["true_high", "missed_anomaly"])
        normal_mask = ~anomaly_mask
        labels_normal = normal_mask.astype(np.int64)
        auc_rows.append(
            {
                "model": model_name,
                "bank_stage": stage,
                "comparison": "all_normal_vs_all_anomaly",
                "positive_class": "normal",
                "auc": safe_auc(labels_normal, similarity),
                "ap": safe_ap(labels_normal, similarity),
                "mean_gap_false_high_minus_missed": np.nan,
            }
        )

        for class_name in sorted(set(metadata["class_name"]) - {"Normal"}):
            class_mask = metadata["class_name"] == class_name
            false_class = false_mask & class_mask
            missed_class = missed_mask & class_mask
            pair = false_class | missed_class
            if false_class.sum() == 0 or missed_class.sum() == 0:
                continue
            labels = false_class[pair].astype(np.int64)
            class_rows.append(
                {
                    "model": model_name,
                    "bank_stage": stage,
                    "class_name": class_name,
                    "false_high_count": int(false_class.sum()),
                    "missed_anomaly_count": int(missed_class.sum()),
                    "auc_false_high_as_more_normal": safe_auc(labels, similarity[pair]),
                    "false_high_similarity_mean": float(similarity[false_class].mean()),
                    "missed_anomaly_similarity_mean": float(similarity[missed_class].mean()),
                }
            )
    return group_rows, auc_rows, class_rows


def context_diagnostics(model_name, records, gammas):
    metric_rows = []
    group_rows = []
    class_rows = []
    frame_gt = np.concatenate([record["gt_frames"] for record in records])
    snippet_gt_all = np.concatenate([record["gt_any"] for record in records])

    for gamma in gammas:
        logits = np.concatenate([record["context_logits"][gamma] for record in records])
        probabilities = torch.from_numpy(logits).softmax(dim=-1).numpy()
        alignment = 1 - probabilities[:, 0]
        margin = logits[:, 1:].max(axis=1) - logits[:, 0]
        frame_scores = np.repeat(alignment, FRAMES_PER_SNIPPET)
        metric_rows.append(
            {
                "model": model_name,
                "gamma": gamma,
                "frame_auc2": safe_auc(frame_gt, frame_scores),
                "frame_ap2": safe_ap(frame_gt, frame_scores),
                "snippet_any_auc2": safe_auc(snippet_gt_all, alignment),
                "snippet_any_ap2": safe_ap(snippet_gt_all, alignment),
            }
        )

        offset = 0
        scope_values = {}
        for record in records:
            count = record["snippet_count"]
            record_alignment = alignment[offset : offset + count]
            record_margin = margin[offset : offset + count]
            offset += count
            scopes = {
                "true_high": record["group"] == "true_high",
                "missed_anomaly": record["group"] == "missed_anomaly",
                "true_low": record["group"] == "true_low",
                "false_high_abnormal_video": (record["group"] == "false_high")
                & (record["class_name"] != "Normal"),
                "false_high_normal_video": (record["group"] == "false_high")
                & (record["class_name"] == "Normal"),
            }
            for scope, mask in scopes.items():
                if not mask.any():
                    continue
                key = scope
                scope_values.setdefault(key, {"alignment": [], "margin": []})
                scope_values[key]["alignment"].extend(record_alignment[mask].tolist())
                scope_values[key]["margin"].extend(record_margin[mask].tolist())

            if record["class_name"] != "Normal":
                for scope in ["false_high", "missed_anomaly"]:
                    mask = record["group"] == scope
                    if mask.any():
                        class_rows.append(
                            {
                                "model": model_name,
                                "gamma": gamma,
                                "class_name": record["class_name"],
                                "scope": scope,
                                "count": int(mask.sum()),
                                "alignment_sum": float(record_alignment[mask].sum()),
                                "margin_sum": float(record_margin[mask].sum()),
                            }
                        )

        for scope, values in scope_values.items():
            group_rows.append(
                {
                    "model": model_name,
                    "gamma": gamma,
                    "scope": scope,
                    "count": len(values["alignment"]),
                    "alignment_mean": float(np.mean(values["alignment"])),
                    "anomaly_margin_mean": float(np.mean(values["margin"])),
                }
            )

    aggregated_class_rows = []
    keys = sorted(
        {(row["model"], row["gamma"], row["class_name"], row["scope"]) for row in class_rows}
    )
    for key in keys:
        rows = [
            row
            for row in class_rows
            if (row["model"], row["gamma"], row["class_name"], row["scope"]) == key
        ]
        count = sum(row["count"] for row in rows)
        aggregated_class_rows.append(
            {
                "model": key[0],
                "gamma": key[1],
                "class_name": key[2],
                "scope": key[3],
                "count": count,
                "alignment_mean": sum(row["alignment_sum"] for row in rows) / count,
                "anomaly_margin_mean": sum(row["margin_sum"] for row in rows) / count,
            }
        )
    return metric_rows, group_rows, aggregated_class_rows


def print_model_results(model_name, head_rows, bank_auc_rows, context_metric_rows, context_group_rows):
    print("\nClassifier versus feature")
    for row in head_rows:
        print(
            f"  {row['scope']} | {row['stage']}: probe={row['probe_auc_mean']:.2f} | "
            f"same-fold head={row['same_fold_classifier_auc_mean']:.2f} | "
            f"delta={row['probe_minus_classifier_auc']:+.2f} | "
            f"kNN purity={row['knn_anomaly_normal_purity']:.3f}"
        )
    print("\nNormal memory bank (diagnostic only)")
    for row in bank_auc_rows:
        if row["comparison"] == "false_high_abnormal_vs_missed_anomaly":
            print(
                f"  {row['bank_stage']}: AUC(FH more normal)={row['auc']:.2f} | "
                f"mean gap FH-MA={row['mean_gap_false_high_minus_missed']:+.4f}"
            )
    print("\nVideo-context gamma ablation")
    group_lookup = {
        (row["gamma"], row["scope"]): row for row in context_group_rows if row["model"] == model_name
    }
    for row in context_metric_rows:
        false_high = group_lookup[(row["gamma"], "false_high_abnormal_video")]
        missed = group_lookup[(row["gamma"], "missed_anomaly")]
        print(
            f"  gamma={row['gamma']:.2f}: AUC2={row['frame_auc2']:.2f} AP2={row['frame_ap2']:.2f} | "
            f"FH-align={false_high['alignment_mean']:.3f} | MA-align={missed['alignment_mean']:.3f} | "
            f"MA-FH={missed['alignment_mean'] - false_high['alignment_mean']:+.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose VadCLIP classifier, Normal bank, and video context.")
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_description.csv")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--output-dir", default="../../code/results/analyze/failure_source_diagnostics")
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
    parser.add_argument("--context-gammas", default="0,0.25,0.5,0.75,1", type=parse_float_list)
    parser.add_argument("--normal-prototypes", default=32, type=int)
    parser.add_argument("--max-normal-train-rows", default=500, type=int)
    parser.add_argument("--max-normal-bank-features", default=50000, type=int)
    parser.add_argument("--normal-batch-size", default=8, type=int)
    parser.add_argument("--max-samples-per-group", default=1000, type=int)
    parser.add_argument("--knn-neighbors", default=5, type=int)
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


def main():
    args = parse_args()
    with open(args.models_json, "r", encoding="utf-8") as handle:
        configs = json.load(handle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = np.load(args.gt_path)
    test_dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    normal_dataset = NormalTrainDataset(
        args.train_list,
        args.feature_root,
        args.visual_length,
        args.max_normal_train_rows,
        args.seed,
    )
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.normal_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "normal_bank_sampled_rows.json").open("w", encoding="utf-8") as handle:
        json.dump(normal_dataset.rows, handle, ensure_ascii=False, indent=2)

    all_head = []
    all_head_groups = []
    all_bank_groups = []
    all_bank_auc = []
    all_bank_class = []
    all_bank_metadata = []
    all_context_metrics = []
    all_context_groups = []
    all_context_classes = []
    sanity_rows = []
    raw_bank_cache = None
    raw_bank_cache_metadata = None

    print("Failure-source diagnostics (analysis only; no training or loss changes)")
    print(f"Device: {device}")
    print(f"Models: {len(configs)}")
    print(f"Sampled Normal train rows: {len(normal_dataset)}")
    print(f"Normal prototypes per stage: {args.normal_prototypes}")
    print(f"Context gammas: {args.context_gammas}")

    for model_index, config in enumerate(configs):
        model_name = config["name"]
        print("\n" + "=" * 80)
        print(f"Analyzing {model_name}: {config['model_path']}")
        model = load_model(args, config, device)
        records, gamma_error = run_test_inference(model, test_loader, args, config, gt, device)
        metadata, test_features = flatten_test_records(records)
        head_rows, head_group_rows = classifier_diagnostics(
            model_name, test_features, metadata, args
        )
        all_head.extend(head_rows)
        all_head_groups.extend(head_group_rows)

        normal_features = collect_normal_train_features(model, normal_loader, args, device)
        banks = {}
        for stage in BANK_STAGES:
            if stage == "raw_clip" and raw_bank_cache is not None:
                prototypes = raw_bank_cache
                fitted_count, inertia = raw_bank_cache_metadata
            else:
                prototypes, fitted_count, inertia = fit_normal_bank(
                    normal_features[stage],
                    args.normal_prototypes,
                    args.max_normal_bank_features,
                    args.seed if stage == "raw_clip" else args.seed + model_index,
                )
                if stage == "raw_clip":
                    raw_bank_cache = prototypes
                    raw_bank_cache_metadata = (fitted_count, inertia)
            banks[stage] = prototypes
            all_bank_metadata.append(
                {
                    "model": model_name,
                    "bank_stage": stage,
                    "normal_train_rows": len(normal_dataset),
                    "normal_train_unique_videos": len(
                        {row["video_id"] for row in normal_dataset.rows}
                    ),
                    "normal_features_used": fitted_count,
                    "prototype_count": len(prototypes),
                    "kmeans_inertia": inertia,
                }
            )
        np.savez_compressed(
            output_dir / f"normal_visual_bank_{model_name}.npz",
            **{stage: prototypes.astype(np.float32) for stage, prototypes in banks.items()},
        )
        bank_groups, bank_auc, bank_class = normal_bank_diagnostics(
            model_name, test_features, metadata, banks
        )
        all_bank_groups.extend(bank_groups)
        all_bank_auc.extend(bank_auc)
        all_bank_class.extend(bank_class)

        context_metrics, context_groups, context_classes = context_diagnostics(
            model_name, records, args.context_gammas
        )
        all_context_metrics.extend(context_metrics)
        all_context_groups.extend(context_groups)
        all_context_classes.extend(context_classes)
        sanity_rows.append(
            {
                "model": model_name,
                "gamma_1_max_absolute_logit_error_vs_original": gamma_error,
                "status": "ok" if gamma_error < 1e-4 else "check_implementation",
            }
        )
        print_model_results(
            model_name, head_rows, bank_auc, context_metrics, context_groups
        )
        print(f"\nGamma=1 reproduction max error: {gamma_error:.8f}")

        del model, records, metadata, test_features, normal_features, banks
        if device == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "classifier_vs_feature.csv", all_head)
    write_csv(output_dir / "classifier_group_summary.csv", all_head_groups)
    write_csv(output_dir / "normal_bank_group_similarity.csv", all_bank_groups)
    write_csv(output_dir / "normal_bank_separation_auc.csv", all_bank_auc)
    write_csv(output_dir / "normal_bank_per_class.csv", all_bank_class)
    write_csv(output_dir / "normal_bank_metadata.csv", all_bank_metadata)
    write_csv(output_dir / "context_gamma_metrics.csv", all_context_metrics)
    write_csv(output_dir / "context_gamma_group_summary.csv", all_context_groups)
    write_csv(output_dir / "context_gamma_per_class.csv", all_context_classes)
    write_csv(output_dir / "context_gamma_sanity_check.csv", sanity_rows)
    with (output_dir / "models_config.json").open("w", encoding="utf-8") as handle:
        json.dump(configs, handle, ensure_ascii=False, indent=2)

    print("\nSaved outputs:")
    for filename in [
        "classifier_vs_feature.csv",
        "classifier_group_summary.csv",
        "normal_bank_group_similarity.csv",
        "normal_bank_separation_auc.csv",
        "normal_bank_per_class.csv",
        "normal_bank_metadata.csv",
        "normal_bank_sampled_rows.json",
        "normal_visual_bank_<model>.npz",
        "context_gamma_metrics.csv",
        "context_gamma_group_summary.csv",
        "context_gamma_per_class.csv",
        "context_gamma_sanity_check.csv",
        "models_config.json",
    ]:
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
