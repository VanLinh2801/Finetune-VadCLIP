import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from ucf_analyze_score_separation import (
    BRANCHES,
    LABEL_MAP,
    UCFRelativeTestDataset,
    get_prompt_text_for_config,
    get_prompt_weights_for_config,
    load_model,
    make_lengths,
)
from utils.tools import get_batch_mask


FRAMES_PER_SNIPPET = 16
GROUPS = ["true_high", "false_high", "missed_anomaly", "true_low"]
STAGES = ["raw_clip", "temporal", "final_visual"]
CLASS_NAMES = list(LABEL_MAP.keys())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(denominator, 1e-8))


def selected_mask(scores, ratio):
    scores = np.asarray(scores)
    count = min(scores.size, max(1, int(np.ceil(scores.size * ratio))))
    indices = np.argsort(-scores, kind="stable")[:count]
    mask = np.zeros(scores.size, dtype=bool)
    mask[indices] = True
    return mask


def snippet_gt(gt_frames, snippet_count):
    grouped = np.asarray(gt_frames).reshape(snippet_count, FRAMES_PER_SNIPPET)
    fraction = grouped.mean(axis=1).astype(np.float32)
    return (fraction > 0).astype(np.int64), fraction


def assign_groups(gt_any, high_mask):
    gt_mask = np.asarray(gt_any).astype(bool)
    groups = np.full(gt_mask.size, "true_low", dtype=object)
    groups[high_mask & gt_mask] = "true_high"
    groups[high_mask & ~gt_mask] = "false_high"
    groups[~high_mask & gt_mask] = "missed_anomaly"
    return groups


def capture_forward(model, visual, padding_mask, prompt_text, lengths, prompt_weights, model_type):
    captured = {}

    def temporal_hook(_module, _inputs, output):
        temporal = output[0] if isinstance(output, tuple) else output
        captured["temporal"] = temporal.permute(1, 0, 2).detach()

    hook = model.temporal.register_forward_hook(temporal_hook)
    try:
        if model_type == "multi_prompt":
            outputs = model(
                visual,
                padding_mask,
                prompt_text,
                lengths,
                return_visual=True,
                return_prompt_logits=True,
                prompt_weights=prompt_weights,
            )
            _, logits1, logits2, prompt_logits, final_visual = outputs
        else:
            _, logits1, logits2, final_visual = model(
                visual,
                padding_mask,
                prompt_text,
                lengths,
                return_visual=True,
            )
            prompt_logits = None
    finally:
        hook.remove()

    return {
        "raw_clip": visual.detach(),
        "temporal": captured["temporal"],
        "final_visual": final_visual.detach(),
        "logits1": logits1.detach(),
        "logits2": logits2.detach(),
        "prompt_logits": prompt_logits.detach() if prompt_logits is not None else None,
    }


def flatten_valid(tensor, snippet_count):
    return tensor.reshape(-1, tensor.shape[-1])[:snippet_count]


def run_inference(model, dataloader, args, config, gt, device):
    model_type = config.get("model_type", "baseline")
    prompt_text = get_prompt_text_for_config(config)
    prompt_weights = get_prompt_weights_for_config(config, device)
    records = []
    gt_cursor = 0

    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            video_id = item[3][0] if isinstance(item[3], (list, tuple)) else item[3]
            snippet_count = int(item[2])
            if snippet_count < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = make_lengths(snippet_count, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            outputs = capture_forward(
                model,
                visual,
                padding_mask,
                prompt_text,
                lengths,
                prompt_weights,
                model_type,
            )

            logits1 = flatten_valid(outputs["logits1"], snippet_count)
            logits2 = flatten_valid(outputs["logits2"], snippet_count)
            classifier = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy()
            class_probabilities = logits2.softmax(dim=-1).cpu().numpy()
            alignment = 1 - class_probabilities[:, 0]
            frame_count = snippet_count * FRAMES_PER_SNIPPET
            gt_frames = gt[gt_cursor : gt_cursor + frame_count]
            gt_cursor += frame_count
            gt_any, gt_fraction = snippet_gt(gt_frames, snippet_count)
            high_mask = selected_mask(classifier, args.top_k_ratio)

            prompt_logits = outputs["prompt_logits"]
            if prompt_logits is not None:
                prompt_logits = prompt_logits.reshape(
                    -1, prompt_logits.shape[-2], prompt_logits.shape[-1]
                )[:snippet_count].cpu().numpy()

            records.append(
                {
                    "video_id": str(video_id),
                    "class_name": str(label),
                    "snippet_count": snippet_count,
                    "gt_any": gt_any,
                    "gt_fraction": gt_fraction,
                    "group": assign_groups(gt_any, high_mask),
                    "classifier": classifier,
                    "alignment": alignment,
                    "class_logits": logits2.cpu().numpy(),
                    "class_probabilities": class_probabilities,
                    "prompt_logits": prompt_logits,
                    "prompt_text": prompt_text,
                    "prompt_weights": prompt_weights.detach().cpu().numpy()
                    if prompt_weights is not None
                    else None,
                    "features": {
                        stage: flatten_valid(outputs[stage], snippet_count).cpu().to(torch.float16).numpy()
                        for stage in STAGES
                    },
                }
            )

    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)} frames.")
    return records


def flatten_records(records):
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
        "gt_fraction": np.concatenate([record["gt_fraction"] for record in records]),
        "group": np.concatenate([record["group"] for record in records]),
        "classifier": np.concatenate([record["classifier"] for record in records]),
        "alignment": np.concatenate([record["alignment"] for record in records]),
        "class_logits": np.concatenate([record["class_logits"] for record in records]),
        "class_probabilities": np.concatenate(
            [record["class_probabilities"] for record in records]
        ),
    }
    features = {
        stage: np.concatenate([record["features"][stage] for record in records]).astype(np.float32)
        for stage in STAGES
    }
    return metadata, features


def sample_indices(groups, max_per_group, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for group in GROUPS:
        candidates = np.flatnonzero(groups == group)
        if candidates.size > max_per_group:
            candidates = rng.choice(candidates, max_per_group, replace=False)
        indices.extend(candidates.tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def centroid_metrics(model_name, features, groups):
    rows = []
    centroids = {}
    for group in GROUPS:
        subset = normalize(features[groups == group])
        if subset.size == 0:
            continue
        centroid = subset.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        centroids[group] = centroid
        rows.append(
            {
                "model": model_name,
                "group_a": group,
                "group_b": group,
                "centroid_cosine": 1.0,
                "within_group_similarity_to_centroid": float((subset @ centroid).mean()),
            }
        )
    for left_index, left in enumerate(GROUPS):
        for right in GROUPS[left_index + 1 :]:
            if left not in centroids or right not in centroids:
                continue
            similarity = cosine(centroids[left], centroids[right])
            rows.append(
                {
                    "model": model_name,
                    "group_a": left,
                    "group_b": right,
                    "centroid_cosine": similarity,
                    "within_group_similarity_to_centroid": np.nan,
                }
            )
    return rows, centroids


def knn_metrics(model_name, stage, features, groups, sample_indices_, neighbors):
    sampled_features = normalize(features[sample_indices_])
    sampled_groups = groups[sample_indices_]
    gt_labels = np.isin(sampled_groups, ["true_high", "missed_anomaly"]).astype(np.int64)
    count = min(neighbors + 1, len(sample_indices_))
    model = NearestNeighbors(n_neighbors=count, metric="cosine")
    model.fit(sampled_features)
    neighbor_indices = model.kneighbors(sampled_features, return_distance=False)[:, 1:]
    label_purity = np.mean(gt_labels[neighbor_indices] == gt_labels[:, None])
    group_purity = np.mean(sampled_groups[neighbor_indices] == sampled_groups[:, None])
    rows = [
        {
            "model": model_name,
            "stage": stage,
            "scope": "all_sampled",
            "sample_count": len(sample_indices_),
            "k": neighbors,
            "anomaly_normal_knn_purity": float(label_purity),
            "four_group_knn_purity": float(group_purity),
        }
    ]
    for group in GROUPS:
        mask = sampled_groups == group
        if not mask.any():
            continue
        rows.append(
            {
                "model": model_name,
                "stage": stage,
                "scope": group,
                "sample_count": int(mask.sum()),
                "k": neighbors,
                "anomaly_normal_knn_purity": float(
                    np.mean(gt_labels[neighbor_indices[mask]] == gt_labels[mask, None])
                ),
                "four_group_knn_purity": float(
                    np.mean(sampled_groups[neighbor_indices[mask]] == group)
                ),
            }
        )
    return rows


def grouped_linear_probe(model_name, stage, features, gt_any, video_ids, sample_indices_):
    x = normalize(features[sample_indices_])
    y = gt_any[sample_indices_]
    groups = video_ids[sample_indices_]
    splitter = GroupKFold(n_splits=3)
    auc_values = []
    for train_indices, test_indices in splitter.split(x, y, groups):
        if np.unique(y[train_indices]).size < 2 or np.unique(y[test_indices]).size < 2:
            continue
        classifier = LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear")
        classifier.fit(x[train_indices], y[train_indices])
        probabilities = classifier.predict_proba(x[test_indices])[:, 1]
        auc_values.append(roc_auc_score(y[test_indices], probabilities) * 100)
    return {
        "model": model_name,
        "stage": stage,
        "sample_count": len(sample_indices_),
        "folds": len(auc_values),
        "grouped_by_video_linear_probe_auc_mean": float(np.mean(auc_values))
        if auc_values
        else np.nan,
        "grouped_by_video_linear_probe_auc_std": float(np.std(auc_values))
        if auc_values
        else np.nan,
    }


def paired_video_metrics(model_name, records):
    rows = []
    pairs = [
        ("false_high", "true_high"),
        ("false_high", "true_low"),
        ("missed_anomaly", "true_high"),
        ("missed_anomaly", "false_high"),
    ]
    for record in records:
        if record["class_name"] == "Normal":
            continue
        for stage in STAGES:
            features = normalize(record["features"][stage])
            centroids = {}
            for group in GROUPS:
                subset = features[record["group"] == group]
                if subset.size:
                    centroid = subset.mean(axis=0)
                    centroid /= max(np.linalg.norm(centroid), 1e-8)
                    centroids[group] = centroid
            for left, right in pairs:
                if left in centroids and right in centroids:
                    rows.append(
                        {
                            "model": model_name,
                            "video_id": record["video_id"],
                            "class_name": record["class_name"],
                            "stage": stage,
                            "group_a": left,
                            "group_b": right,
                            "centroid_cosine": cosine(centroids[left], centroids[right]),
                        }
                    )
    return rows


def stage_drift_metrics(model_name, features, groups):
    rows = []
    for left, right in [("raw_clip", "temporal"), ("temporal", "final_visual"), ("raw_clip", "final_visual")]:
        similarity = np.sum(normalize(features[left]) * normalize(features[right]), axis=1)
        for group in GROUPS:
            mask = groups == group
            rows.append(
                {
                    "model": model_name,
                    "from_stage": left,
                    "to_stage": right,
                    "group": group,
                    "count": int(mask.sum()),
                    "same_snippet_cosine_mean": float(similarity[mask].mean()),
                    "same_snippet_cosine_std": float(similarity[mask].std()),
                }
            )
    return rows


def prompt_attribution(record, snippet_index, predicted_class_index):
    if record["prompt_logits"] is None:
        return "", -1, np.nan, np.nan
    class_prompt_logits = record["prompt_logits"][snippet_index, predicted_class_index]
    if record["prompt_weights"] is None:
        contribution = class_prompt_logits
        weight = np.full_like(contribution, 1 / len(contribution), dtype=np.float32)
    else:
        weight = record["prompt_weights"][predicted_class_index]
        weight = weight / max(weight.sum(), 1e-8)
        contribution = class_prompt_logits + np.log(np.maximum(weight, 1e-8))
    prompt_index = int(np.argmax(contribution))
    prompts = record["prompt_text"][predicted_class_index]
    return str(prompts[prompt_index]), prompt_index, float(class_prompt_logits[prompt_index]), float(weight[prompt_index])


def attribution_rows(model_name, records):
    detail_rows = []
    confusion = {}
    for record in records:
        true_class_index = CLASS_NAMES.index(record["class_name"])
        probabilities = record["class_probabilities"]
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-8)), axis=1)
        anomaly_class_index = np.argmax(record["class_logits"][:, 1:], axis=1) + 1
        predicted_class_index = np.argmax(record["class_logits"], axis=1)
        anomaly_margin = (
            record["class_logits"][np.arange(record["snippet_count"]), anomaly_class_index]
            - record["class_logits"][:, 0]
        )
        for index in range(record["snippet_count"]):
            predicted_index = int(predicted_class_index[index])
            prompt_text, prompt_index, prompt_logit, prompt_weight = prompt_attribution(
                record, index, predicted_index
            )
            group = str(record["group"][index])
            predicted_name = CLASS_NAMES[predicted_index]
            key = (model_name, group, record["class_name"], predicted_name)
            confusion[key] = confusion.get(key, 0) + 1
            detail_rows.append(
                {
                    "model": model_name,
                    "video_id": record["video_id"],
                    "class_name": record["class_name"],
                    "snippet_index": index,
                    "group": group,
                    "gt_any": int(record["gt_any"][index]),
                    "gt_fraction": float(record["gt_fraction"][index]),
                    "classifier_score": float(record["classifier"][index]),
                    "alignment_score": float(record["alignment"][index]),
                    "predicted_class": predicted_name,
                    "predicted_class_probability": float(probabilities[index, predicted_index]),
                    "normal_probability": float(probabilities[index, 0]),
                    "true_video_class_probability": float(probabilities[index, true_class_index]),
                    "best_anomaly_class": CLASS_NAMES[int(anomaly_class_index[index])],
                    "anomaly_vs_normal_logit_margin": float(anomaly_margin[index]),
                    "class_entropy": float(entropy[index]),
                    "top_contributing_prompt": prompt_text,
                    "top_contributing_prompt_index": prompt_index,
                    "top_contributing_prompt_logit": prompt_logit,
                    "top_contributing_prompt_weight": prompt_weight,
                }
            )
    confusion_rows = [
        {
            "model": key[0],
            "group": key[1],
            "video_class": key[2],
            "predicted_class": key[3],
            "count": count,
        }
        for key, count in confusion.items()
    ]
    return detail_rows, confusion_rows


def summarize_attribution(detail_rows):
    output = []
    keys = sorted({(row["model"], row["group"]) for row in detail_rows})
    for model_name, group in keys:
        rows = [row for row in detail_rows if row["model"] == model_name and row["group"] == group]
        output.append(
            {
                "model": model_name,
                "group": group,
                "count": len(rows),
                "classifier_score_mean": float(np.mean([row["classifier_score"] for row in rows])),
                "alignment_score_mean": float(np.mean([row["alignment_score"] for row in rows])),
                "normal_probability_mean": float(np.mean([row["normal_probability"] for row in rows])),
                "true_video_class_probability_mean": float(
                    np.mean([row["true_video_class_probability"] for row in rows])
                ),
                "anomaly_vs_normal_margin_mean": float(
                    np.mean([row["anomaly_vs_normal_logit_margin"] for row in rows])
                ),
                "class_entropy_mean": float(np.mean([row["class_entropy"] for row in rows])),
                "predicted_normal_rate": float(
                    np.mean([row["predicted_class"] == "Normal" for row in rows])
                ),
                "predicted_video_class_rate": float(
                    np.mean([row["predicted_class"] == row["class_name"] for row in rows])
                ),
            }
        )
    return output


def save_figures(output_dir, model_name, sampled_features, sampled_groups):
    colors = {
        "true_high": "#d73027",
        "false_high": "#fdae61",
        "missed_anomaly": "#4575b4",
        "true_low": "#66bd63",
    }
    figure, axes = plt.subplots(1, len(STAGES), figsize=(18, 5))
    for axis, stage in zip(axes, STAGES):
        projected = PCA(n_components=2, random_state=42).fit_transform(
            normalize(sampled_features[stage])
        )
        for group in GROUPS:
            mask = sampled_groups == group
            axis.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=8,
                alpha=0.45,
                color=colors[group],
                label=group,
            )
        axis.set_title(stage)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].legend(markerscale=2, fontsize=8)
    figure.suptitle(f"Stage-wise PCA: {model_name}")
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_name}_stage_pca.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_model_summary(
    model_name,
    probe_rows,
    knn_rows,
    centroid_rows,
    attribution_summary,
    paired_rows,
):
    print("\nStage-wise visual separability")
    for row in probe_rows:
        knn_row = next(
            item for item in knn_rows if item["stage"] == row["stage"] and item["scope"] == "all_sampled"
        )
        print(
            f"  {model_name} | {row['stage']} | grouped-video linear probe AUC="
            f"{row['grouped_by_video_linear_probe_auc_mean']:.2f} +/- "
            f"{row['grouped_by_video_linear_probe_auc_std']:.2f} | "
            f"kNN anomaly/normal purity={knn_row['anomaly_normal_knn_purity']:.3f} | "
            f"four-group purity={knn_row['four_group_knn_purity']:.3f}"
        )
        print("    Global centroid cosine:")
        for left, right in [
            ("false_high", "true_high"),
            ("false_high", "true_low"),
            ("missed_anomaly", "true_high"),
            ("missed_anomaly", "false_high"),
        ]:
            match = next(
                (
                    item
                    for item in centroid_rows
                    if item["stage"] == row["stage"]
                    and {item["group_a"], item["group_b"]} == {left, right}
                ),
                None,
            )
            if match is not None:
                print(f"      {left} vs {right}: {match['centroid_cosine']:.3f}")
    print("\nClass attribution by classifier-defined group")
    for row in attribution_summary:
        if row["model"] == model_name:
            print(
                f"  {row['group']}: n={row['count']} | cls={row['classifier_score_mean']:.3f} | "
                f"align={row['alignment_score_mean']:.3f} | normal_prob={row['normal_probability_mean']:.3f} | "
                f"anomaly_margin={row['anomaly_vs_normal_margin_mean']:.3f} | "
                f"pred_normal={row['predicted_normal_rate']:.3f} | "
                f"pred_video_class={row['predicted_video_class_rate']:.3f}"
            )
    print("\nPaired centroid similarity within the same abnormal video")
    for stage in STAGES:
        stage_rows = [row for row in paired_rows if row["stage"] == stage]
        for left, right in [("false_high", "true_high"), ("false_high", "true_low"), ("missed_anomaly", "true_high")]:
            values = [
                row["centroid_cosine"]
                for row in stage_rows
                if row["group_a"] == left and row["group_b"] == right
            ]
            if values:
                print(f"  {stage} | {left} vs {right}: {np.mean(values):.3f} (videos={len(values)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze stage-wise VadCLIP visual features and class attribution.")
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--output-dir", default="../../code/results/analyze/visual_features")
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
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
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_centroid = []
    all_knn = []
    all_probe = []
    all_paired = []
    all_drift = []
    all_attribution_summary = []
    all_confusion = []

    print("VadCLIP visual feature diagnostics")
    print(f"Device: {device}")
    print(f"Models: {len(configs)}")
    print(f"Classifier top-k ratio defining groups: {args.top_k_ratio}")
    print("Linear probe is diagnostic-only and uses GroupKFold by video.")

    for config_index, config in enumerate(configs):
        model_name = config["name"]
        print("\n" + "=" * 80)
        print(f"Analyzing {model_name}: {config['model_path']}")
        model = load_model(args, config, device)
        records = run_inference(model, dataloader, args, config, gt, device)
        metadata, features = flatten_records(records)
        sampled_indices = sample_indices(
            metadata["group"], args.max_samples_per_group, args.seed + config_index
        )

        model_probe_rows = []
        model_knn_rows = []
        model_centroid_rows = []
        for stage in STAGES:
            centroid_rows, _ = centroid_metrics(model_name, features[stage], metadata["group"])
            for row in centroid_rows:
                row["stage"] = stage
            all_centroid.extend(centroid_rows)
            model_centroid_rows.extend(centroid_rows)
            knn_rows = knn_metrics(
                model_name,
                stage,
                features[stage],
                metadata["group"],
                sampled_indices,
                args.knn_neighbors,
            )
            all_knn.extend(knn_rows)
            model_knn_rows.extend(knn_rows)
            probe_row = grouped_linear_probe(
                model_name,
                stage,
                features[stage],
                metadata["gt_any"],
                metadata["video_id"],
                sampled_indices,
            )
            all_probe.append(probe_row)
            model_probe_rows.append(probe_row)

        paired_rows = paired_video_metrics(model_name, records)
        all_paired.extend(paired_rows)
        all_drift.extend(stage_drift_metrics(model_name, features, metadata["group"]))
        detail_rows, confusion_rows = attribution_rows(model_name, records)
        attribution_summary = summarize_attribution(detail_rows)
        write_csv(output_dir / f"snippet_class_attribution_{model_name}.csv", detail_rows)
        all_attribution_summary.extend(attribution_summary)
        all_confusion.extend(confusion_rows)

        sampled_features = {stage: features[stage][sampled_indices] for stage in STAGES}
        sampled_groups = metadata["group"][sampled_indices]
        np.savez_compressed(
            output_dir / f"{model_name}_feature_samples.npz",
            indices=sampled_indices,
            groups=sampled_groups,
            video_ids=metadata["video_id"][sampled_indices],
            snippet_indices=metadata["snippet_index"][sampled_indices],
            gt_any=metadata["gt_any"][sampled_indices],
            **{stage: sampled_features[stage].astype(np.float16) for stage in STAGES},
        )
        save_figures(output_dir, model_name, sampled_features, sampled_groups)
        print_model_summary(
            model_name,
            model_probe_rows,
            model_knn_rows,
            model_centroid_rows,
            attribution_summary,
            paired_rows,
        )

        del model, records, metadata, features
        if device == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "stage_centroid_similarity.csv", all_centroid)
    write_csv(output_dir / "stage_knn_purity.csv", all_knn)
    write_csv(output_dir / "stage_linear_probe.csv", all_probe)
    write_csv(output_dir / "within_video_group_similarity.csv", all_paired)
    write_csv(output_dir / "stage_feature_drift.csv", all_drift)
    write_csv(output_dir / "class_attribution_summary.csv", all_attribution_summary)
    write_csv(output_dir / "class_confusion_by_group.csv", all_confusion)
    with (output_dir / "models_config.json").open("w", encoding="utf-8") as handle:
        json.dump(configs, handle, ensure_ascii=False, indent=2)

    print("\nSaved outputs:")
    for filename in [
        "stage_centroid_similarity.csv",
        "stage_knn_purity.csv",
        "stage_linear_probe.csv",
        "within_video_group_similarity.csv",
        "stage_feature_drift.csv",
        "snippet_class_attribution_*.csv",
        "class_attribution_summary.csv",
        "class_confusion_by_group.csv",
        "*_feature_samples.npz",
        "*_stage_pca.png",
        "models_config.json",
    ]:
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
