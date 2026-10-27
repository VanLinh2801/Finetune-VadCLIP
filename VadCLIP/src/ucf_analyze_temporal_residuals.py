import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from ucf_analyze_score_separation import UCFRelativeTestDataset, load_model
from ucf_analyze_visual_features import normalize, run_inference, write_csv


METRICS = [
    "first_difference_l2",
    "second_difference_l2",
    "cosine_change_previous",
    "local_residual_l2",
    "local_variance",
    "change_point_cosine",
]


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(roc_auc_score(labels, scores) * 100)


def safe_ap(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(average_precision_score(labels, scores) * 100)


def safe_corr(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def normalized_mean(features):
    value = np.asarray(features, dtype=np.float32).mean(axis=0, keepdims=True)
    return normalize(value)[0]


def temporal_descriptors(features, radius):
    """Compute inexpensive motion proxies from an existing CLIP feature sequence."""
    x = normalize(features)
    count = len(x)
    output = {metric: np.full(count, np.nan, dtype=np.float32) for metric in METRICS}

    if count > 1:
        first = x[1:] - x[:-1]
        output["first_difference_l2"][1:] = np.linalg.norm(first, axis=1)
        output["cosine_change_previous"][1:] = 1.0 - np.sum(x[1:] * x[:-1], axis=1)

    if count > 2:
        second = (x[2:] - x[1:-1]) - (x[1:-1] - x[:-2])
        output["second_difference_l2"][2:] = np.linalg.norm(second, axis=1)

    for index in range(count):
        start = max(0, index - radius)
        end = min(count, index + radius + 1)
        neighbor_indices = [item for item in range(start, end) if item != index]
        if neighbor_indices:
            neighbors = x[neighbor_indices]
            neighbor_mean = normalized_mean(neighbors)
            output["local_residual_l2"][index] = np.linalg.norm(x[index] - neighbor_mean)

            local_window = x[start:end]
            local_mean = normalized_mean(local_window)
            output["local_variance"][index] = np.mean(
                np.sum((local_window - local_mean) ** 2, axis=1)
            )

        left = x[max(0, index - radius) : index]
        right = x[index + 1 : min(count, index + radius + 1)]
        if len(left) and len(right):
            left_mean = normalized_mean(left)
            right_mean = normalized_mean(right)
            output["change_point_cosine"][index] = 1.0 - np.dot(left_mean, right_mean)

    return output


def detailed_scope(group, class_name):
    if group == "false_high":
        return "false_high_normal_video" if class_name == "Normal" else "false_high_abnormal_video"
    if group == "true_low":
        return "true_low_normal_video" if class_name == "Normal" else "true_low_abnormal_video"
    return group


def flatten_records(model_name, records, descriptor_cache, radius):
    rows = []
    for record in records:
        video_id = record["video_id"]
        if video_id not in descriptor_cache:
            descriptor_cache[video_id] = temporal_descriptors(
                record["features"]["raw_clip"].astype(np.float32), radius
            )
        descriptors = descriptor_cache[video_id]
        for index in range(record["snippet_count"]):
            row = {
                "model": model_name,
                "video_id": video_id,
                "class_name": record["class_name"],
                "snippet_index": index,
                "gt_any": int(record["gt_any"][index]),
                "gt_fraction": float(record["gt_fraction"][index]),
                "group": record["group"][index],
                "scope": detailed_scope(record["group"][index], record["class_name"]),
                "classifier_score": float(record["classifier"][index]),
                "alignment_score": float(record["alignment"][index]),
            }
            for metric in METRICS:
                row[metric] = float(descriptors[metric][index])
            rows.append(row)
    return rows


def summarize_groups(rows):
    output = []
    scopes = sorted({row["scope"] for row in rows})
    for metric in METRICS:
        for scope in scopes:
            values = np.asarray(
                [row[metric] for row in rows if row["scope"] == scope], dtype=np.float32
            )
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            output.append(
                {
                    "model": rows[0]["model"],
                    "metric": metric,
                    "scope": scope,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "p10": float(np.percentile(values, 10)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                }
            )
    return output


def comparison_masks(rows):
    gt = np.asarray([row["gt_any"] for row in rows], dtype=np.int64)
    scopes = np.asarray([row["scope"] for row in rows], dtype=object)
    false_abnormal = scopes == "false_high_abnormal_video"
    true_high = scopes == "true_high"
    missed = scopes == "missed_anomaly"
    anomaly = gt == 1
    background = gt == 0
    return {
        "true_high_vs_false_high_abnormal": (true_high, false_abnormal),
        "missed_anomaly_vs_false_high_abnormal": (missed, false_abnormal),
        "all_anomaly_vs_false_high_abnormal": (anomaly, false_abnormal),
        "all_anomaly_vs_all_background": (anomaly, background),
    }


def summarize_separation(rows):
    output = []
    comparisons = comparison_masks(rows)
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float32)
        for name, (positive, negative) in comparisons.items():
            pair = positive | negative
            labels = positive[pair].astype(np.int64)
            scores = values[pair]
            output.append(
                {
                    "model": rows[0]["model"],
                    "metric": metric,
                    "comparison": name,
                    "positive_count": int(positive.sum()),
                    "negative_count": int(negative.sum()),
                    "positive_mean": float(np.nanmean(values[positive])),
                    "negative_mean": float(np.nanmean(values[negative])),
                    "mean_gap": float(np.nanmean(values[positive]) - np.nanmean(values[negative])),
                    "auc": safe_auc(labels, scores),
                    "ap": safe_ap(labels, scores),
                }
            )
    return output


def summarize_per_class(rows):
    output = []
    class_names = sorted({row["class_name"] for row in rows} - {"Normal"})
    for class_name in class_names:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        scopes = np.asarray([row["scope"] for row in class_rows], dtype=object)
        anomaly = np.asarray([row["gt_any"] for row in class_rows], dtype=bool)
        false_high = scopes == "false_high_abnormal_video"
        missed = scopes == "missed_anomaly"
        if not false_high.any():
            continue
        for metric in METRICS:
            values = np.asarray([row[metric] for row in class_rows], dtype=np.float32)
            for comparison, positive in [
                ("all_anomaly_vs_false_high", anomaly),
                ("missed_anomaly_vs_false_high", missed),
            ]:
                if not positive.any():
                    continue
                pair = positive | false_high
                labels = positive[pair].astype(np.int64)
                output.append(
                    {
                        "model": rows[0]["model"],
                        "class_name": class_name,
                        "metric": metric,
                        "comparison": comparison,
                        "positive_count": int(positive.sum()),
                        "false_high_count": int(false_high.sum()),
                        "auc": safe_auc(labels, values[pair]),
                        "ap": safe_ap(labels, values[pair]),
                        "positive_mean": float(np.nanmean(values[positive])),
                        "false_high_mean": float(np.nanmean(values[false_high])),
                    }
                )
    return output


def summarize_correlations(rows):
    output = []
    class_names = np.asarray([row["class_name"] for row in rows], dtype=object)
    gt = np.asarray([row["gt_any"] for row in rows], dtype=np.int64)
    classifier = np.asarray([row["classifier_score"] for row in rows], dtype=np.float32)
    alignment = np.asarray([row["alignment_score"] for row in rows], dtype=np.float32)
    masks = {
        "all_test": np.ones(len(rows), dtype=bool),
        "abnormal_videos_only": class_names != "Normal",
        "anomaly_snippets": gt == 1,
        "background_snippets": gt == 0,
    }
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float32)
        for scope, mask in masks.items():
            output.append(
                {
                    "model": rows[0]["model"],
                    "metric": metric,
                    "scope": scope,
                    "classifier_correlation": safe_corr(values[mask], classifier[mask]),
                    "alignment_correlation": safe_corr(values[mask], alignment[mask]),
                }
            )
    return output


def print_summary(model_name, group_rows, separation_rows):
    print("\n" + "=" * 88)
    print(f"Temporal residual diagnostics: {model_name}")
    print("AUC > 50 means the temporal metric is larger on anomaly than false-high.")
    lookup = {(row["metric"], row["scope"]): row for row in group_rows}
    separation = {
        (row["metric"], row["comparison"]): row for row in separation_rows
    }
    print(
        f"{'metric':30s} {'true_high':>10s} {'false_high':>11s} "
        f"{'missed':>10s} {'AUC all/FH':>11s} {'AUC missed/FH':>14s}"
    )
    for metric in METRICS:
        true_high = lookup.get((metric, "true_high"), {}).get("mean", np.nan)
        false_high = lookup.get((metric, "false_high_abnormal_video"), {}).get("mean", np.nan)
        missed = lookup.get((metric, "missed_anomaly"), {}).get("mean", np.nan)
        all_auc = separation[(metric, "all_anomaly_vs_false_high_abnormal")]["auc"]
        missed_auc = separation[(metric, "missed_anomaly_vs_false_high_abnormal")]["auc"]
        print(
            f"{metric:30s} {true_high:10.4f} {false_high:11.4f} "
            f"{missed:10.4f} {all_auc:11.2f} {missed_auc:14.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze temporal residuals derived from existing UCF CLIP features."
    )
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--output-dir", default="../../code/results/analyze/temporal_residual_diagnostics")
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
    parser.add_argument("--local-radius", default=2, type=int)
    parser.add_argument("--save-snippet-details", action="store_true")
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
    if args.local_radius < 1:
        raise ValueError("--local-radius must be at least 1")
    with open(args.models_json, "r", encoding="utf-8") as handle:
        configs = json.load(handle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = np.load(args.gt_path)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Temporal residual diagnostics")
    print("Device:", device)
    print("Models:", len(configs))
    print("Local radius:", args.local_radius)
    print("No video re-encoding is performed; descriptors use existing CLIP .npy features.")

    descriptor_cache = {}
    all_detail = []
    all_groups = []
    all_separation = []
    all_per_class = []
    all_correlations = []

    for config in configs:
        model_name = config["name"]
        print(f"\nLoading model: {model_name}")
        model = load_model(args, config, device)
        records = run_inference(model, dataloader, args, config, gt, device)
        rows = flatten_records(model_name, records, descriptor_cache, args.local_radius)
        group_rows = summarize_groups(rows)
        separation_rows = summarize_separation(rows)

        if args.save_snippet_details:
            all_detail.extend(rows)
        all_groups.extend(group_rows)
        all_separation.extend(separation_rows)
        all_per_class.extend(summarize_per_class(rows))
        all_correlations.extend(summarize_correlations(rows))
        print_summary(model_name, group_rows, separation_rows)

        del model, records
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_snippet_details:
        write_csv(output_dir / "temporal_residual_snippet_metrics.csv", all_detail)
    write_csv(output_dir / "temporal_residual_group_summary.csv", all_groups)
    write_csv(output_dir / "temporal_residual_separation.csv", all_separation)
    write_csv(output_dir / "temporal_residual_per_class.csv", all_per_class)
    write_csv(output_dir / "temporal_residual_score_correlation.csv", all_correlations)
    (output_dir / "models_config.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "feature_source": "existing CLIP .npy features",
                "local_radius": args.local_radius,
                "top_k_ratio": args.top_k_ratio,
                "metrics": METRICS,
                "ground_truth_usage": "diagnostics only",
                "snippet_details_saved": args.save_snippet_details,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nSaved temporal residual diagnostics to:", output_dir)


if __name__ == "__main__":
    main()
