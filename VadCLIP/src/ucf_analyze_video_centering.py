"""Compare all-snippet and ground-truth-normal CLIP feature means per UCF video.

The normal-only mean is an oracle diagnostic because it uses test ground truth.
Neither mean is used to train or evaluate an existing VadCLIP checkpoint.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FRAMES_PER_SNIPPET = 16
SCORE_NAMES = ("all_cosine", "normal_cosine", "all_residual", "normal_residual")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def score_to_mean(features, mean):
    feature_norm = np.linalg.norm(features, axis=1)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= 1e-8 or np.any(feature_norm <= 1e-8):
        raise ValueError("Found zero-norm CLIP feature or video mean.")
    cosine_distance = 1 - (features @ mean) / (feature_norm * mean_norm)
    residual_norm = np.linalg.norm(features - mean, axis=1)
    return cosine_distance, residual_norm


def topk_stats(labels, scores, ratio):
    count = max(1, math.ceil(len(scores) * ratio))
    selected = np.argsort(-scores, kind="stable")[:count]
    hits = int(np.asarray(labels, dtype=bool)[selected].sum())
    positives = int(np.asarray(labels, dtype=bool).sum())
    return hits / count, hits / positives if positives else None


def summarize(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "ci95_low": None, "ci95_high": None}
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"n": int(len(values)), "mean": mean, "median": float(np.median(values)),
            "ci95_low": mean - 1.96 * sem, "ci95_high": mean + 1.96 * sem}


def compare_video(features, gt_frames, top_k_ratio):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(f"Expected [snippets, 512] CLIP features, got {features.shape}.")
    if len(gt_frames) != features.shape[0] * FRAMES_PER_SNIPPET:
        raise ValueError("Feature and ground-truth frame counts disagree.")
    anomaly = np.asarray(gt_frames).reshape(-1, FRAMES_PER_SNIPPET).max(axis=1) > 0
    normal = ~anomaly
    all_mean = features.mean(axis=0)
    normal_mean = features[normal].mean(axis=0) if normal.any() else None
    all_cosine, all_residual = score_to_mean(features, all_mean)
    scores = {"all_cosine": all_cosine, "all_residual": all_residual}
    if normal_mean is not None:
        normal_cosine, normal_residual = score_to_mean(features, normal_mean)
        scores.update(normal_cosine=normal_cosine, normal_residual=normal_residual)

    result = {
        "snippet_count": len(features), "normal_count": int(normal.sum()),
        "anomaly_count": int(anomaly.sum()), "anomaly_fraction": float(anomaly.mean()),
        "all_mean_norm": float(np.linalg.norm(all_mean)),
        "normal_mean_norm": float(np.linalg.norm(normal_mean)) if normal_mean is not None else None,
        "all_mean_energy_fraction": float(np.dot(all_mean, all_mean) / np.mean(np.sum(features ** 2, axis=1))),
        "mean_cosine": float(np.dot(all_mean, normal_mean) /
                             (np.linalg.norm(all_mean) * np.linalg.norm(normal_mean)))
        if normal_mean is not None else None,
    }
    for name in SCORE_NAMES:
        if name in scores:
            result[f"auc_{name}"] = rank_auc(anomaly, scores[name])
            precision, recall = topk_stats(anomaly, scores[name], top_k_ratio)
            result[f"topk_precision_{name}"] = precision
            result[f"topk_recall_{name}"] = recall
            result[f"normal_score_mean_{name}"] = float(scores[name][normal].mean()) if normal.any() else None
            result[f"anomaly_score_mean_{name}"] = float(scores[name][anomaly].mean()) if anomaly.any() else None
        else:
            for prefix in ("auc", "topk_precision", "topk_recall", "normal_score_mean", "anomaly_score_mean"):
                result[f"{prefix}_{name}"] = None
    return all_mean, normal_mean, anomaly, scores, result


def make_figures(rows, output_dir):
    abnormal = [row for row in rows if row["label"] != "Normal" and row["auc_normal_cosine"] is not None]
    if abnormal:
        figure, axis = plt.subplots(figsize=(6, 6))
        x = [row["auc_all_cosine"] for row in abnormal]
        y = [row["auc_normal_cosine"] for row in abnormal]
        axis.scatter(x, y, s=22, alpha=0.6)
        axis.plot([0, 1], [0, 1], color="black", linewidth=0.8)
        axis.set(xlabel="AUC using all-snippet mean", ylabel="AUC using oracle normal mean",
                 xlim=(0, 1), ylim=(0, 1), title="Within-video anomaly ranking")
        figure.tight_layout()
        figure.savefig(output_dir / "01_video_auc_cosine.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4))
        change = np.asarray(y) - np.asarray(x)
        fraction = [row["anomaly_fraction"] for row in abnormal]
        axis.scatter(fraction, change, s=22, alpha=0.6)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Anomaly snippet fraction in video", ylabel="Oracle minus all-snippet AUC",
                 title="Does anomaly duration affect video centering?")
        figure.tight_layout()
        figure.savefig(output_dir / "02_oracle_gain_vs_anomaly_fraction.png", dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.top_k_ratio <= 1:
        raise ValueError("--top-k-ratio must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.test_list, newline="", encoding="utf-8-sig") as handle:
        test_rows = list(csv.DictReader(handle))
    gt = np.load(args.gt_path, mmap_mode="r")
    cursor = 0
    per_video = []
    per_snippet = []
    all_means = []
    normal_means = []
    for index, row in enumerate(test_rows):
        feature_path = Path(args.feature_root) / row["path"]
        features = np.load(feature_path)
        frame_count = features.shape[0] * FRAMES_PER_SNIPPET
        video_gt = gt[cursor:cursor + frame_count]
        cursor += frame_count
        if len(video_gt) != frame_count:
            raise ValueError(f"GT ended before {row['video_id']}.")
        mean_all, mean_normal, anomaly, scores, result = compare_video(
            features, video_gt, args.top_k_ratio)
        per_video.append({"video_id": row["video_id"], "class_name": row["label"],
                          "label": row["label"], **result})
        all_means.append(mean_all)
        normal_means.append(mean_normal if mean_normal is not None else np.full(512, np.nan))
        for snippet_index in range(len(features)):
            per_snippet.append({
                "video_id": row["video_id"], "class_name": row["label"],
                "snippet_index": snippet_index, "gt_anomaly": int(anomaly[snippet_index]),
                **{name: float(scores[name][snippet_index]) if name in scores else None
                   for name in SCORE_NAMES},
            })
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(test_rows)} videos", flush=True)
    if cursor != len(gt):
        raise ValueError(f"Consumed {cursor} GT frames, expected {len(gt)}.")

    write_csv(output_dir / "per_video.csv", per_video)
    write_csv(output_dir / "per_snippet_scores.csv", per_snippet)
    np.savez_compressed(output_dir / "video_means.npz",
                        video_ids=np.asarray([row["video_id"] for row in per_video]),
                        mean_all=np.stack(all_means), mean_normal_oracle=np.stack(normal_means),
                        normal_count=np.asarray([row["normal_count"] for row in per_video]))
    abnormal = [row for row in per_video if row["label"] != "Normal" and row["auc_all_cosine"] is not None]
    per_class = []
    for class_name in sorted({row["class_name"] for row in abnormal}):
        subset = [row for row in abnormal if row["class_name"] == class_name]
        per_class.append({
            "class_name": class_name, "videos": len(subset),
            **{f"macro_auc_{name}": float(np.mean([row[f"auc_{name}"] for row in subset]))
               for name in SCORE_NAMES},
            "oracle_minus_all_cosine_auc": float(np.mean([
                row["auc_normal_cosine"] - row["auc_all_cosine"] for row in subset])),
        })
    write_csv(output_dir / "per_class.csv", per_class)
    summary = {
        "test_videos": len(per_video), "abnormal_videos_with_both_snippet_labels": len(abnormal),
        "snippet_count": len(per_snippet), "top_k_ratio": args.top_k_ratio,
        "mean_vectors": "Raw float32 CLIP features; normal uses GT-normal snippets only",
        "normal_mean_is_oracle": True,
        "macro_video_auc": {name: summarize([row[f"auc_{name}"] for row in abnormal])
                            for name in SCORE_NAMES},
        "macro_video_topk_precision": {name: summarize([row[f"topk_precision_{name}"] for row in abnormal])
                                       for name in SCORE_NAMES},
        "oracle_minus_all_cosine_auc": summarize([
            row["auc_normal_cosine"] - row["auc_all_cosine"] for row in abnormal]),
        "oracle_better_video_fraction": float(np.mean([
            row["auc_normal_cosine"] > row["auc_all_cosine"] for row in abnormal])) if abnormal else None,
        "all_mean_energy_fraction": summarize([row["all_mean_energy_fraction"] for row in per_video]),
        "mean_all_vs_normal_cosine_abnormal_videos": summarize([
            row["mean_cosine"] for row in abnormal]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figures(per_video, output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
