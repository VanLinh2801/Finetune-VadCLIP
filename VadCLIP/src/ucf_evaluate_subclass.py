"""Compare a subclass checkpoint against the self-trained VadCLIP baseline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model import CLIPVAD
from model_subclass import (
    CLASS_NAMES, aggregate_video_frames, load_subclass_bank,
    subclass_group_mask,
)
from ucf_evaluate import UCFRelativeTestDataset, evaluate_model, print_metrics
from ucf_train_subclass import make_model
from utils.tools import get_batch_mask
from utils.ucf_detectionMAP import getDetectionMAP as dmAP


def evaluate_subclass(model, loader, args, groups, device):
    gt = np.load(args.gt_path)
    segments = np.load(args.gt_segment_path, allow_pickle=True)
    labels = np.load(args.gt_label_path, allow_pickle=True)
    model.eval()
    classifier_frames, alignment_frames = [], []
    classifier_by_video, alignment_by_video, gt_by_video, video_labels = [], [], [], []
    detection_scores = []
    group_mask = subclass_group_mask(groups, device)
    max_subclasses = group_mask.shape[1]
    selection_counts = np.zeros((len(CLASS_NAMES), max_subclasses), dtype=int)
    weight_sums = np.zeros_like(selection_counts, dtype=float)
    true_selection_counts = np.zeros_like(selection_counts)
    true_weight_sums = np.zeros_like(weight_sums)
    true_video_counts = np.zeros(len(CLASS_NAMES), dtype=int)
    cursor = 0
    with torch.no_grad():
        for features, label, length, _ in loader:
            features = features.squeeze(0).to(device)
            length = int(length.item())
            if length < args.visual_length:
                features = features.unsqueeze(0)
            n_chunks = (length + args.visual_length - 1) // args.visual_length
            features = features[:n_chunks]
            remaining = length
            lengths = []
            for _ in range(n_chunks):
                chunk_length = min(remaining, args.visual_length)
                lengths.append(chunk_length)
                remaining -= chunk_length
            mask = get_batch_mask(torch.tensor(lengths), args.visual_length).to(device)
            _, logits1, subclass_logits = model(features, mask, groups, torch.tensor(lengths))
            flat1 = logits1.reshape(-1)[:length]
            flat_sub = subclass_logits.reshape(-1, len(CLASS_NAMES), max_subclasses)[:length]
            class_logits, weights, _ = aggregate_video_frames(
                flat_sub, args.aggregation, args.temperature, group_mask
            )
            best = weights.argmax(dim=-1).cpu().numpy()
            selection_counts[np.arange(len(CLASS_NAMES)), best] += 1
            weight_sums += weights.cpu().numpy()
            true_class = CLASS_NAMES.index(label[0])
            true_selection_counts[true_class, best[true_class]] += 1
            true_weight_sums[true_class] += weights[true_class].cpu().numpy()
            true_video_counts[true_class] += 1
            classifier = torch.sigmoid(flat1).cpu().numpy()
            alignment = (1 - class_logits.softmax(dim=-1)[:, 0]).cpu().numpy()
            classifier = np.repeat(classifier, 16)
            alignment = np.repeat(alignment, 16)
            gt_slice = gt[cursor:cursor + len(classifier)]
            if len(gt_slice) != len(classifier):
                raise ValueError("Ground truth ends before test scores")
            cursor += len(classifier)
            classifier_frames.append(classifier)
            alignment_frames.append(alignment)
            classifier_by_video.append(classifier)
            alignment_by_video.append(alignment)
            gt_by_video.append(gt_slice)
            video_labels.append(label[0])
            detection_scores.append(np.repeat(class_logits.softmax(dim=-1).cpu().numpy(), 16, axis=0))
    if cursor != len(gt):
        raise ValueError(f"Frame count mismatch: scores={cursor}, gt={len(gt)}")
    classifier_frames = np.concatenate(classifier_frames)
    alignment_frames = np.concatenate(alignment_frames)
    auc1 = roc_auc_score(gt, classifier_frames) * 100
    auc2 = roc_auc_score(gt, alignment_frames) * 100
    best_by_video = classifier_by_video if auc1 >= auc2 else alignment_by_video
    abnormal_gt = np.concatenate([truth for truth, label in zip(gt_by_video, video_labels) if label != "Normal"])
    abnormal_scores = np.concatenate([score for score, label in zip(best_by_video, video_labels) if label != "Normal"])
    dmap, iou = dmAP(detection_scores, segments, labels, excludeNormal=False)
    metrics = {
        "classifier_auc": float(auc1),
        "classifier_ap": float(average_precision_score(gt, classifier_frames) * 100),
        "alignment_auc": float(auc2),
        "alignment_ap": float(average_precision_score(gt, alignment_frames) * 100),
        "best_auc": float(max(auc1, auc2)),
        "best_branch": "classifier" if auc1 >= auc2 else "alignment",
        "ano_auc": float(roc_auc_score(abnormal_gt, abnormal_scores) * 100),
        "avg_mAP": float(np.mean(dmap)),
    }
    metrics.update({f"mAP@{threshold:.1f}": float(value) for threshold, value in zip(iou, dmap)})
    usage = (selection_counts, weight_sums, true_selection_counts, true_weight_sums, true_video_counts)
    return metrics, usage, len(video_labels)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", required=True)
    parser.add_argument("--gt-label-path", required=True)
    parser.add_argument("--subclass-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--baseline-metrics-json")
    parser.add_argument("--aggregation", choices=["max", "soft"], required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-length", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_subclass_bank(args.subclass_json)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    if len(dataset) != 290:
        raise ValueError(f"Expected the complete 290-video test set, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    baseline_cache = Path(args.baseline_metrics_json) if args.baseline_metrics_json else None
    baseline_file = Path(args.baseline_model_path).resolve()
    baseline_identity = {"path": str(baseline_file), "size": baseline_file.stat().st_size,
                         "mtime_ns": baseline_file.stat().st_mtime_ns,
                         "test_list": str(Path(args.test_list).resolve()),
                         "test_list_mtime_ns": Path(args.test_list).stat().st_mtime_ns,
                         "gt_mtime_ns": Path(args.gt_path).stat().st_mtime_ns}
    cache_data = json.loads(baseline_cache.read_text(encoding="utf-8")) if baseline_cache and baseline_cache.is_file() else None
    if cache_data and cache_data.get("identity") == baseline_identity:
        baseline_metrics = cache_data["metrics"]
    else:
        baseline = CLIPVAD(14, 512, args.visual_length, 512, 1, 2, 8, 10, 10, device)
        baseline.load_state_dict(torch.load(args.baseline_model_path, map_location="cpu", weights_only=True), strict=True)
        baseline_args = argparse.Namespace(visual_length=args.visual_length, gt_path=args.gt_path,
                                           gt_segment_path=args.gt_segment_path, gt_label_path=args.gt_label_path)
        label_map = {name: name for name in CLASS_NAMES}
        baseline_metrics = evaluate_model(baseline, loader, baseline_args, label_map, device)
        if baseline_cache:
            baseline_cache.parent.mkdir(parents=True, exist_ok=True)
            baseline_cache.write_text(json.dumps({"identity": baseline_identity, "metrics": baseline_metrics}, indent=2), encoding="utf-8")
        del baseline
        if device == "cuda":
            torch.cuda.empty_cache()

    model = make_model(device, args.visual_length)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    subclass_metrics, usage, videos = evaluate_subclass(model, loader, args, groups, device)
    counts, weights, true_counts, true_weights, true_videos = usage
    print_metrics("Self-trained baseline", baseline_metrics)
    print_metrics("Subclass", subclass_metrics)
    print("Subclass aggregation:", args.aggregation, "across entire videos; tested videos:", videos)
    keys = ["classifier_auc", "classifier_ap", "alignment_auc", "alignment_ap", "ano_auc", "avg_mAP"]
    print("Metric,baseline,subclass,delta")
    for key in keys:
        print(f"{key},{baseline_metrics[key]:.2f},{subclass_metrics[key]:.2f},{subclass_metrics[key] - baseline_metrics[key]:+.2f}")

    (output_dir / "metrics.json").write_text(json.dumps({"baseline": baseline_metrics, "subclass": subclass_metrics,
        "config": {"aggregation": args.aggregation, "temperature": args.temperature,
                   "aggregation_order": "subclass_then_temporal_topk",
                   "checkpoint": args.model_path, "baseline_checkpoint": args.baseline_model_path}}, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "subclass", "delta"])
        for key in keys + ["mAP@0.1", "mAP@0.2", "mAP@0.3", "mAP@0.4", "mAP@0.5"]:
            writer.writerow([key, baseline_metrics[key], subclass_metrics[key], subclass_metrics[key] - baseline_metrics[key]])
    with (output_dir / "subclass_usage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "subclass_index", "subclass_text", "top1_all_test_videos", "mean_weight_all_test_videos"])
        for class_index, name in enumerate(CLASS_NAMES):
            for subclass_index, text in enumerate(groups[class_index]):
                writer.writerow([name, subclass_index, text, int(counts[class_index, subclass_index]),
                                 float(weights[class_index, subclass_index] / videos)])
    with (output_dir / "subclass_usage_by_true_class.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class_name", "subclass_index", "subclass_text", "test_videos_in_class", "top1_videos", "mean_weight"])
        for class_index, name in enumerate(CLASS_NAMES):
            for subclass_index, text in enumerate(groups[class_index]):
                writer.writerow([name, subclass_index, text, int(true_videos[class_index]),
                                 int(true_counts[class_index, subclass_index]),
                                 float(true_weights[class_index, subclass_index] / max(1, true_videos[class_index]))])
    print("Saved evaluation:", output_dir)


if __name__ == "__main__":
    main()
