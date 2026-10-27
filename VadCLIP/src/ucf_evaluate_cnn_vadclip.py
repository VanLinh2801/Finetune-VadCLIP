import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model import CLIPVAD
from model_cnn_vadclip import CLIPVADCNN
from ucf_cnn_common import TEST_LABEL_MAP, UCFCNNFeatureDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate CNN VadCLIP with the original UCF frame protocol")
    parser.add_argument("--model-type", choices=("cnn", "baseline"), default="cnn")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", required=True)
    parser.add_argument("--gt-label-path", required=True)
    parser.add_argument("--kernel-size", type=int, choices=(1, 3, 5, 7, 9), default=3)
    parser.add_argument("--score-mode", choices=("scoring_head", "kernel_response"), default="scoring_head")
    parser.add_argument("--residual-mode", choices=("on", "off"), default="on")
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--train-config-json")
    parser.add_argument("--evaluate-dir", required=True)
    parser.add_argument("--analyze-dir", required=True)
    return parser


def load_model(args, device):
    common = (14, 512, args.visual_length, 512, 1, 2, 8, 10, 10, device)
    if args.model_type == "cnn":
        model = CLIPVADCNN(*common, kernel_size=args.kernel_size, score_mode=args.score_mode,
                           use_residual=args.residual_mode == "on",
                           projection_width=args.projection_width, hidden_width=args.cnn_channels)
    else:
        model = CLIPVAD(*common)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "score_mean": float(array.mean()) if array.size else None,
        "score_p10": float(np.percentile(array, 10)) if array.size else None,
        "score_p50": float(np.median(array)) if array.size else None,
        "score_p90": float(np.percentile(array, 90)) if array.size else None,
    }


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VadCLIP's distance graph requires CUDA. Choose a Colab GPU runtime.")
    evaluate_dir = Path(args.evaluate_dir)
    analyze_dir = Path(args.analyze_dir)
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    analyze_dir.mkdir(parents=True, exist_ok=True)
    dataset = UCFCNNFeatureDataset(args.feature_root, args.test_list, args.visual_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(args, device)
    train_config = json.loads(Path(args.train_config_json).read_text(encoding="utf-8")) if args.train_config_json else {}
    if args.model_type == "cnn" and train_config.get("residual_mode", "on") != args.residual_mode:
        raise ValueError("Residual mode does not match the training config")
    if args.model_type == "cnn" and args.train_config_json:
        for key in ("projection_width", "cnn_channels", "kernel_size", "score_mode"):
            expected = train_config.get(key, 128 if key in ("projection_width", "cnn_channels") else None)
            if expected is not None and expected != getattr(args, key):
                raise ValueError(f"{key} does not match the training config")
    alignment_trained = args.model_type == "baseline" or train_config.get("lambda_alignment", 1.0) > 0
    prompt_text = get_prompt_text(TEST_LABEL_MAP)
    gt = np.load(args.gt_path)
    gt_segments = np.load(args.gt_segment_path, allow_pickle=True)
    gt_labels = np.load(args.gt_label_path, allow_pickle=True)

    classifier_frames = []
    alignment_frames = []
    alignment_classes = []
    abnormal_gt = []
    abnormal_classifier = []
    abnormal_alignment = []
    groups = {name: [] for name in (
        "true_high_abnormal", "false_high_abnormal", "false_high_normal", "missed_anomaly"
    )}
    selected_rows = []
    video_rows = []
    cursor = 0
    selected_anomaly = 0
    selected_total = 0
    topk_fraction_macro = []

    with torch.no_grad():
        for chunks, labels, original_lengths, video_ids, chunk_lengths in loader:
            video_label = labels[0]
            video_id = video_ids[0]
            original_length = int(original_lengths[0])
            visual = chunks.squeeze(0).to(device)
            lengths = chunk_lengths.squeeze(0).to(device)
            mask = get_batch_mask(lengths, args.visual_length).to(device)
            _, logits1, logits2 = model(visual, mask, prompt_text, lengths)
            classifier = logits1.reshape(-1)[:original_length].sigmoid().cpu().numpy()
            class_probs = logits2.reshape(-1, 14)[:original_length].softmax(-1).cpu().numpy()
            alignment = 1.0 - class_probs[:, 0]
            if not np.isfinite(classifier).all() or not np.isfinite(class_probs).all():
                raise FloatingPointError(f"Nonfinite score in {video_id}")

            frames = original_length * 16
            video_gt = gt[cursor:cursor + frames]
            cursor += frames
            if len(video_gt) != frames:
                raise ValueError(f"GT frame mismatch at {video_id}: expected {frames}, got {len(video_gt)}")
            snippet_gt = video_gt.reshape(original_length, 16).max(axis=1).astype(np.int8)
            classifier_frame = np.repeat(classifier, 16)
            alignment_frame = np.repeat(alignment, 16)
            classifier_frames.append(classifier_frame)
            alignment_frames.append(alignment_frame)
            alignment_classes.append(np.repeat(class_probs, 16, axis=0))
            if video_label != "Normal":
                abnormal_gt.append(video_gt)
                abnormal_classifier.append(classifier_frame)
                abnormal_alignment.append(alignment_frame)

            k = max(1, int(np.ceil(original_length * 0.15)))
            top_indices = np.argsort(-classifier, kind="stable")[:k]
            top_set = set(top_indices.tolist())
            anomaly_in_top = int(snippet_gt[top_indices].sum())
            if video_label != "Normal":
                selected_anomaly += anomaly_in_top
                selected_total += k
            topk_fraction_macro.append(anomaly_in_top / k if video_label != "Normal" else np.nan)
            video_rows.append({
                "video_id": video_id,
                "label": video_label,
                "num_snippets": original_length,
                "anomaly_snippets": int(snippet_gt.sum()),
                "top15_count": k,
                "anomaly_in_top15": anomaly_in_top,
                "top15_anomaly_fraction": anomaly_in_top / k,
                "classifier_score_mean": float(classifier.mean()),
                "alignment_score_mean": float(alignment.mean()),
            })
            for index in range(original_length):
                if index in top_set:
                    if video_label == "Normal":
                        group = "false_high_normal"
                    elif snippet_gt[index]:
                        group = "true_high_abnormal"
                    else:
                        group = "false_high_abnormal"
                elif snippet_gt[index]:
                    group = "missed_anomaly"
                else:
                    continue
                score = float(classifier[index])
                groups[group].append(score)
                selected_rows.append({
                    "video_id": video_id, "label": video_label, "snippet_index": index,
                    "group": group, "anomaly_frame_present": int(snippet_gt[index]),
                    "classifier_score": score, "alignment_score": float(alignment[index]),
                })

    if cursor != len(gt):
        raise ValueError(f"Consumed {cursor} GT frames, but GT contains {len(gt)}")
    classifier_frames = np.concatenate(classifier_frames)
    alignment_frames = np.concatenate(alignment_frames)
    classifier_auc = roc_auc_score(gt, classifier_frames) * 100
    alignment_auc = roc_auc_score(gt, alignment_frames) * 100
    dmap, iou = getDetectionMAP(alignment_classes, gt_segments, gt_labels, excludeNormal=False)
    best_branch = "classifier" if not alignment_trained or classifier_auc >= alignment_auc else "alignment"
    ano_scores = abnormal_classifier if best_branch == "classifier" else abnormal_alignment
    metrics = {
        "model_type": args.model_type,
        "model_path": str(args.model_path),
        "kernel_size": args.kernel_size if args.model_type == "cnn" else None,
        "score_mode": args.score_mode if args.model_type == "cnn" else None,
        "residual_mode": args.residual_mode if args.model_type == "cnn" else None,
        "projection_width": args.projection_width if args.model_type == "cnn" else None,
        "cnn_channels": args.cnn_channels if args.model_type == "cnn" else None,
        "alignment_trained": alignment_trained,
        "videos": len(dataset),
        "classifier_auc": float(classifier_auc),
        "classifier_ap": float(average_precision_score(gt, classifier_frames) * 100),
        "alignment_auc": float(alignment_auc),
        "alignment_ap": float(average_precision_score(gt, alignment_frames) * 100),
        "best_branch": best_branch,
        "best_auc": float(classifier_auc if best_branch == "classifier" else alignment_auc),
        "ano_auc": float(roc_auc_score(np.concatenate(abnormal_gt), np.concatenate(ano_scores)) * 100),
        "avg_mAP": float(np.mean(dmap)),
    }
    for threshold, value in zip(iou, dmap):
        metrics[f"mAP@{threshold:.1f}"] = float(value)
    diagnostic = {
        "definition": "top 15 percent is selected per video from classifier score; snippet GT is positive when any of its 16 frames is anomalous",
        "selected_anomaly_fraction_micro": selected_anomaly / selected_total,
        "selected_anomaly_fraction_abnormal_video_macro": float(np.nanmean(topk_fraction_macro)),
        "groups": {name: group_summary(values) for name, values in groups.items()},
    }
    write_csv(analyze_dir / "selected_and_missed_snippets.csv", selected_rows)
    write_csv(analyze_dir / "per_video_top15.csv", video_rows)
    (analyze_dir / "summary.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    (evaluate_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print("Top-15 snippet diagnostics:", json.dumps(diagnostic, indent=2), flush=True)
    return metrics, diagnostic


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
