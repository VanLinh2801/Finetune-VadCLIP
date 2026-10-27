import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

from model import CLIPVAD
from model_video_context import CLIPVADVideoContext
from utils import tools
from utils.ucf_detectionMAP import getDetectionMAP as detection_map


LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def split_contiguous(features, max_length):
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected a non-empty [snippets, width] array, got {features.shape}")
    blocks, lengths = [], []
    for start in range(0, len(features), max_length):
        part = np.asarray(features[start : start + max_length], dtype=np.float32)
        lengths.append(len(part))
        if len(part) < max_length:
            part = np.pad(part, ((0, max_length - len(part)), (0, 0)))
        blocks.append(part)
    return torch.from_numpy(np.stack(blocks)), torch.tensor(lengths, dtype=torch.long)


class TestVideoDataset(Dataset):
    def __init__(self, csv_path, feature_root, visual_length):
        self.feature_root = Path(feature_root)
        self.visual_length = visual_length
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        video_ids = [row.get("video_id") for row in self.rows]
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("Test list must contain exactly one spatial view per video for GT alignment.")
        for row in self.rows:
            path = Path(row["path"])
            path = path if path.is_absolute() else self.feature_root / path
            if not path.is_file():
                raise FileNotFoundError(path)
            row["resolved_path"] = path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = np.load(row["resolved_path"], mmap_mode="r")
        if raw.ndim != 2 or raw.shape[1] != 512:
            raise ValueError(f"Invalid feature shape for {row['resolved_path']}: {raw.shape}")
        blocks, lengths = split_contiguous(np.asarray(raw), self.visual_length)
        return blocks, lengths, row["label"], row["video_id"]


def build_model(args, device, context):
    model_cls = CLIPVADVideoContext if context else CLIPVAD
    common = (
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix,
        args.prompt_postfix, device,
    )
    if context:
        return model_cls(
            *common,
            context_bin_size=args.context_bin_size,
            context_hidden_width=args.context_hidden_width,
            max_context_delta=args.max_context_delta,
        )
    return model_cls(*common)


def load_model(path, args, device, context):
    model = build_model(args, device, context)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate_checkpoint(path, args, dataset, gt, gtsegments, gtlabels, device, context):
    model = load_model(path, args, device, context)
    prompt_text = tools.get_prompt_text(LABEL_MAP)
    classifier_scores, alignment_scores, video_labels = [], [], []
    classifier_by_video, alignment_by_video = [], []
    logits2_for_map = []
    video_probabilities = []
    gt_by_video = []
    correction_groups = {
        "true_high_abnormal_video": [],
        "false_high_abnormal_video": [],
        "false_high_normal_video": [],
    }
    paired_gaps = []
    per_video_rows = []
    cursor = 0

    with torch.no_grad():
        for blocks, lengths_cpu, label, video_id in dataset:
            valid_snippets = int(lengths_cpu.sum())
            blocks = blocks.to(device)
            lengths = lengths_cpu.to(device)
            mask = tools.get_batch_mask(lengths_cpu, args.visual_length).to(device)
            if context:
                _, logits1, logits2, details = model(
                    blocks,
                    mask,
                    prompt_text,
                    lengths,
                    [len(lengths)],
                    return_context_details=True,
                )
                video_probabilities.append(
                    float(torch.sigmoid(details["video_logits_per_view"][0]).cpu())
                )
                base_logits1 = details["base_logits1"]
                delta = details["context_delta"]
            else:
                _, logits1, logits2 = model(blocks, mask, prompt_text, lengths)
                base_logits1 = logits1
                delta = torch.zeros_like(logits1)

            flat_logits1 = logits1.reshape(-1, 1)[:valid_snippets]
            flat_base = base_logits1.reshape(-1, 1)[:valid_snippets]
            flat_delta = delta.reshape(-1, 1)[:valid_snippets]
            flat_logits2 = logits2.reshape(-1, logits2.shape[-1])[:valid_snippets]
            classifier = torch.sigmoid(flat_logits1.squeeze(-1)).cpu().numpy()
            base_classifier = torch.sigmoid(flat_base.squeeze(-1)).cpu().numpy()
            delta_values = flat_delta.squeeze(-1).cpu().numpy()
            alignment = (1.0 - flat_logits2.softmax(dim=-1)[:, 0]).cpu().numpy()

            frame_count = valid_snippets * 16
            gt_video = gt[cursor : cursor + frame_count]
            cursor += frame_count
            if len(gt_video) != frame_count:
                raise ValueError(f"Ground truth ended early at video_id={video_id}")
            snippet_gt = gt_video.reshape(valid_snippets, 16).max(axis=1).astype(np.int8)
            gt_by_video.append(gt_video)

            classifier_repeated = np.repeat(classifier, 16)
            alignment_repeated = np.repeat(alignment, 16)
            classifier_scores.append(classifier_repeated)
            alignment_scores.append(alignment_repeated)
            classifier_by_video.append(classifier_repeated)
            alignment_by_video.append(alignment_repeated)
            video_labels.append(label)
            logits2_for_map.append(np.repeat(flat_logits2.softmax(dim=-1).cpu().numpy(), 16, axis=0))

            top_count = max(1, int(np.ceil(valid_snippets * 0.15)))
            top_indices = np.argpartition(base_classifier, -top_count)[-top_count:]
            true_delta, false_delta = [], []
            for snippet_index in top_indices:
                if label == "Normal":
                    correction_groups["false_high_normal_video"].append(float(delta_values[snippet_index]))
                elif snippet_gt[snippet_index]:
                    correction_groups["true_high_abnormal_video"].append(float(delta_values[snippet_index]))
                    true_delta.append(float(delta_values[snippet_index]))
                else:
                    correction_groups["false_high_abnormal_video"].append(float(delta_values[snippet_index]))
                    false_delta.append(float(delta_values[snippet_index]))
            if true_delta and false_delta:
                paired_gaps.append(float(np.mean(true_delta) - np.mean(false_delta)))
            per_video_rows.append({
                "video_id": video_id,
                "label": label,
                "video_context_probability": video_probabilities[-1] if context else None,
                "top15_true_high_count": len(true_delta),
                "top15_false_high_count": len(false_delta),
                "top15_true_high_mean_delta": float(np.mean(true_delta)) if true_delta else None,
                "top15_false_high_mean_delta": float(np.mean(false_delta)) if false_delta else None,
            })

    if cursor != len(gt):
        raise ValueError(f"Consumed {cursor} GT frames, expected {len(gt)}")
    classifier_scores = np.concatenate(classifier_scores)
    alignment_scores = np.concatenate(alignment_scores)
    classifier_auc = roc_auc_score(gt, classifier_scores) * 100
    classifier_ap = average_precision_score(gt, classifier_scores) * 100
    alignment_auc = roc_auc_score(gt, alignment_scores) * 100
    alignment_ap = average_precision_score(gt, alignment_scores) * 100
    if alignment_auc >= classifier_auc:
        best_scores_by_video, best_auc, best_branch = alignment_by_video, alignment_auc, "alignment"
    else:
        best_scores_by_video, best_auc, best_branch = classifier_by_video, classifier_auc, "classifier"
    abnormal_indices = [i for i, label in enumerate(video_labels) if label != "Normal"]
    ano_auc = roc_auc_score(
        np.concatenate([gt_by_video[i] for i in abnormal_indices]),
        np.concatenate([best_scores_by_video[i] for i in abnormal_indices]),
    ) * 100
    dmap, ious = detection_map(logits2_for_map, gtsegments, gtlabels, excludeNormal=False)
    metrics = {
        "classifier_auc": float(classifier_auc),
        "classifier_ap": float(classifier_ap),
        "alignment_auc": float(alignment_auc),
        "alignment_ap": float(alignment_ap),
        "best_auc": float(best_auc),
        "best_branch": best_branch,
        "ano_auc": float(ano_auc),
        "avg_mAP": float(np.mean(dmap)),
        **{f"mAP@{iou:.1f}": float(value) for iou, value in zip(ious, dmap)},
        "video_abnormal_auc": float(
            roc_auc_score(
                [label != "Normal" for label in video_labels], video_probabilities
            )
        ) if context else None,
        "video_abnormal_ap": float(
            average_precision_score(
                [label != "Normal" for label in video_labels], video_probabilities
            )
        ) if context else None,
        "test_videos": len(video_labels),
    }
    diagnostics = {
        "top15_context_delta_by_group": {
            key: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "positive_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
            }
            for key, values in correction_groups.items()
        },
        "paired_abnormal_video_true_minus_false_high_delta": {
            "count": len(paired_gaps),
            "mean": float(np.mean(paired_gaps)) if paired_gaps else None,
            "median": float(np.median(paired_gaps)) if paired_gaps else None,
        },
        "per_video": per_video_rows,
    }
    return metrics, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VadCLIP with a within-view context GRU")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--context-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--context-bin-size", default=32, type=int)
    parser.add_argument("--context-hidden-width", default=128, type=int)
    parser.add_argument("--max-context-delta", default=0.1, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TestVideoDataset(args.test_list, args.feature_root, args.visual_length)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    if len(dataset) != 290:
        print(f"Warning: expected 290 test videos; found {len(dataset)}")

    baseline_metrics, _ = evaluate_checkpoint(
        args.baseline_model_path, args, dataset, gt, gtsegments, gtlabels, device, context=False
    )
    context_metrics, diagnostics = evaluate_checkpoint(
        args.context_model_path, args, dataset, gt, gtsegments, gtlabels, device, context=True
    )
    comparison = {
        "baseline": baseline_metrics,
        "video_context": context_metrics,
        "delta_context_minus_baseline": {
            key: context_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if isinstance(baseline_metrics[key], (int, float))
            and isinstance(context_metrics.get(key), (int, float))
        },
        "baseline_model_path": args.baseline_model_path,
        "context_model_path": args.context_model_path,
        "test_videos": len(dataset),
        "spatial_crop_note": "Test uses one spatial view per video; views are not treated as temporal chunks.",
    }
    output_dir, analysis_dir = Path(args.output_dir), Path(args.analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (analysis_dir / "context_delta_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    with (analysis_dir / "per_video_context_delta.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = diagnostics["per_video"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(comparison, indent=2))
    print("Context-delta diagnostics:")
    print(json.dumps({key: value for key, value in diagnostics.items() if key != "per_video"}, indent=2))
    print(f"Evaluation results: {output_dir}")
    print(f"Analyze results: {analysis_dir}")


if __name__ == "__main__":
    main()
