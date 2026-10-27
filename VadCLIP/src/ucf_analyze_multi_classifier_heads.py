import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model_multi_classifier import (
    CLIPVADAdapterMultiClassifier,
    CLIPVADMultiClassifier,
)
from utils.tools import get_batch_mask, get_prompt_text, process_split


FRAMES_PER_SNIPPET = 16


LABEL_MAP = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "roadAccidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}


class UCFTestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, feature_root, visual_length):
        self.feature_root = Path(feature_root)
        self.visual_length = visual_length
        with Path(file_path).open("r", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature_path = Path(row["path"])
        if not feature_path.is_absolute():
            feature_path = self.feature_root / feature_path
        feature, length = process_split(np.load(feature_path), self.visual_length)
        return torch.tensor(feature), row["label"], length, row.get("video_id", feature_path.stem)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_state_dict(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception:
        payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    return payload


def make_lengths(length, visual_length):
    remaining = int(length)
    values = []
    for _ in range(int(length / visual_length) + 1):
        current = min(remaining, visual_length)
        values.append(current)
        remaining -= current
    return torch.tensor(values, dtype=torch.int64)


def safe_metric(function, target, score):
    target = np.asarray(target)
    if target.size == 0 or np.unique(target).size < 2:
        return float("nan")
    return float(function(target, score) * 100.0)


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def selection_metrics(indices, gt_fraction, anomaly_count):
    indices = sorted(indices)
    if not indices:
        return {
            "selected_count": 0,
            "precision_any": float("nan"),
            "precision_majority": float("nan"),
            "mean_anomaly_fraction": float("nan"),
            "recall_anomaly_snippets": 0.0,
            "anomaly_frame_coverage": 0.0,
        }
    selected_gt = gt_fraction[indices]
    total_anomaly_frames = float(gt_fraction.sum())
    return {
        "selected_count": len(indices),
        "precision_any": float(np.mean(selected_gt > 0)),
        "precision_majority": float(np.mean(selected_gt >= 0.5)),
        "mean_anomaly_fraction": float(np.mean(selected_gt)),
        "recall_anomaly_snippets": safe_div(np.sum(selected_gt > 0), anomaly_count),
        "anomaly_frame_coverage": safe_div(float(selected_gt.sum()), total_anomaly_frames),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze individual VadCLIP multi-classifier heads.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classifier-heads", default=2, type=int)
    parser.add_argument(
        "--model-type",
        default="multi_classifier",
        choices=["multi_classifier", "adapter_multi_classifier"],
    )
    parser.add_argument("--head-adapter-bottleneck", default=128, type=int)
    parser.add_argument("--head-adapter-dropout", default=0.1, type=float)
    parser.add_argument("--head-adapter-residual-scale", default=0.1, type=float)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = UCFTestDataset(args.test_list, args.feature_root, args.visual_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    gt = np.load(args.gt_path)

    model_class = (
        CLIPVADAdapterMultiClassifier
        if args.model_type == "adapter_multi_classifier"
        else CLIPVADMultiClassifier
    )
    model_kwargs = {"num_classifier_heads": args.num_classifier_heads}
    if args.model_type == "adapter_multi_classifier":
        model_kwargs.update(
            adapter_bottleneck=args.head_adapter_bottleneck,
            adapter_dropout=args.head_adapter_dropout,
            adapter_residual_scale=args.head_adapter_residual_scale,
        )
    model = model_class(
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
        **model_kwargs,
    )
    model.load_state_dict(load_state_dict(args.model_path, device), strict=True)
    model.to(device).eval()
    prompt_text = get_prompt_text(LABEL_MAP)

    score_names = [f"head_{index + 1}" for index in range(args.num_classifier_heads)]
    score_names += ["aggregate_raw_mean", "aggregate_probability_mean", "aggregate_probability_max"]
    all_scores = defaultdict(list)
    all_targets = []
    per_class_scores = defaultdict(lambda: defaultdict(list))
    per_class_targets = defaultdict(list)
    topk_rows = []
    overlap_rows = []
    video_rows = []
    gt_cursor = 0

    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            label = item[1][0]
            video_id = item[3][0]
            snippet_count = int(item[2])
            if snippet_count < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = make_lengths(snippet_count, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            _, aggregate_logits, _, head_logits = model(
                visual,
                padding_mask,
                prompt_text,
                lengths,
                return_head_logits=True,
            )
            head_logits = head_logits.reshape(-1, args.num_classifier_heads)[:snippet_count]
            aggregate_logits = aggregate_logits.reshape(-1)[:snippet_count]
            head_probabilities = torch.sigmoid(head_logits).cpu().numpy()
            snippet_scores = {
                f"head_{index + 1}": head_probabilities[:, index]
                for index in range(args.num_classifier_heads)
            }
            snippet_scores["aggregate_raw_mean"] = torch.sigmoid(aggregate_logits).cpu().numpy()
            snippet_scores["aggregate_probability_mean"] = head_probabilities.mean(axis=1)
            snippet_scores["aggregate_probability_max"] = head_probabilities.max(axis=1)

            frame_count = snippet_count * FRAMES_PER_SNIPPET
            frame_gt = np.asarray(gt[gt_cursor:gt_cursor + frame_count], dtype=np.int64)
            if len(frame_gt) != frame_count:
                raise ValueError(f"GT ended early at video {video_id}.")
            gt_cursor += frame_count
            gt_fraction = frame_gt.reshape(snippet_count, FRAMES_PER_SNIPPET).mean(axis=1)
            all_targets.append(frame_gt)
            per_class_targets[label].append(frame_gt)

            row = {"video_id": video_id, "class_name": label, "snippet_count": snippet_count}
            for name, values in snippet_scores.items():
                frame_values = np.repeat(values, FRAMES_PER_SNIPPET)
                all_scores[name].append(frame_values)
                per_class_scores[label][name].append(frame_values)
                row[f"{name}_mean"] = float(np.mean(values))
                row[f"{name}_max"] = float(np.max(values))
            video_rows.append(row)

            if label == "Normal":
                continue
            k = min(snippet_count, int(snippet_count / 16 + 1))
            head_sets = []
            for head_index in range(args.num_classifier_heads):
                values = snippet_scores[f"head_{head_index + 1}"]
                head_sets.append(set(np.argsort(-values, kind="stable")[:k].tolist()))
            for left_index in range(args.num_classifier_heads):
                for right_index in range(left_index + 1, args.num_classifier_heads):
                    left_set = head_sets[left_index]
                    right_set = head_sets[right_index]
                    intersection_size = len(left_set & right_set)
                    union_size = len(left_set | right_set)
                    overlap_rows.append({
                        "video_id": video_id,
                        "class_name": label,
                        "left": f"head_{left_index + 1}",
                        "right": f"head_{right_index + 1}",
                        "mil_k": k,
                        "intersection_count": intersection_size,
                        "union_count": union_size,
                        "overlap_fraction_of_k": safe_div(intersection_size, k),
                        "jaccard": safe_div(intersection_size, union_size),
                    })
            intersection = set.intersection(*head_sets)
            union = set.union(*head_sets)
            anomaly_count = int(np.sum(gt_fraction > 0))
            groups = {"intersection": intersection, "union": union}
            for head_index, selected in enumerate(head_sets):
                others = set.union(*(head_sets[:head_index] + head_sets[head_index + 1:]))
                groups[f"head_{head_index + 1}_unique"] = selected - others
                groups[f"head_{head_index + 1}_all"] = selected
            for group_name, indices in groups.items():
                metrics = selection_metrics(indices, gt_fraction, anomaly_count)
                topk_rows.append({
                    "video_id": video_id,
                    "class_name": label,
                    "group": group_name,
                    "mil_k": k,
                    **metrics,
                })

    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)}.")

    targets = np.concatenate(all_targets)
    metric_rows = []
    flat_scores = {}
    for name in score_names:
        flat_scores[name] = np.concatenate(all_scores[name])
        metric_rows.append({
            "scope": "global",
            "score": name,
            "auc": safe_metric(roc_auc_score, targets, flat_scores[name]),
            "ap": safe_metric(average_precision_score, targets, flat_scores[name]),
            "frames": len(targets),
        })

    normal_target = np.concatenate(per_class_targets["Normal"])
    for class_name in LABEL_MAP:
        if class_name == "Normal" or not per_class_targets[class_name]:
            continue
        class_target = np.concatenate(per_class_targets[class_name])
        target = np.concatenate([normal_target, class_target])
        for name in score_names:
            score = np.concatenate([
                np.concatenate(per_class_scores["Normal"][name]),
                np.concatenate(per_class_scores[class_name][name]),
            ])
            metric_rows.append({
                "scope": class_name,
                "score": name,
                "auc": safe_metric(roc_auc_score, target, score),
                "ap": safe_metric(average_precision_score, target, score),
                "frames": len(target),
            })

    correlation_rows = []
    for left_index in range(args.num_classifier_heads):
        for right_index in range(left_index + 1, args.num_classifier_heads):
            left = f"head_{left_index + 1}"
            right = f"head_{right_index + 1}"
            correlation_rows.append({
                "left": left,
                "right": right,
                "pearson_frame_score": float(np.corrcoef(flat_scores[left], flat_scores[right])[0, 1]),
            })

    overlap_summary = []
    overlap_pairs = sorted({(row["left"], row["right"]) for row in overlap_rows})
    for left, right in overlap_pairs:
        selected = [
            row for row in overlap_rows
            if row["left"] == left and row["right"] == right
        ]
        overlap_summary.append({
            "left": left,
            "right": right,
            "videos": len(selected),
            "mean_overlap_fraction_of_k": float(np.mean([row["overlap_fraction_of_k"] for row in selected])),
            "mean_jaccard": float(np.mean([row["jaccard"] for row in selected])),
        })

    summary_rows = []
    for group in sorted({row["group"] for row in topk_rows}):
        selected = [row for row in topk_rows if row["group"] == group]
        summary_rows.append({
            "group": group,
            "videos": len(selected),
            "mean_selected_count": float(np.mean([row["selected_count"] for row in selected])),
            "precision_any": float(np.nanmean([row["precision_any"] for row in selected])),
            "precision_majority": float(np.nanmean([row["precision_majority"] for row in selected])),
            "mean_anomaly_fraction": float(np.nanmean([row["mean_anomaly_fraction"] for row in selected])),
            "recall_anomaly_snippets": float(np.mean([row["recall_anomaly_snippets"] for row in selected])),
            "anomaly_frame_coverage": float(np.mean([row["anomaly_frame_coverage"] for row in selected])),
        })

    write_csv(output_dir / "head_global_and_per_class_metrics.csv", metric_rows)
    write_csv(output_dir / "topk_group_metrics_per_video.csv", topk_rows)
    write_csv(output_dir / "topk_group_summary.csv", summary_rows)
    write_csv(output_dir / "head_score_correlations.csv", correlation_rows)
    write_csv(output_dir / "head_topk_overlap_per_video.csv", overlap_rows)
    write_csv(output_dir / "head_topk_overlap_summary.csv", overlap_summary)
    write_csv(output_dir / "per_video_head_scores.csv", video_rows)
    summary = {
        "model_path": str(Path(args.model_path)),
        "test_videos": len(dataset),
        "evaluated_frames": len(targets),
        "mil_k_rule": "floor(snippet_count / 16) + 1",
        "global_metrics": [row for row in metric_rows if row["scope"] == "global"],
        "head_correlations": correlation_rows,
        "head_topk_overlap": overlap_summary,
        "topk_group_summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nMULTI-CLASSIFIER HEAD DIAGNOSTICS")
    print(f"Model: {args.model_path}")
    print(f"Test videos: {len(dataset)} | Frames: {len(targets)}")
    print("\nGlobal frame-level AUC/AP")
    for row in summary["global_metrics"]:
        print(f"  {row['score']:<28} AUC={row['auc']:.2f}  AP={row['ap']:.2f}")
    print("\nCross-head frame-score correlation")
    for row in correlation_rows:
        print(f"  {row['left']} vs {row['right']}: {row['pearson_frame_score']:.4f}")
    print("\nCross-head MIL top-k overlap")
    for row in overlap_summary:
        print(
            f"  {row['left']} vs {row['right']}: "
            f"overlap/k={row['mean_overlap_fraction_of_k']:.4f}, "
            f"Jaccard={row['mean_jaccard']:.4f}"
        )
    print("\nMIL top-k quality on abnormal videos")
    print("  precision_any: selected snippets containing at least one anomaly frame")
    print("  recall: fraction of all anomaly snippets covered by the selected group")
    for row in summary_rows:
        print(
            f"  {row['group']:<20} count={row['mean_selected_count']:.2f} "
            f"precision_any={row['precision_any']:.4f} "
            f"mean_gt_fraction={row['mean_anomaly_fraction']:.4f} "
            f"recall={row['recall_anomaly_snippets']:.4f} "
            f"frame_coverage={row['anomaly_frame_coverage']:.4f}"
        )
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
