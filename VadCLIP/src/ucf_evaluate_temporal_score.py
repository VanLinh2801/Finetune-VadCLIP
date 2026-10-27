import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import ucf_option_temporal_score
from model import CLIPVAD
from model_temporal_score import CLIPVADTemporalScore
from ucf_evaluate import (
    UCFRelativeTestDataset,
    evaluate_model,
    print_metrics,
)
from utils.tools import get_batch_mask


LABEL_MAP = {
    "Normal": "Normal", "Abuse": "Abuse", "Arrest": "Arrest", "Arson": "Arson",
    "Assault": "Assault", "Burglary": "Burglary", "Explosion": "Explosion",
    "Fighting": "Fighting", "RoadAccidents": "RoadAccidents", "Robbery": "Robbery",
    "Shooting": "Shooting", "Shoplifting": "Shoplifting", "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}


def build_model(args, device, temporal):
    model_cls = CLIPVADTemporalScore if temporal else CLIPVAD
    common = (
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix,
        args.prompt_postfix, device,
    )
    if temporal:
        return model_cls(
            *common,
            temporal_mode=args.temporal_mode,
            temporal_hidden_width=args.temporal_hidden_width,
            max_temporal_delta=args.max_temporal_delta,
        )
    return model_cls(*common)


def load_model(args, device, temporal, path):
    model = build_model(args, device, temporal)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None,
                "positive_fraction": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "positive_fraction": float((values > 0).mean()),
    }


def collect_delta_diagnostics(model, loader, args, frame_gt, output_dir, device):
    rows = []
    all_snippet_rows = []
    cursor = 0
    abnormal_pair_gaps = []
    group_deltas = {
        "true_high_abnormal_video": [],
        "false_high_abnormal_video": [],
        "false_high_normal_video": [],
    }
    snippet_groups = {
        "anomaly_snippet_abnormal_video": [],
        "normal_snippet_abnormal_video": [],
        "normal_video_snippet": [],
    }
    region_groups = {
        region: {name: [] for name in snippet_groups}
        for region in ("all", "base_top15", "outside_base_top15", "adjusted_top15")
    }
    adjusted_topk_anomaly_counts = []
    selected_count = 0

    with torch.no_grad():
        for item in loader:
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            length = int(item[2])
            video_id = item[3][0] if len(item) > 3 else str(len(adjusted_topk_anomaly_counts))
            len_cur = length
            if len_cur < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / args.visual_length) + 1)
            remaining = length
            for index in range(len(lengths)):
                if index == 0 and remaining < args.visual_length:
                    lengths[index] = remaining
                elif remaining > args.visual_length:
                    lengths[index] = args.visual_length
                    remaining -= args.visual_length
                else:
                    lengths[index] = remaining
            lengths = lengths.to(dtype=torch.long)
            mask = get_batch_mask(lengths, args.visual_length).to(device)
            _, adjusted_logits, _, details = model(
                visual, mask, [], lengths, return_temporal_details=True
            )
            base = details["base_logits1"].reshape(-1)[:len_cur].sigmoid().cpu().numpy()
            delta = details["temporal_delta"].reshape(-1)[:len_cur].cpu().numpy()
            adjusted = adjusted_logits.reshape(-1)[:len_cur].sigmoid().cpu().numpy()

            snippet_frame_count = len_cur * 16
            video_gt_frames = frame_gt[cursor:cursor + snippet_frame_count]
            cursor += snippet_frame_count
            if len(video_gt_frames) != snippet_frame_count:
                raise ValueError(f"Ground-truth length mismatch at {video_id}")
            snippet_gt = video_gt_frames.reshape(len_cur, 16).max(axis=1).astype(np.int8)

            k = max(1, int(np.ceil(len_cur * 0.15)))
            base_top = np.argpartition(base, -k)[-k:]
            adjusted_top = np.argpartition(adjusted, -k)[-k:]
            base_top_set = set(base_top.tolist())
            adjusted_top_set = set(adjusted_top.tolist())
            adjusted_topk_anomaly_counts.append(float(snippet_gt[adjusted_top].mean()))
            selected_count += len(base_top)

            true_deltas = []
            false_deltas = []
            for snippet_index in base_top:
                is_anomaly = int(snippet_gt[snippet_index]) == 1
                if label != "Normal":
                    group = "true_high_abnormal_video" if is_anomaly else "false_high_abnormal_video"
                else:
                    group = "false_high_normal_video"
                group_deltas[group].append(float(delta[snippet_index]))
                if label != "Normal":
                    (true_deltas if is_anomaly else false_deltas).append(float(delta[snippet_index]))
                rows.append({
                    "video_id": video_id,
                    "video_label": label,
                    "snippet_index": int(snippet_index),
                    "uncorrected_top15_group": group,
                    "snippet_has_anomaly_frame": int(is_anomaly),
                    "base_classifier_score": float(base[snippet_index]),
                    "temporal_delta_logit": float(delta[snippet_index]),
                    "adjusted_classifier_score": float(adjusted[snippet_index]),
                })

            # Record every valid snippet so score corrections outside the top 15% are visible.
            for snippet_index in range(len_cur):
                is_anomaly = int(snippet_gt[snippet_index]) == 1
                if label == "Normal":
                    snippet_group = "normal_video_snippet"
                elif is_anomaly:
                    snippet_group = "anomaly_snippet_abnormal_video"
                else:
                    snippet_group = "normal_snippet_abnormal_video"

                delta_value = float(delta[snippet_index])
                snippet_groups[snippet_group].append(delta_value)
                region_groups["all"][snippet_group].append(delta_value)
                in_base_top = snippet_index in base_top_set
                in_adjusted_top = snippet_index in adjusted_top_set
                if in_base_top:
                    region_groups["base_top15"][snippet_group].append(delta_value)
                else:
                    region_groups["outside_base_top15"][snippet_group].append(delta_value)
                if in_adjusted_top:
                    region_groups["adjusted_top15"][snippet_group].append(delta_value)

                all_snippet_rows.append({
                    "video_id": video_id,
                    "video_label": label,
                    "snippet_index": snippet_index,
                    "snippet_group": snippet_group,
                    "snippet_has_anomaly_frame": is_anomaly,
                    "in_uncorrected_top15": int(in_base_top),
                    "in_adjusted_top15": int(in_adjusted_top),
                    "base_classifier_score": float(base[snippet_index]),
                    "temporal_delta_logit": delta_value,
                    "adjusted_classifier_score": float(adjusted[snippet_index]),
                })
            if true_deltas and false_deltas:
                abnormal_pair_gaps.append(float(np.mean(true_deltas) - np.mean(false_deltas)))

    if cursor != len(frame_gt):
        raise ValueError(f"Consumed {cursor} GT frames, expected {len(frame_gt)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_top15_temporal_delta_by_snippet.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    all_csv_path = output_dir / "all_snippets_temporal_delta.csv"
    if all_snippet_rows:
        with all_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_snippet_rows[0]))
            writer.writeheader()
            writer.writerows(all_snippet_rows)

    summary = {
        "topk_ratio": 0.15,
        "selection_basis": "uncorrected classifier score from the trained temporal model",
        "selected_snippets": selected_count,
        "delta_logit_by_group": {name: summarize(values) for name, values in group_deltas.items()},
        "paired_abnormal_video_true_high_minus_false_high_delta": summarize(abnormal_pair_gaps),
        "adjusted_top15_anomaly_fraction_macro_video_mean": float(np.mean(adjusted_topk_anomaly_counts)),
        "all_snippets": {
            "count": len(all_snippet_rows),
            "delta_logit": summarize([row["temporal_delta_logit"] for row in all_snippet_rows]),
            "delta_logit_by_snippet_group": {
                name: summarize(values) for name, values in snippet_groups.items()
            },
            "delta_logit_by_selection_region": {
                region: {name: summarize(values) for name, values in groups.items()}
                for region, groups in region_groups.items()
            },
            "csv": str(all_csv_path),
        },
        "videos": len(adjusted_topk_anomaly_counts),
        "snippet_csv": str(csv_path),
    }
    (output_dir / "temporal_delta_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Temporal correction diagnostics (logit units):")
    for group, stats in summary["delta_logit_by_group"].items():
        print(f"  {group}: n={stats['count']}, mean={stats['mean']}, median={stats['median']}")
    pair = summary["paired_abnormal_video_true_high_minus_false_high_delta"]
    print(
        "  Paired abnormal-video mean(delta true-high - false-high):",
        f"n={pair['count']}, mean={pair['mean']}, median={pair['median']}",
    )
    print(
        "  Adjusted top-15% anomaly fraction (macro-video mean):",
        f"{summary['adjusted_top15_anomaly_fraction_macro_video_mean']:.4f}",
    )
    print("Temporal correction across ALL valid snippets (logit units):")
    all_stats = summary["all_snippets"]
    print(
        f"  All snippets: n={all_stats['count']}, "
        f"mean={all_stats['delta_logit']['mean']:.6f}, "
        f"positive={all_stats['delta_logit']['positive_fraction']:.2%}"
    )
    for group, stats in all_stats["delta_logit_by_snippet_group"].items():
        print(
            f"  {group}: n={stats['count']}, mean={stats['mean']}, "
            f"positive={stats['positive_fraction']}"
        )
    for region, groups in all_stats["delta_logit_by_selection_region"].items():
        print(f"  Region: {region}")
        for group, stats in groups.items():
            print(
                f"    {group}: n={stats['count']}, mean={stats['mean']}, "
                f"positive={stats['positive_fraction']}"
            )
    print(f"  Full snippet table: {all_csv_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a temporal-score VadCLIP model")
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--temporal-model-path", required=True)
    parser.add_argument("--temporal-mode", required=True,
                        choices=["feature_sequence", "feature_difference"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
    parser.add_argument("--temporal-hidden-width", default=128, type=int)
    parser.add_argument("--max-temporal-delta", default=0.5, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    print(f"Test videos: {len(dataset)} | device: {device}")

    baseline = load_model(args, device, False, args.baseline_model_path)
    temporal = load_model(args, device, True, args.temporal_model_path)
    baseline_metrics = evaluate_model(baseline, loader, args, LABEL_MAP, device)
    temporal_metrics = evaluate_model(temporal, loader, args, LABEL_MAP, device)
    print_metrics("Self-trained baseline", baseline_metrics)
    print_metrics(f"Temporal {args.temporal_mode}", temporal_metrics)
    metric_keys = list(baseline_metrics)
    comparison = {
        "baseline": baseline_metrics,
        "temporal": temporal_metrics,
        "delta_temporal_minus_baseline": {
            key: temporal_metrics[key] - baseline_metrics[key]
            for key in metric_keys
            if isinstance(baseline_metrics[key], (int, float))
        },
        "baseline_checkpoint": args.baseline_model_path,
        "temporal_checkpoint": args.temporal_model_path,
        "temporal_mode": args.temporal_mode,
        "test_videos": len(dataset),
    }
    (output_dir / "metrics_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    collect_delta_diagnostics(
        temporal, loader, args, gt, Path(args.analysis_dir), device
    )


if __name__ == "__main__":
    main()
