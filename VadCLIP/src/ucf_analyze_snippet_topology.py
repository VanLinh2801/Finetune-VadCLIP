import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from ucf_analyze_score_separation import (
    BRANCHES,
    UCFRelativeTestDataset,
    get_prompt_text_for_config,
    get_prompt_weights_for_config,
    load_model,
    make_lengths,
)
from utils.tools import get_batch_mask


FRAMES_PER_SNIPPET = 16


def parse_ratios(value):
    ratios = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not ratios or any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise argparse.ArgumentTypeError("Ratios must be comma-separated values between 0 and 1.")
    return ratios


def ratio_key(ratio):
    return f"top{int(round(ratio * 100)):02d}"


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


def selected_indices(scores, ratio):
    scores = np.asarray(scores)
    count = min(scores.size, max(1, int(np.ceil(scores.size * ratio))))
    # Stable sorting makes equal-score selection reproducible.
    return np.sort(np.argsort(-scores, kind="stable")[:count])


def contiguous_runs(indices):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    if indices.size == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    return [chunk.tolist() for chunk in np.split(indices, split_points)]


def topology(indices, visual_length):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    runs = contiguous_runs(indices)
    lengths = np.asarray([len(run) for run in runs], dtype=np.float32)
    selected = set(indices.tolist())
    with_neighbor = sum(
        1 for index in indices if index - 1 in selected or index + 1 in selected
    )
    boundary_crossings = sum(
        1
        for index in indices
        if (index + 1) in selected and (index + 1) % visual_length == 0
    )
    return {
        "selected_count": int(indices.size),
        "run_count": len(runs),
        "longest_run": int(lengths.max()) if lengths.size else 0,
        "mean_run_length": float(lengths.mean()) if lengths.size else 0.0,
        "singleton_run_ratio": float(np.mean(lengths == 1)) if lengths.size else np.nan,
        "selected_with_neighbor_ratio": float(with_neighbor / indices.size) if indices.size else np.nan,
        "mean_selected_index_gap": float(np.diff(indices).mean()) if indices.size > 1 else np.nan,
        "block_boundary_crossings": int(boundary_crossings),
        "runs": runs,
    }


def snippet_ground_truth(gt_frames, snippet_count):
    expected = snippet_count * FRAMES_PER_SNIPPET
    if len(gt_frames) != expected:
        raise ValueError(f"Expected {expected} GT frames, received {len(gt_frames)}.")
    grouped = np.asarray(gt_frames).reshape(snippet_count, FRAMES_PER_SNIPPET)
    fractions = grouped.mean(axis=1).astype(np.float32)
    return (fractions > 0).astype(np.int64), (fractions >= 0.5).astype(np.int64), fractions


def inversion_burden(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive_scores = np.sort(scores[labels == 1])
    negative_scores = np.sort(scores[labels == 0])
    burden = np.zeros(scores.size, dtype=np.float64)
    denominator = np.zeros(scores.size, dtype=np.int64)

    for index, (score, label) in enumerate(zip(scores, labels)):
        if label == 1:
            left = np.searchsorted(negative_scores, score, side="left")
            right = np.searchsorted(negative_scores, score, side="right")
            burden[index] = (negative_scores.size - right) + 0.5 * (right - left)
            denominator[index] = negative_scores.size
        else:
            left = np.searchsorted(positive_scores, score, side="left")
            right = np.searchsorted(positive_scores, score, side="right")
            burden[index] = left + 0.5 * (right - left)
            denominator[index] = positive_scores.size

    fractions = np.divide(
        burden,
        denominator,
        out=np.full_like(burden, np.nan),
        where=denominator > 0,
    )
    return burden, fractions


def run_inference(model, dataloader, args, config, gt, device):
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
            if prompt_weights is None:
                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            else:
                _, logits1, logits2 = model(
                    visual,
                    padding_mask,
                    prompt_text,
                    lengths,
                    prompt_weights=prompt_weights,
                )

            logits1 = logits1.reshape(-1, logits1.shape[-1])[:snippet_count]
            logits2 = logits2.reshape(-1, logits2.shape[-1])[:snippet_count]
            classifier = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy()
            alignment = (1 - logits2.softmax(dim=-1)[:, 0]).cpu().numpy()

            frame_count = snippet_count * FRAMES_PER_SNIPPET
            gt_frames = np.asarray(gt[gt_cursor : gt_cursor + frame_count], dtype=np.int64)
            gt_cursor += frame_count
            gt_any, gt_majority, gt_fraction = snippet_ground_truth(gt_frames, snippet_count)
            records.append(
                {
                    "video_id": str(video_id),
                    "label": str(label),
                    "snippet_count": snippet_count,
                    "gt_frames": gt_frames,
                    "gt_any": gt_any,
                    "gt_majority": gt_majority,
                    "gt_fraction": gt_fraction,
                    "classifier": classifier,
                    "alignment": alignment,
                }
            )

    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)} frames.")
    return records


def analyze_records(model_name, records, ratios, visual_length):
    snippet_rows = []
    topology_rows = []
    group_rows = []
    agreement_rows = []
    ground_truth_rows = []
    sequence_records = []

    offsets = []
    cursor = 0
    for record in records:
        offsets.append(cursor)
        cursor += record["snippet_count"]

    all_gt_any = np.concatenate([record["gt_any"] for record in records])
    global_scores = {
        branch: np.concatenate([record[branch] for record in records]) for branch in BRANCHES
    }
    burdens = {}
    for branch in BRANCHES:
        burdens[branch] = inversion_burden(global_scores[branch], all_gt_any)

    for record, offset in zip(records, offsets):
        count = record["snippet_count"]
        branch_rank = {
            branch: np.argsort(np.argsort(-record[branch], kind="stable"), kind="stable") + 1
            for branch in BRANCHES
        }
        branch_selected = {
            branch: {ratio: selected_indices(record[branch], ratio) for ratio in ratios}
            for branch in BRANCHES
        }
        branch_selected_masks = {
            branch: {ratio: np.zeros(count, dtype=bool) for ratio in ratios}
            for branch in BRANCHES
        }
        for branch in BRANCHES:
            for ratio in ratios:
                branch_selected_masks[branch][ratio][branch_selected[branch][ratio]] = True

        gt_indices = np.flatnonzero(record["gt_any"] == 1)
        gt_topology = topology(gt_indices, visual_length)
        ground_truth_rows.append(
            {
                "model": model_name,
                "video_id": record["video_id"],
                "class_name": record["label"],
                "snippet_count": count,
                "anomaly_snippet_count": int(gt_indices.size),
                **{key: value for key, value in gt_topology.items() if key != "runs"},
            }
        )

        for snippet_index in range(count):
            row = {
                "model": model_name,
                "video_id": record["video_id"],
                "class_name": record["label"],
                "snippet_index": snippet_index,
                "block_index": snippet_index // visual_length,
                "index_in_block": snippet_index % visual_length,
                "frame_start": snippet_index * FRAMES_PER_SNIPPET,
                "frame_end_exclusive": (snippet_index + 1) * FRAMES_PER_SNIPPET,
                "gt_any": int(record["gt_any"][snippet_index]),
                "gt_majority": int(record["gt_majority"][snippet_index]),
                "gt_fraction": float(record["gt_fraction"][snippet_index]),
            }
            for branch in BRANCHES:
                global_index = offset + snippet_index
                row[f"{branch}_score"] = float(record[branch][snippet_index])
                row[f"{branch}_rank_in_video"] = int(branch_rank[branch][snippet_index])
                row[f"{branch}_inversion_count"] = float(burdens[branch][0][global_index])
                row[f"{branch}_inversion_fraction"] = float(burdens[branch][1][global_index])
                for ratio in ratios:
                    row[f"{branch}_{ratio_key(ratio)}"] = int(
                        branch_selected_masks[branch][ratio][snippet_index]
                    )
            snippet_rows.append(row)

        for branch in BRANCHES:
            for ratio in ratios:
                indices = branch_selected[branch][ratio]
                selected_mask = branch_selected_masks[branch][ratio]
                gt_mask = record["gt_any"].astype(bool)
                topo = topology(indices, visual_length)
                hits = int(np.sum(selected_mask & gt_mask))
                groups = {
                    "true_high": selected_mask & gt_mask,
                    "false_high": selected_mask & ~gt_mask,
                    "missed_anomaly": ~selected_mask & gt_mask,
                    "true_low": ~selected_mask & ~gt_mask,
                }
                topology_rows.append(
                    {
                        "model": model_name,
                        "video_id": record["video_id"],
                        "class_name": record["label"],
                        "branch": branch,
                        "top_k_ratio": ratio,
                        "snippet_count": count,
                        "topk_hits": hits,
                        "topk_overlap": float(hits / len(indices)),
                        **{key_: value for key_, value in topo.items() if key_ != "runs"},
                    }
                )
                for group_name, mask in groups.items():
                    group_rows.append(
                        {
                            "model": model_name,
                            "video_id": record["video_id"],
                            "class_name": record["label"],
                            "branch": branch,
                            "top_k_ratio": ratio,
                            "group": group_name,
                            "count": int(mask.sum()),
                            "score_mean": float(record[branch][mask].mean()) if mask.any() else np.nan,
                            "gt_fraction_mean": float(record["gt_fraction"][mask].mean()) if mask.any() else np.nan,
                        }
                    )
                sequence_records.append(
                    {
                        "model": model_name,
                        "video_id": record["video_id"],
                        "class_name": record["label"],
                        "branch": branch,
                        "top_k_ratio": ratio,
                        "selected_indices": indices.tolist(),
                        "contiguous_runs": topo["runs"],
                    }
                )

        for ratio in ratios:
            classifier_mask = np.zeros(count, dtype=bool)
            alignment_mask = np.zeros(count, dtype=bool)
            classifier_mask[branch_selected["classifier"][ratio]] = True
            alignment_mask[branch_selected["alignment"][ratio]] = True
            quadrants = {
                "both_high": classifier_mask & alignment_mask,
                "classifier_only": classifier_mask & ~alignment_mask,
                "alignment_only": ~classifier_mask & alignment_mask,
                "both_low": ~classifier_mask & ~alignment_mask,
            }
            for quadrant, mask in quadrants.items():
                agreement_rows.append(
                    {
                        "model": model_name,
                        "video_id": record["video_id"],
                        "class_name": record["label"],
                        "top_k_ratio": ratio,
                        "quadrant": quadrant,
                        "count": int(mask.sum()),
                        "anomaly_count": int(np.sum(record["gt_any"][mask])),
                        "anomaly_rate": float(record["gt_any"][mask].mean()) if mask.any() else np.nan,
                        "gt_fraction_mean": float(record["gt_fraction"][mask].mean()) if mask.any() else np.nan,
                    }
                )

    summary_rows = []
    frame_gt = np.concatenate([record["gt_frames"] for record in records])
    for branch in BRANCHES:
        scores = global_scores[branch]
        frame_scores = np.repeat(scores, FRAMES_PER_SNIPPET)
        total_pairs = int(np.sum(all_gt_any == 1) * np.sum(all_gt_any == 0))
        inversion_total = float(np.sum(burdens[branch][0][all_gt_any == 1]))
        for ratio in ratios:
            relevant = [
                row
                for row in topology_rows
                if row["branch"] == branch
                and row["top_k_ratio"] == ratio
                and row["class_name"] != "Normal"
            ]
            summary_rows.append(
                {
                    "model": model_name,
                    "branch": branch,
                    "top_k_ratio": ratio,
                    "paper_frame_auc": safe_auc(frame_gt, frame_scores),
                    "paper_frame_ap": safe_ap(frame_gt, frame_scores),
                    "snippet_any_auc": safe_auc(all_gt_any, scores),
                    "snippet_any_ap": safe_ap(all_gt_any, scores),
                    "topk_overlap_mean": float(np.mean([row["topk_overlap"] for row in relevant])),
                    "longest_run_mean": float(np.mean([row["longest_run"] for row in relevant])),
                    "singleton_run_ratio_mean": float(np.nanmean([row["singleton_run_ratio"] for row in relevant])),
                    "selected_with_neighbor_ratio_mean": float(np.nanmean([row["selected_with_neighbor_ratio"] for row in relevant])),
                    "inverted_pair_burden": inversion_total,
                    "total_positive_negative_pairs": total_pairs,
                    "inverted_pair_fraction": float(inversion_total / total_pairs) if total_pairs else np.nan,
                }
            )

    return {
        "snippet_rows": snippet_rows,
        "topology_rows": topology_rows,
        "group_rows": group_rows,
        "agreement_rows": agreement_rows,
        "ground_truth_rows": ground_truth_rows,
        "sequence_records": sequence_records,
        "summary_rows": summary_rows,
    }


def print_summary(rows):
    print("\nSnippet-level topology summary")
    for row in rows:
        print(
            f"  {row['model']} | {row['branch']} | top={row['top_k_ratio']:.0%} | "
            f"frame AUC={row['paper_frame_auc']:.2f} | snippet AUC={row['snippet_any_auc']:.2f} | "
            f"overlap={row['topk_overlap_mean']:.3f} | longest run={row['longest_run_mean']:.2f} | "
            f"singleton={row['singleton_run_ratio_mean']:.3f} | neighbor={row['selected_with_neighbor_ratio_mean']:.3f} | "
            f"inverted pairs={row['inverted_pair_fraction']:.3f}"
        )


def aggregate_rows(collected):
    group_summary = {}
    for row in collected["group_rows"]:
        key = (row["model"], row["branch"], row["top_k_ratio"], row["group"])
        target = group_summary.setdefault(
            key,
            {
                "model": row["model"],
                "branch": row["branch"],
                "top_k_ratio": row["top_k_ratio"],
                "group": row["group"],
                "count": 0,
            },
        )
        target["count"] += row["count"]

    group_totals = {}
    for row in group_summary.values():
        key = (row["model"], row["branch"], row["top_k_ratio"])
        group_totals[key] = group_totals.get(key, 0) + row["count"]
    for row in group_summary.values():
        total = group_totals[(row["model"], row["branch"], row["top_k_ratio"])]
        row["fraction"] = float(row["count"] / total) if total else np.nan

    agreement_summary = {}
    for row in collected["agreement_rows"]:
        key = (row["model"], row["top_k_ratio"], row["quadrant"])
        target = agreement_summary.setdefault(
            key,
            {
                "model": row["model"],
                "top_k_ratio": row["top_k_ratio"],
                "quadrant": row["quadrant"],
                "count": 0,
                "anomaly_count": 0,
            },
        )
        target["count"] += row["count"]
        target["anomaly_count"] += row["anomaly_count"]
    for row in agreement_summary.values():
        row["anomaly_rate"] = (
            float(row["anomaly_count"] / row["count"]) if row["count"] else np.nan
        )

    gt_summary = {}
    for row in collected["ground_truth_rows"]:
        key = (row["model"], row["class_name"])
        target = gt_summary.setdefault(
            key,
            {
                "model": row["model"],
                "class_name": row["class_name"],
                "num_videos": 0,
                "videos_with_anomaly": 0,
                "run_count_values": [],
                "longest_run_values": [],
                "singleton_values": [],
                "neighbor_values": [],
            },
        )
        target["num_videos"] += 1
        if row["anomaly_snippet_count"] > 0:
            target["videos_with_anomaly"] += 1
            target["run_count_values"].append(row["run_count"])
            target["longest_run_values"].append(row["longest_run"])
            target["singleton_values"].append(row["singleton_run_ratio"])
            target["neighbor_values"].append(row["selected_with_neighbor_ratio"])

    gt_summary_rows = []
    for target in gt_summary.values():
        def mean_or_nan(values):
            return float(np.nanmean(values)) if values else np.nan

        gt_summary_rows.append(
            {
                "model": target["model"],
                "class_name": target["class_name"],
                "num_videos": target["num_videos"],
                "videos_with_anomaly": target["videos_with_anomaly"],
                "anomaly_run_count_mean": mean_or_nan(target["run_count_values"]),
                "anomaly_longest_run_mean": mean_or_nan(target["longest_run_values"]),
                "anomaly_singleton_run_ratio_mean": mean_or_nan(target["singleton_values"]),
                "anomaly_with_neighbor_ratio_mean": mean_or_nan(target["neighbor_values"]),
            }
        )

    class_summary = {}
    group_by_class = {}
    for row in collected["group_rows"]:
        key = (
            row["model"],
            row["class_name"],
            row["branch"],
            row["top_k_ratio"],
            row["group"],
        )
        group_by_class[key] = group_by_class.get(key, 0) + row["count"]
    for row in collected["topology_rows"]:
        key = (row["model"], row["class_name"], row["branch"], row["top_k_ratio"])
        target = class_summary.setdefault(
            key,
            {
                "model": row["model"],
                "class_name": row["class_name"],
                "branch": row["branch"],
                "top_k_ratio": row["top_k_ratio"],
                "num_videos": 0,
                "overlap_values": [],
                "longest_run_values": [],
                "singleton_values": [],
            },
        )
        target["num_videos"] += 1
        target["overlap_values"].append(row["topk_overlap"])
        target["longest_run_values"].append(row["longest_run"])
        target["singleton_values"].append(row["singleton_run_ratio"])

    class_summary_rows = []
    for key, target in class_summary.items():
        model, class_name, branch, ratio = key
        class_summary_rows.append(
            {
                "model": model,
                "class_name": class_name,
                "branch": branch,
                "top_k_ratio": ratio,
                "num_videos": target["num_videos"],
                "topk_overlap_mean": float(np.nanmean(target["overlap_values"])),
                "longest_run_mean": float(np.nanmean(target["longest_run_values"])),
                "singleton_run_ratio_mean": float(np.nanmean(target["singleton_values"])),
                "true_high": group_by_class.get((model, class_name, branch, ratio, "true_high"), 0),
                "false_high": group_by_class.get((model, class_name, branch, ratio, "false_high"), 0),
                "missed_anomaly": group_by_class.get((model, class_name, branch, ratio, "missed_anomaly"), 0),
                "true_low": group_by_class.get((model, class_name, branch, ratio, "true_low"), 0),
            }
        )

    return (
        list(group_summary.values()),
        list(agreement_summary.values()),
        gt_summary_rows,
        class_summary_rows,
    )


def print_aggregate_summary(group_rows, agreement_rows, target_ratio=0.15):
    print(f"\nFour-group counts at top {target_ratio:.0%}")
    for row in group_rows:
        if np.isclose(row["top_k_ratio"], target_ratio):
            print(
                f"  {row['model']} | {row['branch']} | {row['group']}: "
                f"{row['count']} ({row['fraction']:.3f})"
            )
    print(f"\nClassifier-alignment agreement at top {target_ratio:.0%}")
    for row in agreement_rows:
        if np.isclose(row["top_k_ratio"], target_ratio):
            print(
                f"  {row['model']} | {row['quadrant']}: count={row['count']} | "
                f"anomaly_rate={row['anomaly_rate']:.3f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze snippet-level score topology and ranking errors.")
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--output-dir", default="../../code/results/analyze/snippet_topology")
    parser.add_argument("--top-k-ratios", default="0.05,0.10,0.15", type=parse_ratios)
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
        model_configs = json.load(handle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = np.load(args.gt_path)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = {
        "snippet_rows": [],
        "topology_rows": [],
        "group_rows": [],
        "agreement_rows": [],
        "ground_truth_rows": [],
        "sequence_records": [],
        "summary_rows": [],
    }
    print("Snippet topology diagnostics")
    print(f"Device: {device}")
    print(f"Models: {len(model_configs)}")
    print(f"Top-k ratios: {args.top_k_ratios}")

    for config in model_configs:
        print("\n" + "=" * 80)
        print(f"Analyzing {config['name']}: {config['model_path']}")
        model = load_model(args, config, device)
        records = run_inference(model, dataloader, args, config, gt, device)
        result = analyze_records(config["name"], records, args.top_k_ratios, args.visual_length)
        print_summary(result["summary_rows"])
        for key in collected:
            collected[key].extend(result[key])
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "snippet_index_scores.csv", collected["snippet_rows"])
    write_csv(output_dir / "high_score_topology.csv", collected["topology_rows"])
    write_csv(output_dir / "topk_four_groups.csv", collected["group_rows"])
    write_csv(output_dir / "branch_agreement.csv", collected["agreement_rows"])
    write_csv(output_dir / "ground_truth_anomaly_topology.csv", collected["ground_truth_rows"])
    write_csv(output_dir / "summary.csv", collected["summary_rows"])
    group_summary, agreement_summary, gt_summary, class_summary = aggregate_rows(collected)
    write_csv(output_dir / "four_group_summary.csv", group_summary)
    write_csv(output_dir / "branch_agreement_summary.csv", agreement_summary)
    write_csv(output_dir / "ground_truth_topology_by_class.csv", gt_summary)
    write_csv(output_dir / "per_class_snippet_summary.csv", class_summary)
    with (output_dir / "high_score_index_sequences.json").open("w", encoding="utf-8") as handle:
        json.dump(collected["sequence_records"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "models_config.json").open("w", encoding="utf-8") as handle:
        json.dump(model_configs, handle, ensure_ascii=False, indent=2)

    print_aggregate_summary(group_summary, agreement_summary)

    print("\nSaved outputs:")
    for name in [
        "summary.csv",
        "snippet_index_scores.csv",
        "high_score_topology.csv",
        "high_score_index_sequences.json",
        "topk_four_groups.csv",
        "four_group_summary.csv",
        "branch_agreement.csv",
        "branch_agreement_summary.csv",
        "ground_truth_anomaly_topology.csv",
        "ground_truth_topology_by_class.csv",
        "per_class_snippet_summary.csv",
        "models_config.json",
    ]:
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
