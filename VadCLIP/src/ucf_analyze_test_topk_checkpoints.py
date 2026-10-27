import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ucf_analyze_score_separation import (
    LABEL_MAP,
    UCFRelativeTestDataset,
    build_model,
    get_prompt_text_for_config,
    get_prompt_weights_for_config,
    make_lengths,
)
from utils.tools import get_batch_mask


FRAMES_PER_SNIPPET = 16
BRANCHES = ("classifier", "alignment_true_class_logit")


def checkpoint_sort_key(path):
    numbers = re.findall(r"\d+", Path(path).stem)
    return int(numbers[-1]) if numbers else 10**9


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


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def safe_mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def jaccard(left, right):
    if right is None:
        return float("nan")
    union = left | right
    return safe_div(len(left & right), len(union))


def snippet_ground_truth(gt_frames, snippet_count):
    expected = snippet_count * FRAMES_PER_SNIPPET
    if len(gt_frames) != expected:
        raise ValueError(f"Expected {expected} GT frames, received {len(gt_frames)}.")
    fractions = np.asarray(gt_frames).reshape(snippet_count, FRAMES_PER_SNIPPET).mean(axis=1)
    return {
        "fraction": fractions.astype(np.float32),
        "any": fractions > 0,
        "majority": fractions >= 0.5,
    }


def load_state_dict(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception as error:
        print(f"Safe checkpoint load failed for {path}: {error}")
        print("Retrying with weights_only=False for this trusted local checkpoint.")
        payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    return payload


def load_checkpoint_model(args, checkpoint, config, device):
    model = build_model(args, args.model_type, device)
    state_dict = load_state_dict(checkpoint, device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Checkpoint compatibility: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    model.to(device)
    model.eval()
    return model


def run_inference(model, dataloader, args, config, gt, device):
    prompt_text = get_prompt_text_for_config(config)
    prompt_weights = get_prompt_weights_for_config(config, device)
    class_index = {label: index for index, label in enumerate(LABEL_MAP)}
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
            scores = {
                "classifier": torch.sigmoid(logits1[:, 0]).cpu().numpy(),
                "alignment_true_class_logit": logits2[:, class_index[str(label)]].cpu().numpy(),
            }

            frame_count = snippet_count * FRAMES_PER_SNIPPET
            gt_frames = np.asarray(gt[gt_cursor : gt_cursor + frame_count], dtype=np.int64)
            gt_cursor += frame_count
            records.append(
                {
                    "video_id": str(video_id),
                    "label": str(label),
                    "snippet_count": snippet_count,
                    "ground_truth": snippet_ground_truth(gt_frames, snippet_count),
                    "scores": scores,
                }
            )

    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)} frames.")
    return records


def discover_checkpoints(args):
    paths = []
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        if directory.exists():
            paths.extend(sorted(directory.glob(args.checkpoint_pattern), key=checkpoint_sort_key))
    paths.extend(Path(path) for path in args.checkpoint_path)

    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            if not path.exists():
                raise FileNotFoundError(path)
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("No checkpoints were found. Set --checkpoint-dir or --checkpoint-path.")
    return unique


def analyze_checkpoint(
    checkpoint_name,
    checkpoint_order,
    records,
    previous,
    cumulative,
    selection_counts,
    seen_checkpoints,
):
    detail_rows = []
    snippet_metadata = {}

    for record in records:
        if record["label"] == "Normal":
            continue
        count = record["snippet_count"]
        k = min(count, max(1, int(count / 16 + 1)))
        gt = record["ground_truth"]

        for branch in BRANCHES:
            scores = record["scores"][branch]
            selected_array = np.sort(np.argsort(-scores, kind="stable")[:k])
            selected = set(int(index) for index in selected_array)
            key = (branch, record["video_id"])
            previous_jaccard = jaccard(selected, previous.get(key))
            previous[key] = selected
            cumulative[key].update(selected)

            for index in selected:
                selection_counts[(branch, record["video_id"], index)] += 1
            seen_checkpoints[key] += 1
            for index in range(count):
                snippet_metadata[(branch, record["video_id"], index)] = {
                    "label": record["label"],
                    "gt_fraction": float(gt["fraction"][index]),
                }

            row = {
                "checkpoint": checkpoint_name,
                "checkpoint_order": checkpoint_order,
                "video_id": record["video_id"],
                "label": record["label"],
                "branch": branch,
                "snippet_count": count,
                "k": k,
                "selected_indices": json.dumps(sorted(selected)),
                "previous_checkpoint_jaccard": previous_jaccard,
                "selected_gt_frame_fraction_mean": safe_mean(gt["fraction"][list(selected)]),
            }
            for definition in ("any", "majority"):
                positive = set(np.flatnonzero(gt[definition]).tolist())
                true_selected = selected & positive
                cumulative_true = cumulative[key] & positive
                row.update(
                    {
                        f"gt_{definition}_count": len(positive),
                        f"selected_true_{definition}_count": len(true_selected),
                        f"topk_precision_{definition}": safe_div(len(true_selected), len(selected)),
                        f"anomaly_recall_{definition}": safe_div(len(true_selected), len(positive)),
                        f"cumulative_anomaly_coverage_{definition}": safe_div(
                            len(cumulative_true), len(positive)
                        ),
                        f"persistent_missed_{definition}_count": len(positive - cumulative[key]),
                        f"persistent_missed_ratio_{definition}": safe_div(
                            len(positive - cumulative[key]), len(positive)
                        ),
                    }
                )
            detail_rows.append(row)

    return detail_rows, snippet_metadata


def aggregate(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for key, group in grouped.items():
        row = dict(zip(group_fields, key))
        row["num_abnormal_videos"] = len(group)
        row["num_selected_snippets"] = sum(item["k"] for item in group)
        row["previous_checkpoint_jaccard_macro"] = safe_mean(
            item["previous_checkpoint_jaccard"] for item in group
        )
        row["selected_gt_frame_fraction_mean"] = safe_mean(
            item["selected_gt_frame_fraction_mean"] for item in group
        )
        for definition in ("any", "majority"):
            positives = sum(item[f"gt_{definition}_count"] for item in group)
            selected_true = sum(item[f"selected_true_{definition}_count"] for item in group)
            missed = sum(item[f"persistent_missed_{definition}_count"] for item in group)
            row[f"topk_precision_{definition}_micro"] = safe_div(
                selected_true, row["num_selected_snippets"]
            )
            row[f"anomaly_recall_{definition}_micro"] = safe_div(selected_true, positives)
            row[f"anomaly_recall_{definition}_macro"] = safe_mean(
                item[f"anomaly_recall_{definition}"] for item in group
            )
            row[f"cumulative_anomaly_coverage_{definition}_macro"] = safe_mean(
                item[f"cumulative_anomaly_coverage_{definition}"] for item in group
            )
            row[f"persistent_missed_ratio_{definition}_micro"] = safe_div(missed, positives)
        output.append(row)
    return output


def format_percent(value):
    return "N/A" if not np.isfinite(value) else f"{100 * value:.2f}%"


def print_summary(rows):
    print("\nTest-set MIL top-k versus temporal ground truth")
    print("Primary GT definition: a snippet is positive when at least 50% of its 16 frames are anomaly.")
    for row in rows:
        print(
            f"  {row['checkpoint']} | {row['branch']} | "
            f"precision={format_percent(row['topk_precision_majority_micro'])} | "
            f"recall={format_percent(row['anomaly_recall_majority_micro'])} | "
            f"cumulative_coverage={format_percent(row['cumulative_anomaly_coverage_majority_macro'])} | "
            f"persistent_missed={format_percent(row['persistent_missed_ratio_majority_micro'])} | "
            f"prev_jaccard={format_percent(row['previous_checkpoint_jaccard_macro'])}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the MIL top-k selected by saved checkpoints with UCF temporal GT."
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-pattern", default="*.pth")
    parser.add_argument("--checkpoint-path", action="append", default=[])
    parser.add_argument(
        "--model-type",
        default="baseline",
        choices=["baseline", "description", "prompt_replacement", "multi_prompt"],
    )
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--output-dir", default="../../code/results/analyze/test_topk_checkpoints")

    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--prototype-json", default="../../code/ucf_weighted_class_prompts.json")
    parser.add_argument("--prompt-weights-json", default=None)
    parser.add_argument("--prototype-source", default="manual")
    parser.add_argument("--prompt-mode", default="manual_caption")
    parser.add_argument(
        "--compact-prototype-types",
        nargs="+",
        default=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
    )
    parser.add_argument(
        "--long-prototype-types",
        nargs="+",
        default=["core_action", "actor_object_interaction", "temporal_progression"],
    )
    parser.add_argument("--prompt-replacement-mode", default="class_plus_prototype")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = discover_checkpoints(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = np.load(args.gt_path)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["model_type"] = args.model_type
    previous = {}
    cumulative = defaultdict(set)
    selection_counts = defaultdict(int)
    seen_checkpoints = defaultdict(int)
    all_details = []
    all_metadata = {}

    print("Test-set top-k checkpoint diagnostics")
    print("Device:", device)
    print("Checkpoints:", len(checkpoints))
    print("No gradients or parameter updates are performed.")

    for order, checkpoint in enumerate(checkpoints, start=1):
        print("\n" + "=" * 80)
        print(f"[{order}/{len(checkpoints)}] {checkpoint}")
        model = load_checkpoint_model(args, checkpoint, config, device)
        records = run_inference(model, dataloader, args, config, gt, device)
        details, metadata = analyze_checkpoint(
            checkpoint.stem,
            order,
            records,
            previous,
            cumulative,
            selection_counts,
            seen_checkpoints,
        )
        all_details.extend(details)
        all_metadata.update(metadata)
        checkpoint_summary = aggregate(details, ["checkpoint", "checkpoint_order", "branch"])
        print_summary(checkpoint_summary)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate(all_details, ["checkpoint", "checkpoint_order", "branch"])
    class_rows = aggregate(all_details, ["checkpoint", "checkpoint_order", "branch", "label"])
    frequency_rows = []
    for key, metadata in sorted(all_metadata.items()):
        branch, video_id, index = key
        seen = seen_checkpoints[(branch, video_id)]
        count = selection_counts[key]
        frequency_rows.append(
            {
                "video_id": video_id,
                "label": metadata["label"],
                "branch": branch,
                "snippet_index": index,
                "gt_anomaly_frame_fraction": metadata["gt_fraction"],
                "selected_checkpoint_count": count,
                "seen_checkpoint_count": seen,
                "selection_frequency": safe_div(count, seen),
            }
        )

    write_csv(output_dir / "checkpoint_summary.csv", summary_rows)
    write_csv(output_dir / "per_class_checkpoint_summary.csv", class_rows)
    write_csv(output_dir / "per_video_checkpoint_details.csv", all_details)
    write_csv(output_dir / "snippet_selection_frequency.csv", frequency_rows)
    with (output_dir / "analysis_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": args.model_type,
                "checkpoints": [str(path) for path in checkpoints],
                "mil_topk": "k = floor(valid_length / 16) + 1",
                "gt_any": "at least one of 16 frames is anomalous",
                "gt_majority": "at least 8 of 16 frames are anomalous",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(summary_rows)
    print("\nSaved outputs:")
    for name in (
        "checkpoint_summary.csv",
        "per_class_checkpoint_summary.csv",
        "per_video_checkpoint_details.csv",
        "snippet_selection_frequency.csv",
        "analysis_config.json",
    ):
        print(" ", output_dir / name)


if __name__ == "__main__":
    main()
