import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ucf_analyze_score_separation import (
    LABEL_MAP,
    UCFRelativeTestDataset,
    get_prompt_text_for_config,
    get_prompt_weights_for_config,
    load_model,
    make_lengths,
)
from utils.tools import get_batch_mask


FRAMES_PER_SNIPPET = 16
CLASS_NAMES = list(LABEL_MAP.keys())
DELTA_COLUMNS = [
    "classifier_segment_delta",
    "classifier_fixed_delta",
    "classifier_global_delta",
    "alignment_segment_delta",
    "alignment_fixed_delta",
    "alignment_global_delta",
    "target_margin_segment_delta",
    "target_margin_fixed_delta",
    "target_margin_global_delta",
    "classifier_outside_abs_delta",
    "classifier_neighbor_abs_delta",
    "classifier_topk_overlap",
    "alignment_topk_overlap",
]


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


def selected_indices(scores, ratio, largest=True):
    scores = np.asarray(scores)
    count = min(scores.size, max(1, int(np.ceil(scores.size * ratio))))
    order = np.argsort(-scores if largest else scores, kind="stable")
    return np.sort(order[:count])


def contiguous_runs(indices):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    if indices.size == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    return [chunk for chunk in np.split(indices, split_points) if chunk.size]


def snippet_ground_truth(gt_frames, snippet_count):
    expected = snippet_count * FRAMES_PER_SNIPPET
    if len(gt_frames) != expected:
        raise ValueError(f"Expected {expected} GT frames, received {len(gt_frames)}")
    fractions = np.asarray(gt_frames).reshape(snippet_count, FRAMES_PER_SNIPPET).mean(axis=1)
    return fractions.astype(np.float32)


def flatten_valid(visual, snippet_count):
    return visual.reshape(-1, visual.shape[-1])[:snippet_count]


def reshape_visual(flat_visual, template):
    return flat_visual.reshape(template.shape)


def forward_scores(model, visual, snippet_count, args, prompt_text, prompt_weights):
    lengths = make_lengths(snippet_count, args.visual_length)
    padding_mask = get_batch_mask(lengths, args.visual_length).to(visual.device)
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
    classifier = torch.sigmoid(logits1.squeeze(-1)).float().cpu().numpy()
    probabilities = logits2.softmax(dim=-1).float().cpu().numpy()
    alignment = 1.0 - probabilities[:, 0]
    return {
        "classifier": classifier,
        "alignment": alignment,
        "class_probabilities": probabilities,
    }


def build_replacement(flat_visual, start, end, method, context_radius):
    valid_count = flat_visual.shape[0]
    segment_length = end - start + 1
    if method == "linear":
        left_index = start - 1 if start > 0 else None
        right_index = end + 1 if end + 1 < valid_count else None
        if left_index is None and right_index is None:
            return flat_visual.mean(dim=0, keepdim=True).repeat(segment_length, 1)
        if left_index is None:
            return flat_visual[right_index].unsqueeze(0).repeat(segment_length, 1)
        if right_index is None:
            return flat_visual[left_index].unsqueeze(0).repeat(segment_length, 1)
        weights = torch.linspace(
            1.0 / (segment_length + 1),
            segment_length / (segment_length + 1),
            segment_length,
            device=flat_visual.device,
            dtype=flat_visual.dtype,
        ).unsqueeze(1)
        return flat_visual[left_index].unsqueeze(0) * (1 - weights) + flat_visual[
            right_index
        ].unsqueeze(0) * weights

    if method == "context_mean":
        context_indices = list(range(max(0, start - context_radius), start))
        context_indices.extend(range(end + 1, min(valid_count, end + 1 + context_radius)))
        if context_indices:
            context = flat_visual[context_indices].mean(dim=0, keepdim=True)
        else:
            context = flat_visual.mean(dim=0, keepdim=True)
        return context.repeat(segment_length, 1)

    raise ValueError(f"Unsupported mask method: {method}")


def mask_segment(visual, snippet_count, start, end, method, context_radius):
    masked = visual.clone()
    valid = flatten_valid(masked, snippet_count).clone()
    valid[start : end + 1] = build_replacement(
        valid, start, end, method, context_radius
    )
    padded = masked.reshape(-1, masked.shape[-1])
    padded[:snippet_count] = valid
    return reshape_visual(padded, visual)


def run_type(run, gt_fraction, label, high_mask):
    run_gt = gt_fraction[run]
    if np.all(high_mask[run]):
        if np.all(run_gt == 0):
            suffix = "normal_video" if label == "Normal" else "abnormal_video"
            return f"false_high_{suffix}"
        if np.all(run_gt >= 0.5):
            return "true_high"
        return "mixed_high"
    return "missed_anomaly"


def candidate_runs(scores, gt_fraction, label, ratio, max_per_group, min_run_length, include_missed):
    high_indices = selected_indices(scores, ratio)
    high_mask = np.zeros(scores.size, dtype=bool)
    high_mask[high_indices] = True
    candidates = []

    for run in contiguous_runs(high_indices):
        if run.size < min_run_length:
            continue
        candidates.append(
            {
                "indices": run,
                "group": run_type(run, gt_fraction, label, high_mask),
                "priority": float(np.mean(scores[run])),
            }
        )

    if include_missed and label != "Normal":
        missed_indices = np.where((gt_fraction >= 0.5) & ~high_mask)[0]
        for run in contiguous_runs(missed_indices):
            if run.size < min_run_length:
                continue
            candidates.append(
                {
                    "indices": run,
                    "group": "missed_anomaly",
                    "priority": float(np.mean(scores[run])),
                }
            )

    selected = []
    groups = sorted({candidate["group"] for candidate in candidates})
    for group in groups:
        group_candidates = [candidate for candidate in candidates if candidate["group"] == group]
        group_candidates.sort(
            key=lambda candidate: (candidate["priority"], candidate["indices"].size),
            reverse=True,
        )
        selected.extend(group_candidates[:max_per_group])
    return selected


def mean_at(values, indices):
    return float(np.mean(np.asarray(values)[np.asarray(indices, dtype=np.int64)]))


def overlap_ratio(first, second):
    first = set(np.asarray(first, dtype=np.int64).tolist())
    second = set(np.asarray(second, dtype=np.int64).tolist())
    return float(len(first & second) / max(1, len(first)))


def finite_or_nan(value):
    value = float(value)
    return value if np.isfinite(value) else np.nan


def response_row(
    model_name,
    video_id,
    label,
    group,
    run,
    method,
    original,
    masked,
    ratio,
    neighbor_radius,
):
    classifier_topk_original = selected_indices(original["classifier"], ratio)
    classifier_topk_masked = selected_indices(masked["classifier"], ratio)
    alignment_topk_original = selected_indices(original["alignment"], ratio)
    alignment_topk_masked = selected_indices(masked["alignment"], ratio)

    target_index = CLASS_NAMES.index(label)
    if label == "Normal":
        target_margin_original = np.full_like(original["classifier"], np.nan)
        target_margin_masked = np.full_like(masked["classifier"], np.nan)
    else:
        target_margin_original = (
            original["class_probabilities"][:, target_index]
            - original["class_probabilities"][:, 0]
        )
        target_margin_masked = (
            masked["class_probabilities"][:, target_index]
            - masked["class_probabilities"][:, 0]
        )

    start = int(run[0])
    end = int(run[-1])
    segment_indices = np.arange(start, end + 1)
    outside_indices = np.concatenate(
        [np.arange(0, start), np.arange(end + 1, len(original["classifier"]))]
    )
    neighbor_indices = np.concatenate(
        [
            np.arange(max(0, start - neighbor_radius), start),
            np.arange(end + 1, min(len(original["classifier"]), end + 1 + neighbor_radius)),
        ]
    )

    def delta(values_original, values_masked, fixed_indices, new_indices):
        return mean_at(values_original, fixed_indices) - mean_at(values_masked, new_indices)

    row = {
        "model": model_name,
        "video_id": video_id,
        "class_name": label,
        "candidate_group": group,
        "mask_method": method,
        "segment_start": start,
        "segment_end": end,
        "segment_length": int(len(segment_indices)),
        "classifier_segment_original": mean_at(original["classifier"], segment_indices),
        "classifier_segment_masked": mean_at(masked["classifier"], segment_indices),
        "classifier_segment_delta": delta(
            original["classifier"], masked["classifier"], segment_indices, segment_indices
        ),
        "classifier_fixed_delta": delta(
            original["classifier"],
            masked["classifier"],
            classifier_topk_original,
            classifier_topk_original,
        ),
        "classifier_global_delta": delta(
            original["classifier"],
            masked["classifier"],
            classifier_topk_original,
            classifier_topk_masked,
        ),
        "alignment_segment_delta": delta(
            original["alignment"], masked["alignment"], segment_indices, segment_indices
        ),
        "alignment_fixed_delta": delta(
            original["alignment"],
            masked["alignment"],
            alignment_topk_original,
            alignment_topk_original,
        ),
        "alignment_global_delta": delta(
            original["alignment"],
            masked["alignment"],
            alignment_topk_original,
            alignment_topk_masked,
        ),
        "classifier_topk_overlap": overlap_ratio(
            classifier_topk_original, classifier_topk_masked
        ),
        "alignment_topk_overlap": overlap_ratio(alignment_topk_original, alignment_topk_masked),
        "classifier_outside_abs_delta": float(
            np.mean(np.abs(original["classifier"][outside_indices] - masked["classifier"][outside_indices]))
        )
        if outside_indices.size
        else np.nan,
        "classifier_neighbor_abs_delta": float(
            np.mean(np.abs(original["classifier"][neighbor_indices] - masked["classifier"][neighbor_indices]))
        )
        if neighbor_indices.size
        else np.nan,
    }

    if label == "Normal":
        row.update(
            {
                "target_margin_segment_delta": np.nan,
                "target_margin_fixed_delta": np.nan,
                "target_margin_global_delta": np.nan,
            }
        )
    else:
        row.update(
            {
                "target_margin_segment_delta": delta(
                    target_margin_original,
                    target_margin_masked,
                    segment_indices,
                    segment_indices,
                ),
                "target_margin_fixed_delta": delta(
                    target_margin_original,
                    target_margin_masked,
                    classifier_topk_original,
                    classifier_topk_original,
                ),
                "target_margin_global_delta": delta(
                    target_margin_original,
                    target_margin_masked,
                    classifier_topk_original,
                    classifier_topk_masked,
                ),
            }
        )
    return row


def summarize(rows, include_class=False):
    summary = []
    if include_class:
        keys = sorted(
            {
                (row["model"], row["class_name"], row["candidate_group"], row["mask_method"])
                for row in rows
            }
        )
    else:
        keys = sorted(
            {(row["model"], row["candidate_group"], row["mask_method"]) for row in rows}
        )

    for key in keys:
        if include_class:
            model_name, class_name, group, method = key
        else:
            model_name, group, method = key
            class_name = None
        subset = [
            row
            for row in rows
            if row["model"] == model_name
            and row["candidate_group"] == group
            and row["mask_method"] == method
            and (not include_class or row["class_name"] == class_name)
        ]
        output = {
            "model": model_name,
            "candidate_group": group,
            "mask_method": method,
            "segment_count": len(subset),
            "mean_segment_length": float(np.mean([row["segment_length"] for row in subset])),
        }
        if include_class:
            output["class_name"] = class_name
        for column in DELTA_COLUMNS:
            values = np.asarray([row[column] for row in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            output[f"{column}_mean"] = float(np.mean(values)) if values.size else np.nan
            output[f"{column}_median"] = float(np.median(values)) if values.size else np.nan
            output[f"{column}_p25"] = float(np.quantile(values, 0.25)) if values.size else np.nan
            output[f"{column}_p75"] = float(np.quantile(values, 0.75)) if values.size else np.nan
            if column in {"classifier_segment_delta", "classifier_global_delta"}:
                for threshold in (0.05, 0.10, 0.20):
                    threshold_name = str(threshold).replace(".", "_")
                    output[f"{column}_gt_{threshold_name}_ratio"] = (
                        float(np.mean(values > threshold)) if values.size else np.nan
                    )
        summary.append(output)
    return summary


def print_summary(model_name, summary_rows):
    print(f"\nCounterfactual response: {model_name}")
    print(
        "  group / method | n | cls segment delta | cls global delta | "
        "align global delta | cls top-k overlap"
    )
    for row in summary_rows:
        if row["model"] != model_name:
            continue
        print(
            f"  {row['candidate_group']:<32} {row['mask_method']:<12} "
            f"n={row['segment_count']:<4d} "
            f"seg={row['classifier_segment_delta_mean']:+.4f} "
            f"global={row['classifier_global_delta_mean']:+.4f} "
            f"align={row['alignment_global_delta_mean']:+.4f} "
            f"overlap={row['classifier_topk_overlap_mean']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure VadCLIP response to counterfactual masking of snippet segments."
    )
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument(
        "--output-dir", default="../../code/results/analyze/counterfactual_masking"
    )
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
    parser.add_argument("--mask-methods", default="linear,context_mean")
    parser.add_argument("--context-radius", default=3, type=int)
    parser.add_argument("--neighbor-radius", default=4, type=int)
    parser.add_argument("--max-segments-per-group", default=1, type=int)
    parser.add_argument("--min-run-length", default=1, type=int)
    parser.add_argument("--include-missed-anomaly", action="store_true")
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
    if not 0 < args.top_k_ratio < 1:
        raise ValueError("--top-k-ratio must be between 0 and 1")
    if args.max_segments_per_group < 1:
        raise ValueError("--max-segments-per-group must be at least 1")
    methods = [method.strip() for method in args.mask_methods.split(",") if method.strip()]
    unsupported = sorted(set(methods) - {"linear", "context_mean"})
    if unsupported:
        raise ValueError(f"Unsupported mask methods: {unsupported}")

    with open(args.models_json, "r", encoding="utf-8") as handle:
        configs = json.load(handle)
    gt = np.load(args.gt_path)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Counterfactual segment masking diagnostics")
    print("Device:", device)
    print("Models:", len(configs))
    print("Mask methods:", methods)
    print("Top-k ratio:", args.top_k_ratio)
    print("Ground truth is used only to name diagnostic groups, never as model input.")
    print("No AUC or AP is computed; all outputs are before/after response deltas.")

    all_rows = []
    for config in configs:
        model_name = config["name"]
        print(f"\nLoading model: {model_name}")
        model = load_model(args, config, device)
        prompt_text = get_prompt_text_for_config(config)
        prompt_weights = get_prompt_weights_for_config(config, device)
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

                frame_count = snippet_count * FRAMES_PER_SNIPPET
                gt_frames = gt[gt_cursor : gt_cursor + frame_count]
                gt_cursor += frame_count
                gt_fraction = snippet_ground_truth(gt_frames, snippet_count)
                original = forward_scores(
                    model, visual, snippet_count, args, prompt_text, prompt_weights
                )
                candidates = candidate_runs(
                    original["classifier"],
                    gt_fraction,
                    str(label),
                    args.top_k_ratio,
                    args.max_segments_per_group,
                    args.min_run_length,
                    args.include_missed_anomaly,
                )

                for candidate in candidates:
                    run = candidate["indices"]
                    for method in methods:
                        masked_visual = mask_segment(
                            visual,
                            snippet_count,
                            int(run[0]),
                            int(run[-1]),
                            method,
                            args.context_radius,
                        )
                        masked = forward_scores(
                            model,
                            masked_visual,
                            snippet_count,
                            args,
                            prompt_text,
                            prompt_weights,
                        )
                        row = response_row(
                            model_name,
                            str(video_id),
                            str(label),
                            candidate["group"],
                            run,
                            method,
                            original,
                            masked,
                            args.top_k_ratio,
                            args.neighbor_radius,
                        )
                        run_indices = torch.as_tensor(
                            run, dtype=torch.long, device=visual.device
                        )
                        original_segment = flatten_valid(visual, snippet_count)[run_indices]
                        masked_segment = flatten_valid(masked_visual, snippet_count)[run_indices]
                        row["feature_replacement_l2"] = float(
                            torch.linalg.vector_norm(
                                original_segment.float() - masked_segment.float(), dim=-1
                            ).mean()
                        )
                        original_norm = torch.nn.functional.normalize(
                            original_segment.float(), dim=-1
                        )
                        masked_norm = torch.nn.functional.normalize(masked_segment.float(), dim=-1)
                        row["feature_replacement_cosine_change"] = float(
                            (1.0 - (original_norm * masked_norm).sum(dim=-1)).mean()
                        )
                        all_rows.append(row)

        if gt_cursor != len(gt):
            raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)}")
        current_summary = summarize([row for row in all_rows if row["model"] == model_name])
        print_summary(model_name, current_summary)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = summarize(all_rows)
    per_class_rows = summarize(all_rows, include_class=True)
    write_csv(output_dir / "counterfactual_segment_details.csv", all_rows)
    write_csv(output_dir / "counterfactual_response_summary.csv", summary_rows)
    write_csv(output_dir / "counterfactual_response_per_class.csv", per_class_rows)
    (output_dir / "models_config.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "top_k_ratio": args.top_k_ratio,
                "mask_methods": methods,
                "context_radius": args.context_radius,
                "neighbor_radius": args.neighbor_radius,
                "max_segments_per_group": args.max_segments_per_group,
                "min_run_length": args.min_run_length,
                "include_missed_anomaly": args.include_missed_anomaly,
                "ground_truth_usage": "diagnostic grouping only",
                "auc_ap_computed": False,
                "feature_source": "existing CLIP .npy features",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nSaved counterfactual diagnostics to:", output_dir)


if __name__ == "__main__":
    main()
