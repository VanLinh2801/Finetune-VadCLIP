"""Measure whether local temporal attention helps selected true/false-high snippets."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from model import CLIPVAD
from ucf_analyze_temporal_order import (
    CLASS_COUNT,
    EMBED_DIM,
    PROMPT_POSTFIX,
    PROMPT_PREFIX,
    SEGMENTS,
    VISUAL_LENGTH,
)
from ucf_analyze_temporal_graph_factorial import (
    capture_model_stages,
    collect_gt_by_video,
    graph_forward,
    pad_video,
)


def apply_attention_intervention(model, visual, mode, target_indices=None):
    """Run temporal blocks with a controlled attention mask; visual is [B,L,D]."""
    batch_size, seq_len, _ = visual.shape
    if any(block.attn.num_heads != 1 for block in model.temporal.resblocks):
        raise ValueError("This diagnostic currently expects VadCLIP visual_head=1.")

    positions = torch.arange(seq_len, device=visual.device).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = model.frame_position_embeddings(positions).permute(1, 0, 2)
    sequence = visual.float().permute(1, 0, 2) + position_embeddings
    originals = []

    for block in model.temporal.resblocks:
        attention = block.attn
        original_forward = attention.forward
        originals.append((attention, original_forward))

        def patched_forward(
            query, key, value, *args, _original=original_forward,
            _attention=attention, **kwargs
        ):
            attention_mask = kwargs.get("attn_mask")
            heads = _attention.num_heads
            current_batch = query.shape[1]
            length = query.shape[0]
            if attention_mask is None:
                attention_mask = torch.zeros(
                    (current_batch * heads, length, length),
                    device=query.device,
                    dtype=query.dtype,
                )
            elif attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(0).expand(
                    current_batch * heads, -1, -1
                ).clone()
            else:
                attention_mask = attention_mask.clone()

            if attention_mask.dtype == torch.bool:
                blocked = True
            else:
                blocked = float("-inf")

            if mode == "target_self_only":
                if target_indices is None or len(target_indices) != current_batch:
                    raise ValueError("target_indices must contain one query index per batch item.")
                targets = torch.as_tensor(target_indices, device=query.device, dtype=torch.long)
                batch_rows = torch.arange(current_batch, device=query.device)
                for head in range(heads):
                    rows = batch_rows * heads + head
                    attention_mask[rows, targets, :] = blocked
                    attention_mask[rows, targets, targets] = (
                        False if attention_mask.dtype == torch.bool else 0.0
                    )
            elif mode == "all_self_only":
                attention_mask.fill_(blocked)
                diagonal = torch.arange(length, device=query.device)
                attention_mask[:, diagonal, diagonal] = (
                    False if attention_mask.dtype == torch.bool else 0.0
                )
            elif mode == "past_only":
                indices = torch.arange(length, device=query.device)
                disallowed = indices.unsqueeze(0) > indices.unsqueeze(1)
                if attention_mask.dtype == torch.bool:
                    attention_mask |= disallowed.unsqueeze(0)
                else:
                    attention_mask.masked_fill_(disallowed.unsqueeze(0), blocked)
            elif mode == "future_only":
                indices = torch.arange(length, device=query.device)
                disallowed = indices.unsqueeze(1) > indices.unsqueeze(0)
                if attention_mask.dtype == torch.bool:
                    attention_mask |= disallowed.unsqueeze(0)
                else:
                    attention_mask.masked_fill_(disallowed.unsqueeze(0), blocked)
            else:
                raise ValueError(f"Unknown attention intervention: {mode}")

            kwargs["attn_mask"] = attention_mask
            return _original(query, key, value, *args, **kwargs)

        attention.forward = patched_forward

    try:
        with torch.no_grad():
            output, _ = model.temporal((sequence, None))
    finally:
        for attention, original_forward in originals:
            attention.forward = original_forward
    return output.permute(1, 0, 2).detach()


def scores_from_encoded(model, encoded):
    with torch.no_grad():
        logits = model.classifier(encoded + model.mlp2(encoded)).squeeze(-1)
        return torch.sigmoid(logits).detach()


def score_target_replacements(model, original_temporal, replacement_temporal, target_indices,
                              similarity_adj, distance_adj, device):
    """Replace only each target's temporal vector; keep all other graph inputs fixed."""
    batch_size = len(target_indices)
    targets = torch.as_tensor(target_indices, device=device, dtype=torch.long)
    batch_indices = torch.arange(batch_size, device=device)
    temporal = original_temporal.expand(batch_size, -1, -1).clone()
    temporal[batch_indices, targets] = replacement_temporal[batch_indices, targets]
    with torch.no_grad():
        encoded = graph_forward(
            model,
            temporal,
            similarity_adj.expand(batch_size, -1, -1),
            distance_adj.expand(batch_size, -1, -1),
        )
        scores = scores_from_encoded(model, encoded)
        return scores[batch_indices, targets].cpu().numpy()


def build_targets(groups, count):
    target_map = {}
    for group, (start, end) in groups.items():
        if end >= count:
            raise ValueError(f"Selected {group} segment ends beyond video length.")
        for index in range(start, end + 1):
            if index in target_map:
                raise ValueError(f"Overlapping selected target snippet: {index}")
            target_map[index] = group
    return target_map


def select_topk_targets(scores, gt_labels, top_k_ratio):
    """Select each video's highest-scoring snippets, then label them for analysis only."""
    count = len(gt_labels)
    if len(scores) < count:
        raise ValueError(f"Got {len(scores)} scores for {count} ground-truth snippets.")
    k = max(1, int(np.ceil(count * top_k_ratio)))
    selected = np.argpartition(scores[:count], -k)[-k:]
    return {
        int(index): "true_high" if int(gt_labels[index]) == 1 else "false_high"
        for index in selected
    }


def analyze_video(model, features, groups, gt_labels, device, target_batch_size,
                  selection_mode="fixed", top_k_ratio=0.15):
    count = len(features)
    if len(gt_labels) != count:
        raise ValueError(f"Got {len(gt_labels)} ground-truth snippets for {count} features.")
    target_map = build_targets(groups, count) if selection_mode == "fixed" else None

    padded_np, lengths_np = pad_video(features)
    padded = torch.from_numpy(padded_np).to(device)
    lengths = torch.as_tensor(lengths_np, device=device, dtype=torch.long)
    captured = capture_model_stages(model, padded, lengths)
    original_encoded = captured["encoded"]
    original_scores = scores_from_encoded(model, original_encoded).cpu().numpy().reshape(-1)[:count]
    if selection_mode == "top-k":
        target_map = select_topk_targets(original_scores, gt_labels, top_k_ratio)
    for index, group in target_map.items():
        expected = 0 if group == "false_high" else 1
        if int(gt_labels[index]) != expected:
            raise ValueError(f"Ground-truth mismatch at selected {group} snippet {index}.")

    rows = []
    target_indices_by_block = defaultdict(list)
    for global_index, group in target_map.items():
        target_indices_by_block[global_index // VISUAL_LENGTH].append((global_index, group))

    for block_index, block_targets in target_indices_by_block.items():
        length = int(lengths_np[block_index])
        raw_block = padded[block_index : block_index + 1]
        sim_adj = captured["similarity_adj"][block_index : block_index + 1]
        dist_adj = captured["distance_adj"][block_index : block_index + 1]

        original_temporal = captured["temporal"][block_index : block_index + 1]
        # Directional masks apply at every query/layer, preventing future/past information
        # from leaking indirectly through contextualized key/value representations.
        directional_temporal = {}
        for mode in ("past_only", "future_only"):
            directional_temporal[mode] = apply_attention_intervention(model, raw_block, mode)

        for start in range(0, len(block_targets), target_batch_size):
            batch_targets = block_targets[start : start + target_batch_size]
            local_indices = [global_index - block_index * VISUAL_LENGTH for global_index, _ in batch_targets]
            repeated_raw = raw_block.expand(len(local_indices), -1, -1).contiguous()
            self_only_temporal = apply_attention_intervention(
                model, repeated_raw, "target_self_only", target_indices=local_indices
            )
            selected_self_scores = score_target_replacements(
                model, original_temporal, self_only_temporal, local_indices,
                sim_adj, dist_adj, device,
            )
            selected_directional_scores = {}
            for mode, temporal in directional_temporal.items():
                repeated_temporal = temporal.expand(len(local_indices), -1, -1)
                selected_directional_scores[mode] = score_target_replacements(
                    model, original_temporal, repeated_temporal, local_indices,
                    sim_adj, dist_adj, device,
                )

            for offset, (global_index, group) in enumerate(batch_targets):
                local_index = global_index - block_index * VISUAL_LENGTH
                baseline = float(original_scores[global_index])
                self_score = float(selected_self_scores[offset])
                past_score = float(selected_directional_scores["past_only"][offset])
                future_score = float(selected_directional_scores["future_only"][offset])
                rows.append({
                    "group": group,
                    "snippet_index": global_index,
                    "original_score": baseline,
                    "self_only_score": self_score,
                    "past_only_score": past_score,
                    "future_only_score": future_score,
                    "context_gain": baseline - self_score,
                    "future_context_effect": baseline - past_score,
                    "past_context_effect": baseline - future_score,
                })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    result = {}
    for group, members in grouped.items():
        result[group] = {
            "video_count": len(members),
            "target_snippet_count": sum(member["snippet_count"] for member in members),
        }
        for metric in (
            "original_score", "self_only_score", "past_only_score", "future_only_score",
            "context_gain", "future_context_effect", "past_context_effect",
        ):
            values = np.asarray([member[metric] for member in members], dtype=np.float64)
            result[group][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "positive_fraction": float((values > 0).mean()) if metric.endswith("effect") or metric == "context_gain" else None,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--selection-mode", choices=("fixed", "top-k"), default="fixed")
    parser.add_argument("--top-k-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.target_batch_size < 1:
        raise ValueError("--target-batch-size must be positive")
    if not 0 < args.top_k_ratio <= 1:
        raise ValueError("--top-k-ratio must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VadCLIP graph adjacency requires CUDA in this implementation.")
    with Path(args.test_list).open(newline="", encoding="utf-8-sig") as handle:
        test_rows = list(csv.DictReader(handle))
    rows_by_video = {row["video_id"]: row for row in test_rows}
    gt_by_video = collect_gt_by_video(
        test_rows, args.feature_root, np.load(args.gt_path, mmap_mode="r")
    )

    model = CLIPVAD(
        CLASS_COUNT, EMBED_DIM, VISUAL_LENGTH, 512, 1, 2, 8, PROMPT_PREFIX, PROMPT_POSTFIX, device
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    model.eval()

    video_ids = list(SEGMENTS) if args.selection_mode == "fixed" else [
        row["video_id"] for row in test_rows
    ]
    snippet_rows = []
    video_group_rows = []
    for video_number, video_id in enumerate(video_ids, start=1):
        if video_id not in rows_by_video:
            raise KeyError(f"Video missing from test list: {video_id}")
        row = rows_by_video[video_id]
        features = np.asarray(
            np.load(Path(args.feature_root) / row["path"], mmap_mode="r"), dtype=np.float32
        )
        detailed = analyze_video(
            model,
            features,
            SEGMENTS.get(video_id, {}) if args.selection_mode == "fixed" else None,
            gt_by_video[video_id],
            device,
            args.target_batch_size,
            selection_mode=args.selection_mode,
            top_k_ratio=args.top_k_ratio,
        )
        for item in detailed:
            snippet_rows.append({"video_id": video_id, "class_name": row["label"], **item})
        by_group = defaultdict(list)
        for item in detailed:
            by_group[item["group"]].append(item)
        for group, members in by_group.items():
            video_group_rows.append({
                "video_id": video_id,
                "class_name": row["label"],
                "group": group,
                "snippet_count": len(members),
                **{
                    metric: float(np.mean([member[metric] for member in members]))
                    for metric in (
                        "original_score", "self_only_score", "past_only_score", "future_only_score",
                        "context_gain", "future_context_effect", "past_context_effect",
                    )
                },
            })
        print(f"Analyzed {video_number}/{len(video_ids)}: {video_id} ({row['label']})", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targeted_temporal_context_snippets.csv": snippet_rows,
        "targeted_temporal_context_video_groups.csv": video_group_rows,
    }
    for filename, data in outputs.items():
        path = output_dir / filename
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    macro = summarize(video_group_rows)
    summary = {
        "model_path": str(args.model_path),
        "videos": len(video_ids),
        "videos_with_selected_targets": len({row["video_id"] for row in video_group_rows}),
        "selection_mode": args.selection_mode,
        "top_k_ratio": args.top_k_ratio if args.selection_mode == "top-k" else None,
        "target_snippets": len(snippet_rows),
        "target_snippets_by_group": {
            group: sum(row["group"] == group for row in snippet_rows)
            for group in ("true_high", "false_high")
        },
        "metric_note": "Context effects are baseline classifier score minus intervened score; positive means the removed context raised the original score.",
        "intervention_note": "For scoring, only the target snippet's post-temporal vector is replaced; all other snippet vectors and both original graph adjacency matrices are held fixed. Past-only/future-only temporal masks are applied to all queries in the block to prevent indirect context leakage.",
        "macro_video_group_summary": macro,
        "selectivity_true_minus_false": {},
        "files": {name.removesuffix(".csv"): str(output_dir / name) for name in outputs},
    }
    groups_by_video = defaultdict(dict)
    for row in video_group_rows:
        groups_by_video[row["video_id"]][row["group"]] = row
    paired_videos = [
        groups for groups in groups_by_video.values()
        if "true_high" in groups and "false_high" in groups
    ]
    summary["selectivity_true_minus_false"]["paired_video_count"] = len(paired_videos)
    for metric in ("context_gain", "future_context_effect", "past_context_effect"):
        deltas = [
            groups["true_high"][metric] - groups["false_high"][metric]
            for groups in paired_videos
        ]
        summary["selectivity_true_minus_false"][metric] = (
            float(np.mean(deltas)) if deltas else None
        )
    summary_path = output_dir / "targeted_temporal_context_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nSelection: {args.selection_mode}"
        + (f" ({args.top_k_ratio:.0%} top scores per video)" if args.selection_mode == "top-k" else "")
    )
    print(f"Videos analyzed: {len(video_ids)}; selected snippets: {len(snippet_rows)}")
    print("Macro per-video context effects (score drop after context intervention):")
    for group in ("true_high", "false_high"):
        if group not in macro:
            print(f"{group}: no selected snippets")
            continue
        stats = macro[group]
        print(
            f"{group}: context={stats['context_gain']['mean']:+.5f}; "
            f"remove future={stats['future_context_effect']['mean']:+.5f}; "
            f"remove past={stats['past_context_effect']['mean']:+.5f}"
        )
    print(
        "Paired true-high minus false-high context gain "
        f"(n={len(paired_videos)} videos):",
        summary["selectivity_true_minus_false"]["context_gain"],
    )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
