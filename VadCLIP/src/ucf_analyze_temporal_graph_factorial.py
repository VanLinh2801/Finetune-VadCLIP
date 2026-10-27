"""Factorial inference diagnostics for temporal-order and graph effects in VadCLIP."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import CLIPVAD
from ucf_analyze_temporal_order import (
    CLASS_COUNT,
    EMBED_DIM,
    LABEL_MAP,
    PROMPT_POSTFIX,
    PROMPT_PREFIX,
    SEGMENTS,
    VISUAL_LENGTH,
)


def read_test_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pad_video(features):
    count, width = features.shape
    blocks = (count + VISUAL_LENGTH - 1) // VISUAL_LENGTH
    padded = np.zeros((blocks, VISUAL_LENGTH, width), dtype=np.float32)
    lengths = np.empty(blocks, dtype=np.int64)
    for block in range(blocks):
        start = block * VISUAL_LENGTH
        end = min(start + VISUAL_LENGTH, count)
        padded[block, : end - start] = features[start:end]
        lengths[block] = end - start
    return padded, lengths


def capture_model_stages(model, visual, lengths):
    captured = {}
    hooks = [
        model.temporal.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "temporal", output[0].permute(1, 0, 2).detach()
            )
        ),
        model.gc1.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("similarity_adj", inputs[1].detach())
        ),
        model.gc3.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("distance_adj", inputs[1].detach())
        ),
    ]
    try:
        with torch.no_grad():
            encoded = model.encode_video(visual, None, lengths)
    finally:
        for hook in hooks:
            hook.remove()
    captured["encoded"] = encoded.detach()
    return captured


def graph_forward(model, temporal, similarity_adj, distance_adj):
    sim_first = model.gelu(model.gc1(temporal, similarity_adj))
    dist_first = model.gelu(model.gc3(temporal, distance_adj))
    sim_second = model.gelu(model.gc2(sim_first, similarity_adj))
    dist_second = model.gelu(model.gc4(dist_first, distance_adj))
    return model.linear(torch.cat((sim_second, dist_second), dim=-1))


def classifier_scores(model, encoded, count):
    logits = model.classifier(encoded + model.mlp2(encoded)).reshape(-1)
    return torch.sigmoid(logits[:count]).detach().cpu().numpy()


def identity_adjacency(batch, device):
    return torch.eye(VISUAL_LENGTH, device=device).unsqueeze(0).expand(batch, -1, -1)


def shuffle_padded(padded, lengths, seed):
    rng = np.random.default_rng(seed)
    shuffled = padded.copy()
    permutations = np.tile(np.arange(VISUAL_LENGTH), (len(lengths), 1))
    for block, length in enumerate(lengths):
        permutations[block, :length] = rng.permutation(length)
        shuffled[block, :length] = padded[block, permutations[block, :length]]
    return shuffled, permutations


def permute_adjacency(adjacency, permutations, device):
    indices = torch.as_tensor(permutations, device=device, dtype=torch.long)
    result = []
    for block in range(adjacency.shape[0]):
        result.append(
            adjacency[block].index_select(0, indices[block]).index_select(1, indices[block])
        )
    return torch.stack(result)


def restore_shuffled_scores(block_scores, permutations, lengths, count):
    restored = np.empty(count, dtype=np.float32)
    for block, length in enumerate(lengths):
        offset = block * VISUAL_LENGTH
        for new_index in range(length):
            old_index = int(permutations[block, new_index])
            restored[offset + old_index] = block_scores[offset + new_index]
    return restored


def interval_mean(values, interval):
    start, end = interval
    return float(np.mean(values[start : end + 1]))


def collect_gt_by_video(rows, feature_root, gt_frames):
    cursor = 0
    gt_by_video = {}
    for row in rows:
        features = np.load(Path(feature_root) / row["path"], mmap_mode="r")
        frame_count = len(features) * 16
        video_gt = np.asarray(gt_frames[cursor : cursor + frame_count])
        if len(video_gt) != frame_count:
            raise ValueError(f"Ground truth is shorter than features for {row['video_id']}.")
        gt_by_video[row["video_id"]] = video_gt.reshape(-1, 16).max(axis=1)
        cursor += frame_count
    if cursor != len(gt_frames):
        raise ValueError(f"Consumed {cursor} GT labels, expected {len(gt_frames)}.")
    return gt_by_video


def analyze_video(model, features, groups, device, repeats):
    count = len(features)
    padded_np, lengths_np = pad_video(features)
    padded = torch.from_numpy(padded_np).to(device)
    lengths = torch.as_tensor(lengths_np, device=device, dtype=torch.long)
    captured = capture_model_stages(model, padded, lengths)
    original_sim = captured["similarity_adj"]
    original_dist = captured["distance_adj"]
    eye = identity_adjacency(len(lengths_np), device)

    outputs = {
        "original_full": captured["encoded"],
        "original_similarity_adj_only": graph_forward(model, captured["temporal"], original_sim, eye),
        "original_distance_adj_only": graph_forward(model, captured["temporal"], eye, original_dist),
        "original_self_only_adjacency": graph_forward(model, captured["temporal"], eye, eye),
    }
    replay_error = float((outputs["original_full"] - graph_forward(
        model, captured["temporal"], original_sim, original_dist
    )).abs().max())
    if replay_error > 1e-4:
        raise RuntimeError(f"Graph replay does not match original encode_video: {replay_error}")
    scores = {name: classifier_scores(model, value, count) for name, value in outputs.items()}

    shuffled_scores = {
        "shuffle_full_recomputed_graph": [],
        "shuffle_original_graph_links": [],
        "shuffle_self_only_adjacency": [],
    }
    for repeat in range(repeats):
        shuffled_np, permutations = shuffle_padded(
            padded_np, lengths_np, seed=7103 + repeat
        )
        shuffled_visual = torch.from_numpy(shuffled_np).to(device)
        shuffled_capture = capture_model_stages(model, shuffled_visual, lengths)
        shuffled_sim = shuffled_capture["similarity_adj"]
        shuffled_dist = shuffled_capture["distance_adj"]

        # Keep each snippet connected to the same original graph neighbors after permutation.
        fixed_sim = permute_adjacency(original_sim, permutations, device)
        fixed_dist = permute_adjacency(original_dist, permutations, device)
        conditions = {
            "shuffle_full_recomputed_graph": shuffled_capture["encoded"],
            "shuffle_original_graph_links": graph_forward(
                model, shuffled_capture["temporal"], fixed_sim, fixed_dist
            ),
            "shuffle_self_only_adjacency": graph_forward(
                model, shuffled_capture["temporal"], eye, eye
            ),
        }
        for name, encoded in conditions.items():
            new_order_scores = classifier_scores(model, encoded, count)
            shuffled_scores[name].append(
                restore_shuffled_scores(new_order_scores, permutations, lengths_np, count)
            )

    for name, runs in shuffled_scores.items():
        scores[name] = np.mean(runs, axis=0)

    rows = []
    for group_name, interval in groups.items():
        rows.append({
            "group": group_name,
            **{condition: interval_mean(values, interval) for condition, values in scores.items()},
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.shuffle_repeats < 1:
        raise ValueError("--shuffle-repeats must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VadCLIP graph adjacency requires CUDA in this implementation.")
    rows = read_test_rows(args.test_list)
    rows_by_video = {row["video_id"]: row for row in rows}
    gt_by_video = collect_gt_by_video(rows, args.feature_root, np.load(args.gt_path, mmap_mode="r"))
    model = CLIPVAD(
        CLASS_COUNT, EMBED_DIM, VISUAL_LENGTH, 512, 1, 2, 8, PROMPT_PREFIX, PROMPT_POSTFIX, device
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    model.eval()

    detail_rows = []
    gap_rows = []
    conditions = (
        "original_full", "original_similarity_adj_only", "original_distance_adj_only",
        "original_self_only_adjacency", "shuffle_full_recomputed_graph",
        "shuffle_original_graph_links", "shuffle_self_only_adjacency",
    )
    for video_id, groups in SEGMENTS.items():
        if video_id not in rows_by_video:
            raise KeyError(f"Selected video missing from test list: {video_id}")
        row = rows_by_video[video_id]
        features = np.asarray(
            np.load(Path(args.feature_root) / row["path"], mmap_mode="r"), dtype=np.float32
        )
        labels = gt_by_video[video_id]
        for group_name, (start, end) in groups.items():
            expected = 0 if group_name == "false_high" else 1
            if end >= len(features) or not np.all(labels[start : end + 1] == expected):
                raise ValueError(f"Ground-truth mismatch for {video_id}/{group_name}.")

        video_rows = analyze_video(model, features, groups, device, args.shuffle_repeats)
        means = {item["group"]: item for item in video_rows}
        for item in video_rows:
            detail_rows.append({
                "video_id": video_id,
                "class_name": row["label"],
                **item,
            })
        gap_rows.append({
            "video_id": video_id,
            "class_name": row["label"],
            **{
                condition: means["true_high"][condition] - means["false_high"][condition]
                for condition in conditions
            },
        })
        print(f"Analyzed {video_id} ({row['label']})", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "temporal_graph_factorial_scores.csv"
    gap_path = output_dir / "temporal_graph_factorial_gaps.csv"
    for path, data in ((detail_path, detail_rows), (gap_path, gap_rows)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    summary = {
        "model_path": str(args.model_path),
        "videos": len(gap_rows),
        "shuffle_repeats": args.shuffle_repeats,
        "metric": "macro mean classifier score gap: true_high minus false_high, per selected video",
        "condition_notes": {
            "original_similarity_adj_only": "Original temporal output; actual similarity adjacency; self-only distance adjacency. Both graph branches and their learned transforms remain present.",
            "original_distance_adj_only": "Original temporal output; self-only similarity adjacency; actual distance adjacency. Both graph branches and their learned transforms remain present.",
            "original_self_only_adjacency": "Original temporal output; both graph adjacencies are identity. Graph-layer convolution residuals remain active.",
            "shuffle_full_recomputed_graph": "Shuffled input; similarity adjacency recomputed; distance adjacency remains positional as in normal forward.",
            "shuffle_original_graph_links": "Shuffled input; both original adjacency matrices are permuted to preserve links between the same snippet identities.",
            "shuffle_self_only_adjacency": "Shuffled input; both graph adjacencies are identity. Graph-layer convolution residuals remain active.",
        },
    }
    for condition in conditions:
        values = np.asarray([row[condition] for row in gap_rows], dtype=np.float64)
        summary[condition] = {
            "macro_true_high_minus_false_high": float(values.mean()),
            "videos_positive_gap": int((values > 0).sum()),
        }
    summary["files"] = {"scores": str(detail_path), "gaps": str(gap_path)}
    summary_path = output_dir / "temporal_graph_factorial_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMacro classifier gap (true-high minus false-high):")
    for condition in conditions:
        stats = summary[condition]
        print(
            f"{condition:38s} {stats['macro_true_high_minus_false_high']:+.5f} "
            f"({stats['videos_positive_gap']}/{len(gap_rows)} videos positive)"
        )
    print("Saved:", detail_path)
    print("Saved:", gap_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
