"""Unified feature-stage and classifier-impact diagnostics for baseline VadCLIP."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CLIPVAD
from ucf_analyze_targeted_temporal_context import apply_attention_intervention
from ucf_analyze_temporal_graph_factorial import graph_forward, pad_video
from ucf_analyze_temporal_order import (
    CLASS_COUNT,
    EMBED_DIM,
    FRAMES_PER_SNIPPET,
    PROMPT_POSTFIX,
    PROMPT_PREFIX,
    VISUAL_LENGTH,
)


def collect_video_ground_truth(rows, feature_root, frame_gt):
    """Align frame labels and snippet labels with feature rows in test-list order."""
    cursor = 0
    result = {}
    for row in rows:
        features = np.load(Path(feature_root) / row["path"], mmap_mode="r")
        snippet_count = len(features)
        frame_count = snippet_count * FRAMES_PER_SNIPPET
        labels = np.asarray(frame_gt[cursor : cursor + frame_count], dtype=np.int8)
        if len(labels) != frame_count:
            raise ValueError(
                f"Ground truth length mismatch for {row['video_id']}: "
                f"expected {frame_count}, got {len(labels)}"
            )
        result[row["video_id"]] = {
            "frames": labels,
            "snippets": labels.reshape(snippet_count, FRAMES_PER_SNIPPET).max(axis=1),
        }
        cursor += frame_count
    if cursor != len(frame_gt):
        raise ValueError(f"Consumed {cursor} frame labels, but GT contains {len(frame_gt)}")
    return result


def get_positioned_input(model, visual):
    with torch.no_grad():
        batch, seq_len, _ = visual.shape
        positions = torch.arange(seq_len, device=visual.device).unsqueeze(0).expand(batch, -1)
        position_embeddings = model.frame_position_embeddings(positions).permute(1, 0, 2)
        return visual.float().permute(1, 0, 2).add(position_embeddings).permute(1, 0, 2)


def classifier_outputs(model, encoded):
    with torch.no_grad():
        classifier_input = encoded + model.mlp2(encoded)
        logits = model.classifier(classifier_input).squeeze(-1)
        return classifier_input, logits, torch.sigmoid(logits)


def build_similarity_adjacency(model, temporal, lengths):
    return model.adj4(temporal, lengths)


def graph_without_cross_snippet_messages(model, temporal):
    identity = torch.eye(
        temporal.shape[1], device=temporal.device, dtype=temporal.dtype
    ).unsqueeze(0).expand(temporal.shape[0], -1, -1)
    return graph_forward(model, temporal, identity, identity)


def tensor_change_metrics(left, right):
    left_f = left.float()
    right_f = right.float()
    cosine = F.cosine_similarity(left_f, right_f, dim=-1)
    relative_l2 = (left_f - right_f).norm(dim=-1) / left_f.norm(dim=-1).clamp_min(1e-8)
    return cosine, relative_l2


def select_topk_groups(scores, snippet_labels, ratio):
    count = len(snippet_labels)
    k = max(1, int(np.ceil(count * ratio)))
    chosen = np.argpartition(scores[:count], -k)[-k:]
    return {
        int(index): "true_high" if int(snippet_labels[index]) else "false_high"
        for index in chosen
    }


def summarize_values(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "positive_fraction": float((array > 0).mean()),
    }


def metric_pair(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size != 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(100 * roc_auc_score(labels, scores)),
        "ap": float(100 * average_precision_score(labels, scores)),
    }


def analyze_one_video(model, features, gt, device, top_k_ratio):
    count = len(features)
    padded_np, lengths_np = pad_video(features)
    visual = torch.from_numpy(padded_np).to(device)
    lengths = torch.as_tensor(lengths_np, device=device, dtype=torch.long)

    positioned = get_positioned_input(model, visual)
    # Capture the original temporal output and graph adjacency through the public model path.
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
            full_encoded = model.encode_video(visual, None, lengths)
    finally:
        for hook in hooks:
            hook.remove()

    temporal_full = captured["temporal"]
    sim_full = captured["similarity_adj"]
    dist_full = captured["distance_adj"]
    classifier_input_full, logits_full, scores_full = classifier_outputs(model, full_encoded)

    # Restrict every query in every temporal layer to itself; this removes cross-snippet
    # temporal exchange while leaving the transformer's per-token layers and residuals active.
    temporal_self = apply_attention_intervention(model, visual, "all_self_only")
    temporal_bypass = positioned
    sim_self = build_similarity_adjacency(model, temporal_self, lengths)
    sim_bypass = build_similarity_adjacency(model, temporal_bypass, lengths)

    with torch.no_grad():
        encoded_self_fixed = graph_forward(model, temporal_self, sim_full, dist_full)
        encoded_self_rebuilt = graph_forward(model, temporal_self, sim_self, dist_full)
        encoded_graph_identity = graph_without_cross_snippet_messages(model, temporal_full)
        encoded_both_identity = graph_without_cross_snippet_messages(model, temporal_self)
        encoded_temporal_bypass = graph_forward(model, temporal_bypass, sim_full, dist_full)
        encoded_temporal_bypass_rebuilt = graph_forward(model, temporal_bypass, sim_bypass, dist_full)

    conditions = {
        "full": (classifier_input_full, logits_full, scores_full),
        "temporal_self_only_fixed_graph": classifier_outputs(model, encoded_self_fixed),
        "temporal_self_only_rebuilt_graph": classifier_outputs(model, encoded_self_rebuilt),
        "graph_identity": classifier_outputs(model, encoded_graph_identity),
        "temporal_and_graph_identity": classifier_outputs(model, encoded_both_identity),
        "temporal_bypass_fixed_graph": classifier_outputs(model, encoded_temporal_bypass),
        "temporal_bypass_rebuilt_graph": classifier_outputs(model, encoded_temporal_bypass_rebuilt),
    }

    # The full score defines a fixed top-15% sample for every intervention.
    full_snippet_scores = scores_full.detach().cpu().numpy().reshape(-1)[:count]
    target_groups = select_topk_groups(full_snippet_scores, gt["snippets"], top_k_ratio)

    valid = torch.arange(VISUAL_LENGTH, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    metric_specs = {
        "clip_position_to_temporal": (positioned, temporal_full),
        "full_temporal_to_self_only": (temporal_full, temporal_self),
        "temporal_to_graph_output": (temporal_full, full_encoded),
        "graph_output_to_classifier_input": (full_encoded, classifier_input_full),
    }
    feature_arrays = {}
    for name, (left, right) in metric_specs.items():
        cosine, relative_l2 = tensor_change_metrics(left, right)
        feature_arrays[f"{name}_cosine"] = cosine.detach().cpu().numpy()[valid.cpu().numpy()]
        feature_arrays[f"{name}_relative_l2"] = relative_l2.detach().cpu().numpy()[valid.cpu().numpy()]

    scores_np = {}
    logits_np = {}
    for name, (_encoded, logits, scores) in conditions.items():
        logits_np[name] = logits.detach().cpu().numpy().reshape(-1)[:count]
        scores_np[name] = scores.detach().cpu().numpy().reshape(-1)[:count]

    per_snippet = []
    for index in range(count):
        group = target_groups.get(index, "")
        row = {
            "snippet_index": index,
            "gt_label": int(gt["snippets"][index]),
            "analysis_group": group or ("other_anomaly" if gt["snippets"][index] else "other_normal"),
        }
        for name in conditions:
            row[f"logit_{name}"] = float(logits_np[name][index])
            row[f"score_{name}"] = float(scores_np[name][index])
        for name in ("temporal_self_only_fixed_graph", "temporal_self_only_rebuilt_graph",
                     "graph_identity", "temporal_and_graph_identity",
                     "temporal_bypass_fixed_graph", "temporal_bypass_rebuilt_graph"):
            row[f"delta_logit_full_minus_{name}"] = float(logits_np["full"][index] - logits_np[name][index])
            row[f"delta_score_full_minus_{name}"] = float(scores_np["full"][index] - scores_np[name][index])
        for name, values in feature_arrays.items():
            row[name] = float(values[index])
        per_snippet.append(row)

    return per_snippet, scores_np, logits_np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.top_k_ratio <= 1:
        raise ValueError("--top-k-ratio must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VadCLIP graph adjacency requires CUDA in this implementation.")

    with Path(args.test_list).open(newline="", encoding="utf-8-sig") as handle:
        test_rows = list(csv.DictReader(handle))
    frame_gt = np.load(args.gt_path, mmap_mode="r")
    gt_by_video = collect_video_ground_truth(test_rows, args.feature_root, frame_gt)

    model = CLIPVAD(
        CLASS_COUNT, EMBED_DIM, VISUAL_LENGTH, 512, 1, 2, 8,
        PROMPT_PREFIX, PROMPT_POSTFIX, device,
    ).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    conditions = [
        "full", "temporal_self_only_fixed_graph", "temporal_self_only_rebuilt_graph",
        "graph_identity", "temporal_and_graph_identity", "temporal_bypass_fixed_graph",
        "temporal_bypass_rebuilt_graph",
    ]
    frame_scores = {name: [] for name in conditions}
    all_frame_labels = []
    snippet_rows = []
    video_group_rows = []
    feature_group_values = defaultdict(lambda: defaultdict(list))
    metric_columns = [
        "clip_position_to_temporal_cosine", "clip_position_to_temporal_relative_l2",
        "full_temporal_to_self_only_cosine", "full_temporal_to_self_only_relative_l2",
        "temporal_to_graph_output_cosine", "temporal_to_graph_output_relative_l2",
        "graph_output_to_classifier_input_cosine", "graph_output_to_classifier_input_relative_l2",
    ]

    for video_number, row in enumerate(test_rows, start=1):
        video_id = row["video_id"]
        features = np.asarray(
            np.load(Path(args.feature_root) / row["path"], mmap_mode="r"), dtype=np.float32
        )
        details, scores, logits = analyze_one_video(
            model, features, gt_by_video[video_id], device, args.top_k_ratio
        )
        frame_labels = gt_by_video[video_id]["frames"]
        all_frame_labels.append(frame_labels)
        for condition in conditions:
            frame_scores[condition].append(
                np.repeat(scores[condition], FRAMES_PER_SNIPPET).astype(np.float32)
            )

        by_group = defaultdict(list)
        for detail in details:
            detail = {"video_id": video_id, "class_name": row["label"], **detail}
            snippet_rows.append(detail)
            by_group[detail["analysis_group"]].append(detail)
            for metric in metric_columns:
                feature_group_values[detail["analysis_group"]][metric].append(detail[metric])
        for group, members in by_group.items():
            video_group_rows.append({
                "video_id": video_id,
                "class_name": row["label"],
                "analysis_group": group,
                "snippet_count": len(members),
                **{
                    metric: float(np.mean([item[metric] for item in members]))
                    for metric in metric_columns
                },
                **{
                    f"delta_score_full_minus_{condition}": float(np.mean([
                        item[f"delta_score_full_minus_{condition}"] for item in members
                    ]))
                    for condition in conditions if condition != "full"
                },
                **{
                    f"delta_logit_full_minus_{condition}": float(np.mean([
                        item[f"delta_logit_full_minus_{condition}"] for item in members
                    ]))
                    for condition in conditions if condition != "full"
                },
            })
        print(f"Analyzed {video_number}/{len(test_rows)}: {video_id}", flush=True)

    labels_all = np.concatenate(all_frame_labels)
    score_metrics = {}
    for condition in conditions:
        scores_all = np.concatenate(frame_scores[condition])
        if len(scores_all) != len(labels_all):
            raise ValueError(f"Frame score/GT mismatch for condition {condition}.")
        score_metrics[condition] = metric_pair(labels_all, scores_all)

    target_rows = [row for row in snippet_rows if row["analysis_group"] in ("true_high", "false_high")]
    score_delta_columns = [
        f"delta_score_full_minus_{name}" for name in conditions if name != "full"
    ]
    logit_delta_columns = [
        f"delta_logit_full_minus_{name}" for name in conditions if name != "full"
    ]
    target_summary = {}
    feature_summary_by_group = {}
    for group, values_by_metric in feature_group_values.items():
        feature_summary_by_group[group] = {
            metric: summarize_values(values)
            for metric, values in values_by_metric.items()
        }
    for group in ("true_high", "false_high"):
        members = [row for row in target_rows if row["analysis_group"] == group]
        target_summary[group] = {
            "snippet_count": len(members),
            "videos": len({row["video_id"] for row in members}),
            "score_deltas": {
                column.removeprefix("delta_score_full_minus_"): summarize_values(
                    [row[column] for row in members]
                ) for column in score_delta_columns
            },
            "logit_deltas": {
                column.removeprefix("delta_logit_full_minus_"): summarize_values(
                    [row[column] for row in members]
                ) for column in logit_delta_columns
            },
            "feature_changes": {
                column: summarize_values(feature_group_values[group][column])
                for column in metric_columns
            },
        }

    paired = defaultdict(dict)
    for row in video_group_rows:
        if row["analysis_group"] in ("true_high", "false_high"):
            paired[row["video_id"]][row["analysis_group"]] = row
    paired_videos = [groups for groups in paired.values() if set(groups) == {"true_high", "false_high"}]
    paired_effects = {}
    for condition in conditions:
        if condition == "full":
            continue
        column = f"delta_score_full_minus_{condition}"
        differences = [groups["true_high"][column] - groups["false_high"][column] for groups in paired_videos]
        paired_effects[condition] = {
            "paired_video_count": len(differences),
            "mean_true_minus_false": float(np.mean(differences)) if differences else None,
            "videos_true_effect_greater": int(sum(value > 0 for value in differences)),
            "videos_false_effect_greater": int(sum(value < 0 for value in differences)),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("stage_scores_by_snippet.csv", snippet_rows),
        ("stage_features_by_video_group.csv", video_group_rows),
    ):
        if rows:
            with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    summary = {
        "model_path": str(args.model_path),
        "test_videos": len(test_rows),
        "test_frames": int(len(labels_all)),
        "top_k_ratio_per_video": args.top_k_ratio,
        "top_k_selected_snippets": len(target_rows),
        "top_k_group_counts": {
            group: target_summary[group]["snippet_count"] for group in ("true_high", "false_high")
        },
        "conditions": {
            "full": "Unmodified VadCLIP inference.",
            "temporal_self_only_fixed_graph": "Every temporal query sees only itself; original graph adjacency is held fixed.",
            "temporal_self_only_rebuilt_graph": "Every temporal query sees only itself; similarity adjacency is recomputed from that representation.",
            "graph_identity": "Full temporal features; both graph adjacencies are identity, so graph layers do not exchange information across snippets.",
            "temporal_and_graph_identity": "Self-only temporal attention followed by identity graph adjacency.",
            "temporal_bypass_fixed_graph": "Bypass temporal Transformer blocks, retain positional embeddings and original graph adjacency.",
            "temporal_bypass_rebuilt_graph": "Bypass temporal Transformer blocks, retain positional embeddings and recompute similarity adjacency.",
        },
        "metric_protocol": "AUC/AP are frame-level: each snippet classifier score is repeated for its 16 source frames and compared with the original frame GT. Snippet groups are selected once from full-model top-k per video, then labeled using GT for post-hoc analysis only.",
        "limitations": [
            "Identity graph adjacency removes cross-snippet graph message passing but keeps graph projections, residuals, nonlinearities, and fusion.",
            "Temporal bypass changes feature distribution; its score effect is an intervention sensitivity check, not a standalone trained-model performance estimate.",
            "Feature distance and classifier score impact are separate quantities; a feature change alone does not establish task usefulness.",
        ],
        "frame_level_metrics": score_metrics,
        "top_k_group_analysis": target_summary,
        "feature_change_by_group": feature_summary_by_group,
        "paired_video_true_minus_false_effect": paired_effects,
        "outputs": {
            "snippet_scores": str(output_dir / "stage_scores_by_snippet.csv"),
            "video_group_features": str(output_dir / "stage_features_by_video_group.csv"),
        },
    }
    summary_path = output_dir / "stage_contribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFrame-level classifier metrics (AUC/AP %):")
    for condition, metrics in score_metrics.items():
        print(f"{condition:42s} AUC={metrics['auc']} AP={metrics['ap']}")
    print("\nTop-k snippets by GT group:")
    for group in ("true_high", "false_high"):
        stats = target_summary[group]
        print(f"{group}: {stats['snippet_count']} snippets in {stats['videos']} videos")
        for condition, values in stats["score_deltas"].items():
            print(f"  full - {condition}: mean={values['mean']:+.6f}, median={values['median']:+.6f}")
    print("\nFeature changes for the fixed top-k groups:")
    for group in ("true_high", "false_high"):
        stats = feature_summary_by_group[group]
        cosine = stats["clip_position_to_temporal_cosine"]
        shift = stats["clip_position_to_temporal_relative_l2"]
        print(
            f"{group}: position->temporal cosine={cosine['mean']:.4f}; "
            f"relative L2={shift['mean']:.4f}"
        )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
