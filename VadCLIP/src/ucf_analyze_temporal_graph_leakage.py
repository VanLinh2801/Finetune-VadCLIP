"""Inspect whether VadCLIP context moves normal snippets toward anomaly snippets.

Ground truth is used only to define analysis groups and graph interventions.
No optimizer, gradient, or training label is involved.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import CLIPVAD
from utils.tools import process_split


FRAMES_PER_SNIPPET = 16
STAGES = ("raw_clip", "temporal", "final_visual")
GROUPS = ("true_high", "false_high", "missed_anomaly", "true_low")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def similarity_rows(features, groups, video_id, class_name):
    result = []
    device = features["raw_clip"].device
    false_high = torch.as_tensor(groups == "false_high", device=device)
    true_high = torch.as_tensor(groups == "true_high", device=device)
    true_low = torch.as_tensor(groups == "true_low", device=device)
    if not (false_high.any() and true_high.any() and true_low.any()):
        return result
    for stage in STAGES:
        x = F.normalize(features[stage].float(), dim=-1)
        centroids = {
            name: F.normalize(x[mask].mean(dim=0), dim=0)
            for name, mask in (("false_high", false_high), ("true_high", true_high), ("true_low", true_low))
        }
        sim_high = float((centroids["false_high"] * centroids["true_high"]).sum())
        sim_low = float((centroids["false_high"] * centroids["true_low"]).sum())
        result.append({
            "video_id": video_id,
            "class_name": class_name,
            "stage": stage,
            "false_high_to_true_high": sim_high,
            "false_high_to_true_low": sim_low,
            "delta": sim_high - sim_low,
            "false_high_count": int(false_high.sum()),
            "true_high_count": int(true_high.sum()),
            "true_low_count": int(true_low.sum()),
        })
    return result


def graph_forward(model, temporal, similarity_adj, distance_adj):
    sim_first = model.gelu(model.gc1(temporal, similarity_adj))
    dist_first = model.gelu(model.gc3(temporal, distance_adj))
    sim_second = model.gelu(model.gc2(sim_first, similarity_adj))
    dist_second = model.gelu(model.gc4(dist_first, distance_adj))
    return model.linear(torch.cat((sim_second, dist_second), dim=-1))


def remove_sources(adjacency, gt_blocks, lengths, source_type, seed):
    """Remove equal numbers of anomaly or other-normal sources per normal query."""
    modified = adjacency.clone()
    rng = np.random.default_rng(seed)
    removed = 0
    for block, length in enumerate(lengths):
        gt = gt_blocks[block, :length].bool()
        anomaly_sources = torch.where(gt)[0].cpu().numpy()
        normal_sources = torch.where(~gt)[0].cpu().numpy()
        source_count = min(len(anomaly_sources), len(normal_sources) - 1)
        if source_count <= 0:
            continue
        # The first and second GCN layers both use this same adjacency matrix.
        if source_type == "anomaly" and source_count == len(anomaly_sources):
            chosen = np.broadcast_to(anomaly_sources, (len(normal_sources), source_count))
        else:
            chosen = np.stack([
                rng.choice(
                    anomaly_sources if source_type == "anomaly" else normal_sources[normal_sources != target],
                    size=source_count, replace=False,
                )
                for target in normal_sources
            ])
        target_rows = torch.as_tensor(normal_sources, device=adjacency.device)
        source_cols = torch.as_tensor(chosen.copy(), device=adjacency.device)
        old_sum = adjacency[block, target_rows, :length].sum(dim=-1, keepdim=True)
        modified[block, target_rows[:, None], source_cols] = 0
        new_sum = modified[block, target_rows, :length].sum(dim=-1, keepdim=True)
        scale = torch.where(new_sum > 1e-12, old_sum / new_sum.clamp_min(1e-12), 1.0)
        modified[block, target_rows, :length] *= scale
        removed += len(normal_sources) * source_count
    return modified, removed


def capture_encode_video(model, visual, lengths):
    captured = {"attention": []}
    hooks = []

    def capture_temporal(_module, _inputs, output):
        captured["temporal"] = output[0].permute(1, 0, 2).detach()

    def capture_similarity(_module, inputs):
        captured["similarity_adj"] = inputs[1].detach()

    def capture_distance(_module, inputs):
        captured["distance_adj"] = inputs[1].detach()

    def capture_attention(module, inputs, kwargs):
        # Calling forward directly avoids invoking hooks recursively and leaves
        # the original need_weights=False forward completely untouched.
        diagnostic_kwargs = dict(kwargs)
        diagnostic_kwargs["need_weights"] = True
        diagnostic_kwargs["average_attn_weights"] = False
        _, weights = module.forward(*inputs, **diagnostic_kwargs)
        captured["attention"].append(weights.detach())

    hooks.append(model.temporal.register_forward_hook(capture_temporal))
    hooks.append(model.gc1.register_forward_pre_hook(capture_similarity))
    hooks.append(model.gc3.register_forward_pre_hook(capture_distance))
    for block in model.temporal.resblocks:
        hooks.append(block.attn.register_forward_pre_hook(capture_attention, with_kwargs=True))
    try:
        with torch.no_grad():
            final_visual = model.encode_video(visual, None, lengths)
            logits1 = model.classifier(final_visual + model.mlp2(final_visual))
    finally:
        for hook in hooks:
            hook.remove()
    captured["final_visual"] = final_visual.detach()
    captured["logits1"] = logits1.detach()
    if len(captured["attention"]) != len(model.temporal.resblocks):
        raise RuntimeError("Failed to record all temporal attention layers.")
    return captured


def mass_from_sources(weights, source_mask):
    weights = weights.float()
    source_mask = source_mask.to(weights.device)
    row_sum = weights.sum(dim=-1).clamp_min(1e-12)
    return ((weights * source_mask.unsqueeze(0)).sum(dim=-1) / row_sum).cpu().numpy()


def block_exposure(captured, gt_blocks, lengths, attention_window):
    attention = torch.stack(captured["attention"]).mean(dim=(0, 2))
    sim_adj = captured["similarity_adj"]
    dist_adj = captured["distance_adj"]
    rows = []
    for block, length in enumerate(lengths):
        gt = gt_blocks[block, :length].bool()
        window_fraction = np.zeros(length, dtype=np.float32)
        for start in range(0, length, attention_window):
            stop = min(start + attention_window, length)
            window_fraction[start:stop] = float(gt[start:stop].float().mean())
        attention_mass = mass_from_sources(attention[block, :length, :length], gt)
        sim_mass = mass_from_sources(sim_adj[block, :length, :length], gt)
        dist_mass = mass_from_sources(dist_adj[block, :length, :length], gt)
        block_fraction = float(gt.float().mean())
        for index in range(length):
            rows.append({
                "temporal_anomaly_mass": float(attention_mass[index]),
                "temporal_anomaly_available": float(window_fraction[index]),
                "temporal_excess": float(attention_mass[index] - window_fraction[index]),
                "similarity_graph_anomaly_mass": float(sim_mass[index]),
                "similarity_graph_anomaly_available": block_fraction,
                "similarity_graph_excess": float(sim_mass[index] - block_fraction),
                "distance_graph_anomaly_mass": float(dist_mass[index]),
                "distance_graph_anomaly_available": block_fraction,
                "distance_graph_excess": float(dist_mass[index] - block_fraction),
            })
    return rows


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"n": 0, "mean": None, "median": None, "ci95_low": None, "ci95_high": None}
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "n": int(len(arr)), "mean": mean, "median": float(np.median(arr)),
        "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
    }


def per_video_means(rows, metrics):
    grouped = {}
    for row in rows:
        key = (row["video_id"], row["class_name"], row["group"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (video_id, class_name, group), members in grouped.items():
        result.append({
            "video_id": video_id, "class_name": class_name, "group": group,
            "snippet_count": len(members),
            **{metric: float(np.nanmean([member[metric] for member in members]))
               for metric in metrics},
        })
    return result


def paired_group_differences(video_rows, metrics):
    by_video = {}
    for row in video_rows:
        by_video.setdefault(row["video_id"], {})[row["group"]] = row
    result = {}
    for metric in metrics:
        differences = [groups["false_high"][metric] - groups["true_low"][metric]
                       for groups in by_video.values()
                       if "false_high" in groups and "true_low" in groups]
        result[metric] = summarize(differences)
    return result


def exposure_affinity_association(stage_rows, video_exposure):
    stage_by_video = {}
    for row in stage_rows:
        stage_by_video.setdefault(row["video_id"], {})[row["stage"]] = row["delta"]
    exposure_by_video = {row["video_id"]: row for row in video_exposure
                         if row["group"] == "false_high"}
    rows = []
    for video_id, stages in stage_by_video.items():
        if video_id not in exposure_by_video or not all(stage in stages for stage in STAGES):
            continue
        exposure = exposure_by_video[video_id]
        rows.append({
            "video_id": video_id,
            "delta_raw_to_final": stages["final_visual"] - stages["raw_clip"],
            "delta_temporal_to_final": stages["final_visual"] - stages["temporal"],
            "temporal_excess": exposure["temporal_excess"],
            "similarity_graph_excess": exposure["similarity_graph_excess"],
            "distance_graph_excess": exposure["distance_graph_excess"],
        })
    correlations = {}
    for metric in ("temporal_excess", "similarity_graph_excess", "distance_graph_excess"):
        x = np.asarray([row[metric] for row in rows])
        for change in ("delta_raw_to_final", "delta_temporal_to_final"):
            y = np.asarray([row[change] for row in rows])
            key = f"{metric}_vs_{change}"
            correlations[key] = float(np.corrcoef(x, y)[0, 1]) if (
                len(rows) > 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12
            ) else None
    return rows, correlations


def create_plots(output_dir, stage_rows, exposure_rows, intervention_rows, association_rows):
    stages = list(STAGES)
    paired = {}
    for row in stage_rows:
        paired.setdefault(row["video_id"], {})[row["stage"]] = row["delta"]
    complete = [data for data in paired.values() if all(stage in data for stage in stages)]
    if complete:
        figure, axis = plt.subplots(figsize=(7, 4))
        x = np.arange(len(stages))
        for data in complete:
            axis.plot(x, [data[s] for s in stages], color="slategray", alpha=0.13)
        axis.plot(x, [np.mean([data[s] for data in complete]) for s in stages],
                  marker="o", color="crimson", linewidth=2.5, label="Video mean")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["CLIP", "Temporal", "Temporal + graph"])
        axis.set_ylabel("Cosine to true-high minus cosine to true-low")
        axis.set_title("False-high affinity across visual stages")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "01_feature_affinity_by_stage.png", dpi=180)
        plt.close(figure)

    if exposure_rows:
        keys = ("temporal_excess", "similarity_graph_excess", "distance_graph_excess")
        figure, axis = plt.subplots(figsize=(8, 4))
        groups = ("false_high", "true_low")
        values = [[np.nanmean([r[key] for r in exposure_rows if r["group"] == group])
                   for key in keys] for group in groups]
        x = np.arange(len(keys))
        axis.bar(x - 0.18, values[0], width=0.36, label="False-high")
        axis.bar(x + 0.18, values[1], width=0.36, label="True-low")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["Temporal", "Similarity graph", "Distance graph"])
        axis.set_ylabel("Anomaly mass minus available anomaly fraction")
        axis.set_title("Anomaly source exposure for normal snippets")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "02_anomaly_source_exposure.png", dpi=180)
        plt.close(figure)

    if intervention_rows:
        figure, axis = plt.subplots(figsize=(7, 4))
        groups = ("false_high", "true_low")
        for index, metric in enumerate(("remove_anomaly_score_change", "remove_normal_score_change")):
            values = [np.nanmean([r[metric] for r in intervention_rows if r["group"] == group])
                      for group in groups]
            axis.bar(np.arange(2) + (index - 0.5) * 0.32, values, width=0.32,
                     label="Remove anomaly" if index == 0 else "Remove matched normal")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(2), ["False-high", "True-low"])
        axis.set_ylabel("Counterfactual classifier score minus original")
        axis.set_title("Graph source intervention")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "03_graph_intervention.png", dpi=180)
        plt.close(figure)

    if association_rows:
        x = np.asarray([row["similarity_graph_excess"] for row in association_rows])
        y = np.asarray([row["delta_temporal_to_final"] for row in association_rows])
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(x, y, alpha=0.65, s=28)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.set_xlabel("False-high anomaly graph mass above availability")
        axis.set_ylabel("Affinity delta change: temporal to graph")
        axis.set_title("Does graph exposure track feature drift?")
        figure.tight_layout()
        figure.savefig(output_dir / "04_graph_exposure_vs_drift.png", dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-ratio", type=float, default=0.15)
    parser.add_argument("--max-abnormal-videos", type=int, default=0,
                        help="0 means all abnormal test videos")
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--attn-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VadCLIP DistanceAdj requires CUDA. Choose a GPU runtime.")
    if not 0 < args.top_k_ratio <= 1:
        raise ValueError("--top-k-ratio must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = CLIPVAD(14, 512, args.visual_length, 512, 1, 2,
                    args.attn_window, 10, 10, device)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    with open(args.test_list, newline="", encoding="utf-8-sig") as handle:
        test_rows = list(csv.DictReader(handle))
    gt = np.load(args.gt_path)
    stage_rows = []
    exposure_rows = []
    intervention_rows = []
    cursor = 0
    analyzed = 0
    max_replay_error = 0.0
    for row in test_rows:
        feature_path = Path(args.feature_root) / row["path"]
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        raw_feature = np.load(feature_path, mmap_mode="r")
        snippet_count = int(raw_feature.shape[0])
        video_gt = gt[cursor:cursor + snippet_count * FRAMES_PER_SNIPPET]
        cursor += snippet_count * FRAMES_PER_SNIPPET
        if len(video_gt) != snippet_count * FRAMES_PER_SNIPPET:
            raise ValueError(f"Ground truth is shorter than test features at {row['video_id']}.")
        if row["label"] == "Normal":
            continue
        if args.max_abnormal_videos and analyzed >= args.max_abnormal_videos:
            break
        analyzed += 1
        processed, _ = process_split(np.asarray(raw_feature), args.visual_length)
        if processed.ndim == 2:
            processed = processed[None, ...]
        block_count = math.ceil(snippet_count / args.visual_length)
        processed = np.array(processed[:block_count], dtype=np.float32, copy=True)
        lengths = [min(args.visual_length, snippet_count - b * args.visual_length)
                   for b in range(block_count)]
        visual = torch.from_numpy(processed).to(device)
        gt_any = (video_gt.reshape(snippet_count, FRAMES_PER_SNIPPET).mean(axis=1) > 0)
        gt_blocks = torch.zeros((block_count, args.visual_length), dtype=torch.bool, device=device)
        for block, length in enumerate(lengths):
            start = block * args.visual_length
            gt_blocks[block, :length] = torch.from_numpy(gt_any[start:start + length]).to(device)

        captured = capture_encode_video(model, visual, lengths)
        final_visual = captured["final_visual"]
        replay = graph_forward(model, captured["temporal"], captured["similarity_adj"],
                               captured["distance_adj"])
        replay_error = float((replay - final_visual).abs().max())
        max_replay_error = max(max_replay_error, replay_error)
        if replay_error > 1e-4:
            raise RuntimeError(f"Graph replay changed the original model output at {row['video_id']}: {replay_error}")

        scores = torch.sigmoid(captured["logits1"].reshape(-1)[:snippet_count]).cpu().numpy()
        selected = np.zeros(snippet_count, dtype=bool)
        selected[np.argsort(-scores, kind="stable")[:max(1, math.ceil(snippet_count * args.top_k_ratio))]] = True
        groups = np.full(snippet_count, "true_low", dtype=object)
        groups[selected & gt_any] = "true_high"
        groups[selected & ~gt_any] = "false_high"
        groups[~selected & gt_any] = "missed_anomaly"
        features = {"raw_clip": visual.reshape(-1, visual.shape[-1])[:snippet_count],
                    "temporal": captured["temporal"].reshape(-1, visual.shape[-1])[:snippet_count],
                    "final_visual": final_visual.reshape(-1, visual.shape[-1])[:snippet_count]}
        stage_rows.extend(similarity_rows(features, groups, row["video_id"], row["label"]))
        exposures = block_exposure(captured, gt_blocks, lengths, args.attn_window)

        sim_no_anomaly, sim_removed = remove_sources(
            captured["similarity_adj"], gt_blocks, lengths, "anomaly", args.seed)
        dist_no_anomaly, dist_removed = remove_sources(
            captured["distance_adj"], gt_blocks, lengths, "anomaly", args.seed)
        sim_no_normal, _ = remove_sources(
            captured["similarity_adj"], gt_blocks, lengths, "normal", args.seed + analyzed)
        dist_no_normal, _ = remove_sources(
            captured["distance_adj"], gt_blocks, lengths, "normal", args.seed + analyzed)
        with torch.no_grad():
            visual_no_anomaly = graph_forward(model, captured["temporal"], sim_no_anomaly, dist_no_anomaly)
            visual_no_normal = graph_forward(model, captured["temporal"], sim_no_normal, dist_no_normal)
            score_no_anomaly = torch.sigmoid(model.classifier(
                visual_no_anomaly + model.mlp2(visual_no_anomaly))).reshape(-1)[:snippet_count].cpu().numpy()
            score_no_normal = torch.sigmoid(model.classifier(
                visual_no_normal + model.mlp2(visual_no_normal))).reshape(-1)[:snippet_count].cpu().numpy()
        if (groups == "true_high").any():
            true_high_mask = torch.as_tensor(groups == "true_high", device=device)
            high_centroid = F.normalize(F.normalize(features["final_visual"].float(), dim=-1)[
                true_high_mask].mean(dim=0), dim=0)
            original_affinity = (F.normalize(features["final_visual"].float(), dim=-1) @ high_centroid).cpu().numpy()
            new_affinity = (F.normalize(visual_no_anomaly.reshape(-1, visual.shape[-1])[:snippet_count].float(),
                                        dim=-1) @ high_centroid).cpu().numpy()
        else:
            original_affinity = np.full(snippet_count, np.nan)
            new_affinity = np.full(snippet_count, np.nan)
        for index, exposure in enumerate(exposures):
            if groups[index] not in ("false_high", "true_low"):
                continue
            base = {"video_id": row["video_id"], "class_name": row["label"],
                    "snippet_index": index, "group": groups[index],
                    "classifier_score": float(scores[index])}
            exposure_rows.append({**base, **exposure})
            intervention_rows.append({
                **base,
                "remove_anomaly_score_change": float(score_no_anomaly[index] - scores[index]),
                "remove_normal_score_change": float(score_no_normal[index] - scores[index]),
                "remove_anomaly_true_high_affinity_change": float(new_affinity[index] - original_affinity[index]),
                "similarity_graph_removed_edges": sim_removed,
                "distance_graph_removed_edges": dist_removed,
            })
        if analyzed % 10 == 0:
            print(f"Analyzed {analyzed} abnormal videos; latest: {row['video_id']}", flush=True)

    if args.max_abnormal_videos == 0 and cursor != len(gt):
        raise ValueError(f"GT alignment mismatch: consumed {cursor} frames, expected {len(gt)}.")
    write_csv(output_dir / "per_video_stage_similarity.csv", stage_rows)
    write_csv(output_dir / "normal_snippet_exposure.csv", exposure_rows)
    write_csv(output_dir / "graph_intervention.csv", intervention_rows)

    exposure_metrics = (
        "temporal_anomaly_mass", "temporal_excess", "similarity_graph_anomaly_mass",
        "similarity_graph_excess", "distance_graph_anomaly_mass", "distance_graph_excess",
    )
    intervention_metrics = (
        "remove_anomaly_score_change", "remove_normal_score_change",
        "remove_anomaly_true_high_affinity_change",
    )
    video_exposure = per_video_means(exposure_rows, exposure_metrics)
    video_intervention = per_video_means(intervention_rows, intervention_metrics)
    association_rows, correlations = exposure_affinity_association(stage_rows, video_exposure)
    write_csv(output_dir / "per_video_exposure.csv", video_exposure)
    write_csv(output_dir / "per_video_intervention.csv", video_intervention)
    write_csv(output_dir / "graph_exposure_vs_drift.csv", association_rows)

    paired = {}
    for row in stage_rows:
        paired.setdefault(row["video_id"], {})[row["stage"]] = row["delta"]
    stage_change = [values["final_visual"] - values["raw_clip"]
                    for values in paired.values() if all(stage in values for stage in STAGES)]
    summary = {
        "model_path": str(args.model_path), "top_k_ratio": args.top_k_ratio,
        "abnormal_videos_analyzed": analyzed, "videos_with_all_three_groups": len(stage_change),
        "max_graph_replay_error": max_replay_error,
        "stage_delta": {stage: summarize([r["delta"] for r in stage_rows if r["stage"] == stage])
                        for stage in STAGES},
        "raw_to_final_delta_change": summarize(stage_change),
        "raw_to_final_positive_change_fraction": float(np.mean(np.asarray(stage_change) > 0)) if stage_change else None,
        "exposure": {group: {key: summarize([r[key] for r in video_exposure if r["group"] == group])
                              for key in exposure_metrics}
                     for group in ("false_high", "true_low")},
        "paired_exposure_false_high_minus_true_low": paired_group_differences(
            video_exposure, exposure_metrics),
        "intervention": {group: {key: summarize([r[key] for r in video_intervention if r["group"] == group])
                                  for key in intervention_metrics}
                         for group in ("false_high", "true_low")},
        "paired_intervention_false_high_minus_true_low": paired_group_differences(
            video_intervention, intervention_metrics),
        "exposure_affinity_correlations": correlations,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_plots(output_dir, stage_rows, video_exposure, video_intervention, association_rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved diagnostics to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
