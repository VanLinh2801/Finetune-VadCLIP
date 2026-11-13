"""Compare trained FC weights and explore kernel selection without retraining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model_multiscale_cnn import MultiscaleCNN
from ucf_cnn_common import UCFCNNFeatureDataset
from ucf_evaluate_multiscale_cnn import metrics_binary, write_csv


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("no-l1-checkpoint", "l1-checkpoint", "feature-root", "test-list", "gt-path", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--expected-videos", type=int, default=290)
    parser.add_argument("--expected-epoch", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=8)
    return parser


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("model_type") != "multiscale_video_cnn":
        raise ValueError(f"Not a multiscale video CNN checkpoint: {path}")
    config = saved["config"]
    model = MultiscaleCNN(config["kernel_sizes"], config["channels_per_branch"], config["visual_length"],
                          activation=config.get("cnn_activation", "gelu")).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), config, saved["epoch"]


def validate_pair(config0, config1, epoch0, epoch1, expected_epoch):
    if epoch0 != epoch1 or (expected_epoch and epoch0 != expected_epoch):
        raise ValueError(f"Checkpoints must have the same expected epoch: {epoch0}, {epoch1}")
    keys = ("kernel_sizes", "channels_per_branch", "visual_length", "seed", "batch_size", "lr",
            "weight_decay", "scheduler_milestones", "scheduler_rate", "grad_clip_norm")
    mismatches = [key for key in keys if config0.get(key) != config1.get(key)]
    if config0.get("cnn_activation", "gelu") != config1.get("cnn_activation", "gelu"):
        mismatches.append("cnn_activation")
    if mismatches:
        raise ValueError(f"Incompatible checkpoint configurations: {mismatches}")
    lambda0 = config0.get("lambda_fc_l1", 0.0)
    lambda1 = config1.get("lambda_fc_l1", 0.0)
    if lambda0 != 0 or not lambda1 > 0:
        raise ValueError(f"Expected no-L1 vs positive-L1, got {lambda0}, {lambda1}")


def build_selections(weight, sizes, include_selections):
    channels = weight.shape[1]
    if any(k < 1 or k > channels for k in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError(f"Top-K sizes must be distinct and in [1,{channels}]")
    selections = {"all": np.arange(channels)}
    if include_selections:
        delta = (weight[1] - weight[0]).detach().cpu().numpy()
        # Stable ties use the channel index; GT and test responses are not used.
        order = np.argsort(-np.abs(delta), kind="stable")
        selections.update({f"top_{k}_abs_delta": np.sort(order[:k]) for k in sizes})
        selections["positive_delta"] = np.flatnonzero(delta > 0)
    return selections


def selection_logits(responses, fc, indices):
    """Apply the original two-class FC to selected channels, retain both biases."""
    indices = torch.as_tensor(indices, dtype=torch.long, device=responses.device)
    if not len(indices):
        return fc.bias.unsqueeze(0).expand(responses.shape[0], -1)
    return F.linear(responses.index_select(1, indices), fc.weight.index_select(1, indices), fc.bias)


def array_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {key: None for key in ("mean", "p10", "p50", "p90", "min", "max")}
    return {"mean": float(np.mean(values)), "p10": float(np.percentile(values, 10)),
            "p50": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "min": float(values.min()), "max": float(values.max())}


@torch.inference_mode()
def analyze_model(name, model, config, dataset, gt, sizes, micro_batch, selection_enabled):
    device = next(model.parameters()).device
    fc = model.head.fc
    weight = fc.weight.detach()
    delta = (weight[1] - weight[0]).cpu().numpy()
    selections = build_selections(weight, sizes, selection_enabled)
    frames = {key: [] for key in selections}
    abnormal_frames = {key: [] for key in selections}
    abnormal_labels, video_labels, video_scores = [], [], []
    group_sums = {name: np.zeros(len(delta), dtype=np.float64) for name in ("all", "normal", "abnormal")}
    group_abs_sums = {name: np.zeros(len(delta), dtype=np.float64) for name in group_sums}
    group_counts = {name: 0 for name in group_sums}
    cursor = 0
    for i in range(len(dataset)):
        chunks, label, length, video_id, lengths = dataset[i]
        out = model.forward_video(chunks.to(device), lengths.to(device), micro_batch)
        responses = out["responses"][0].transpose(0, 1)
        if responses.shape != (length, len(delta)):
            raise ValueError(f"Invalid response dimensions for {video_id}")
        target = gt[cursor:cursor + length * 16]
        cursor += length * 16
        if len(target) != length * 16:
            raise ValueError(f"Ground-truth length mismatch at {video_id}")
        abnormal = label != "Normal"
        if not abnormal and target.any():
            raise ValueError(f"Unexpected anomaly GT in normal video {video_id}; check test-list order")
        if abnormal:
            abnormal_labels.append(target)
        video_labels.append(int(abnormal))
        video_scores.append(float(out["video_logits"].softmax(-1)[0, 1]))
        contributions = out["contributions"][0].cpu().numpy().astype(np.float64)
        for group in ("all", "abnormal" if abnormal else "normal"):
            group_counts[group] += 1
            group_sums[group] += contributions
            group_abs_sums[group] += np.abs(contributions)
        for selection, indices in selections.items():
            logits = selection_logits(responses, fc, indices)
            if selection == "all" and not torch.allclose(logits, out["frame_logits"][0], rtol=1e-5, atol=1e-5):
                raise AssertionError("Full-channel frame logits differ from original evaluator")
            scores = logits.softmax(-1)[:, 1].cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite scores for {video_id}")
            frame = np.repeat(scores, 16)
            frames[selection].append(frame)
            if abnormal:
                abnormal_frames[selection].append(frame)
    if cursor != len(gt):
        raise ValueError(f"Consumed {cursor} GT frames; file contains {len(gt)}")
    rows = []
    for selection, indices in selections.items():
        scores = np.concatenate(frames[selection])
        metric = metrics_binary(gt, scores)
        ano = metrics_binary(np.concatenate(abnormal_labels), np.concatenate(abnormal_frames[selection])) if abnormal_labels else {"auc": None}
        rows.append({"model": name, "selection": selection, "channels_used": len(indices),
                     "frame_auc": metric["auc"], "frame_ap": metric["ap"], "ano_auc": ano["auc"],
                     "exact_one_fraction": float(np.mean(scores == 1)),
                     "exact_zero_fraction": float(np.mean(scores == 0)),
                     "video_auc_full_model": metrics_binary(video_labels, video_scores)["auc"]})
    summary = {"model": name, "epoch": None, "lambda_fc_l1": config.get("lambda_fc_l1", 0.0),
               "fc_l1_target": config.get("fc_l1_target", "weight"),
               "lambda_source": "checkpoint" if "lambda_fc_l1" in config else "legacy_default_zero",
               "channels": len(delta), "fc_weight_l1": float(weight.abs().sum()),
               "abs_delta": array_stats(np.abs(delta)), "delta_positive_count": int((delta > 0).sum()),
               "delta_negative_count": int((delta < 0).sum()),
               "abs_delta_below_0_001_count": int((np.abs(delta) < 0.001).sum()),
               "video_counts": group_counts}
    kernel_rows = []
    for j, value in enumerate(delta):
        row = {"model": name, "channel": j,
               "kernel_size": config["kernel_sizes"][j // config["channels_per_branch"]],
               "weight_normal": float(weight[0, j]), "weight_abnormal": float(weight[1, j]),
               "delta_w": float(value), "abs_delta_w": float(abs(value))}
        for group, n in group_counts.items():
            row[f"mean_contribution_{group}"] = float(group_sums[group][j] / n) if n else None
            row[f"mean_abs_contribution_{group}"] = float(group_abs_sums[group][j] / n) if n else None
        for selection, indices in selections.items():
            row[f"selected_{selection}"] = int(j in indices)
        kernel_rows.append(row)
    return summary, kernel_rows, rows


def main(args):
    if args.micro_batch < 1:
        raise ValueError("micro-batch must be positive")
    p0, p1 = Path(args.no_l1_checkpoint).resolve(), Path(args.l1_checkpoint).resolve()
    if p0 == p1:
        raise ValueError("Supply two distinct checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoints sequentially to avoid keeping two full LGT models on GPU.
    model, config0, epoch0 = load_checkpoint(p0, device)
    saved1 = torch.load(p1, map_location="cpu", weights_only=True)
    if saved1.get("model_type") != "multiscale_video_cnn":
        raise ValueError("Invalid L1 checkpoint")
    config1, epoch1 = saved1["config"], saved1["epoch"]
    validate_pair(config0, config1, epoch0, epoch1, args.expected_epoch)
    del saved1
    dataset = UCFCNNFeatureDataset(args.feature_root, args.test_list, config0["visual_length"])
    if args.expected_videos and len(dataset) != args.expected_videos:
        raise ValueError(f"Expected {args.expected_videos} videos, got {len(dataset)}")
    ids = [row["video_id"] for row in dataset.rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate video IDs in test list")
    gt = np.load(args.gt_path).reshape(-1)
    if not np.isin(gt, [0, 1]).all():
        raise ValueError("Ground truth must be binary")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}; epoch: {epoch0}; videos: {len(dataset)}; GT frames: {len(gt)}", flush=True)
    print("Compare distributions only: matching kernel indices across models is not assumed.", flush=True)
    print("Analyzing no-L1 checkpoint on full test set...", flush=True)
    s0, kernels0, metrics0 = analyze_model("no_l1", model, config0, dataset, gt, args.top_k, args.micro_batch, True)
    s0["epoch"] = epoch0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, _, _ = load_checkpoint(p1, device)
    print("Analyzing L1 checkpoint and all kernel selections on full test set...", flush=True)
    s1, kernels1, metrics1 = analyze_model("l1", model, config1, dataset, gt, args.top_k, args.micro_batch, True)
    s1["epoch"] = epoch1
    # Each CSV is self-contained; never compare learned kernel identities by index.
    write_csv(outdir / "weights_no_l1.csv", kernels0)
    write_csv(outdir / "weights_l1.csv", kernels1)
    write_csv(outdir / "selection_metrics.csv", metrics0 + metrics1)
    by_size = []
    for summary, kernels in ((s0, kernels0), (s1, kernels1)):
        for size in config0["kernel_sizes"]:
            subset = [row for row in kernels if row["kernel_size"] == size]
            row = {"model": summary["model"], "kernel_size": size, "count": len(subset),
                   "mean_abs_delta_w": float(np.mean([r["abs_delta_w"] for r in subset])),
                   "mean_abs_contribution_abnormal": float(np.mean([r["mean_abs_contribution_abnormal"] for r in subset])) if summary["video_counts"]["abnormal"] else None}
            for key in ("positive_delta",) + tuple(f"top_{k}_abs_delta" for k in args.top_k):
                row[f"selected_{key}"] = sum(r[f"selected_{key}"] for r in subset)
            by_size.append(row)
    write_csv(outdir / "kernel_size_summary.csv", by_size)
    provenance = {"no_l1_checkpoint": str(p0), "l1_checkpoint": str(p1), "epoch": epoch0,
                  "test_list": str(Path(args.test_list).resolve()), "gt_path": str(Path(args.gt_path).resolve()),
                  "test_list_sha256": hashlib.sha256(Path(args.test_list).read_bytes()).hexdigest(),
                  "videos": len(dataset), "gt_frames": len(gt), "top_k": args.top_k,
                  "selection_rule": "abs(delta_w), or delta_w>0; selection uses FC weights only, not test GT",
                  "frame_score": "original FC per frame with selected response channels and unchanged biases, then softmax",
                  "caution": "Exploratory test-set results, not validation; do not select K on test and report as unbiased.",
                  "kernel_identity": "channel indices are not comparable across independently trained models",
                  "weight_summaries": [s0, s1]}
    (outdir / "metadata.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print("\nWEIGHT SUMMARIES:\n" + json.dumps([s0, s1], indent=2), flush=True)
    print("\nKERNEL SIZE SUMMARY:", flush=True)
    for row in by_size:
        print(json.dumps(row), flush=True)
    print("\nFRAME METRICS (percent):", flush=True)
    for row in metrics0 + metrics1:
        print(json.dumps(row), flush=True)
    print("\nOutputs:", outdir, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
