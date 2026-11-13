"""Video classification and frame evidence evaluation; no training or GT fitting."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from model_multiscale_cnn import MultiscaleCNN
from ucf_cnn_common import UCFCNNFeatureDataset, TEST_LABEL_MAP


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def metrics_binary(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not len(labels) or np.unique(labels).size < 2:
        return {"auc": None, "ap": None}
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Score shape or finiteness check failed")
    return {"auc": float(roc_auc_score(labels, scores) * 100),
            "ap": float(average_precision_score(labels, scores) * 100)}


def summarize(values):
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            **{f"p{p}": float(np.percentile(values, p)) if values else None for p in (10, 50, 90)}}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path")
    source.add_argument("--baseline-model-path", help="Self-trained baseline only, not author checkpoint")
    for name in ("feature-root", "test-list", "gt-path", "evaluate-dir", "analyze-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-videos", type=int, default=290)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--top-ratio", type=float, default=0.15)
    return parser


@torch.inference_mode()
def evaluate(args):
    if args.micro_batch < 1 or not 0 < args.top_ratio <= 1:
        raise ValueError("Invalid micro-batch or top-ratio")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = bool(args.baseline_model_path)
    if baseline:
        if device.type != "cuda":
            raise RuntimeError("Original baseline distance graph requires CUDA")
        from model import CLIPVAD
        from utils.tools import get_prompt_text
        model = CLIPVAD(14, 512, 256, 512, 1, 2, 8, 10, 10, device).to(device)
        state = torch.load(args.baseline_model_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        visual_length, config = 256, {}
        prompts = get_prompt_text(TEST_LABEL_MAP)
    else:
        saved = torch.load(args.model_path, map_location="cpu", weights_only=True)
        if saved.get("model_type") != "multiscale_video_cnn":
            raise ValueError("Not a multiscale video CNN checkpoint")
        config = saved["config"]
        visual_length = config["visual_length"]
        model = MultiscaleCNN(config["kernel_sizes"], config["channels_per_branch"], visual_length,
                              activation=config.get("cnn_activation", "gelu")).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    dataset = UCFCNNFeatureDataset(args.feature_root, args.test_list, visual_length)
    if args.expected_videos and len(dataset) != args.expected_videos:
        raise ValueError(f"Expected {args.expected_videos} videos, found {len(dataset)}")
    ids = [row["video_id"] for row in dataset.rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate video IDs in test list")
    gt = np.load(args.gt_path).reshape(-1)
    if not np.isin(gt, [0, 1]).all():
        raise ValueError("Frame GT must be binary")
    ev, an = Path(args.evaluate_dir), Path(args.analyze_dir)
    ev.mkdir(parents=True, exist_ok=True)
    an.mkdir(parents=True, exist_ok=True)
    groups = {key: [] for key in ("true_high_abnormal", "false_high_abnormal", "false_high_normal", "missed_anomaly")}
    rows, evidence, frame_rows = [], [], []
    frame_scores, abnormal_gt, abnormal_scores = [], [], []
    cursor = selected_positive = selected_total = 0
    macro = []
    for index in range(len(dataset)):
        chunks, label, length, video_id, lengths = dataset[index]
        chunks, lengths = chunks.to(device), lengths.to(device)
        if baseline:
            pieces = []
            for start in range(0, len(chunks), args.micro_batch):
                lens = lengths[start:start + args.micro_batch]
                _, logits, _ = model(chunks[start:start + args.micro_batch], None, prompts, lens)
                pieces.extend(row[:int(n), 0].sigmoid() for row, n in zip(logits, lens))
            scores = torch.cat(pieces).cpu().numpy()
            video_score = prediction = None
        else:
            output = model.forward_video(chunks, lengths, args.micro_batch)
            scores = output["frame_logits"][0].softmax(-1)[:, 1].cpu().numpy()
            video_score = float(output["video_logits"].softmax(-1)[0, 1])
            prediction = int(output["video_logits"].argmax(-1)[0])
            contributions = output["contributions"][0].cpu().numpy()
            for channel, value in enumerate(contributions):
                evidence.append({"video_id": video_id, "label": label, "channel": channel,
                    "kernel_size": config["kernel_sizes"][channel // config["channels_per_branch"]],
                    "contribution": float(value), "pooled_response": float(output["pooled"][0, channel]),
                    "argmax_index": int(output["argmax"][0, channel]),
                    "video_predicted_abnormal": prediction})
        if len(scores) != length or not np.isfinite(scores).all():
            raise ValueError(f"Invalid scores for {video_id}")
        count = length * 16
        target = gt[cursor:cursor + count]
        cursor += count
        if len(target) != count:
            raise ValueError(f"GT length mismatch at {video_id}")
        positive = target.reshape(length, 16).max(1).astype(bool)
        expanded = np.repeat(scores, 16)
        frame_scores.append(expanded)
        abnormal = label != "Normal"
        if not abnormal and positive.any():
            raise ValueError(f"Normal video has anomaly GT: {video_id}; check test-list order")
        if abnormal:
            abnormal_gt.append(target)
            abnormal_scores.append(expanded)
        k = max(1, int(np.ceil(length * args.top_ratio)))
        top = np.zeros(length, dtype=bool)
        top[np.argsort(-scores, kind="stable")[:k]] = True
        n_positive = int(positive[top].sum())
        if abnormal:
            selected_positive += n_positive
            selected_total += k
            macro.append(n_positive / k)
        rows.append({"video_id": video_id, "label": label, "positions": length,
                     "video_target": int(abnormal), "video_score": video_score, "video_prediction": prediction,
                     "top15_count": k, "top15_anomaly_fraction": n_positive / k})
        for t in range(length):
            group = ("true_high_abnormal" if positive[t] else "false_high_abnormal") if top[t] and abnormal else (
                "false_high_normal" if top[t] else ("missed_anomaly" if positive[t] else "true_low"))
            if group in groups:
                groups[group].append(float(scores[t]))
            frame_rows.append({"video_id": video_id, "position": t, "group": group,
                               "anomaly_frame_fraction": float(target[t*16:(t+1)*16].mean()),
                               "score": float(scores[t])})
    if cursor != len(gt):
        raise ValueError(f"Consumed {cursor} GT frames, expected {len(gt)}; verify test-list order")
    frame_metrics = metrics_binary(gt, np.concatenate(frame_scores))
    ano = metrics_binary(np.concatenate(abnormal_gt), np.concatenate(abnormal_scores)) if abnormal_gt else {"auc": None}
    metrics = {"model_type": "baseline_self_trained" if baseline else "multiscale_video_cnn",
               "model_path": args.baseline_model_path or args.model_path, "videos": len(dataset),
               "frames": cursor, "classifier_auc": frame_metrics["auc"], "classifier_ap": frame_metrics["ap"],
               "ano_auc": ano["auc"], "epoch": None if baseline else saved["epoch"],
               "kernel_sizes": config.get("kernel_sizes"), "channels_per_branch": config.get("channels_per_branch"),
               "cnn_activation": None if baseline else config.get("cnn_activation", "gelu"),
               "fc_l1_target": None if baseline else config.get("fc_l1_target", "weight"),
               "lambda_fc_l1": None if baseline else config.get("lambda_fc_l1", 0.0)}
    if not baseline:
        video_metrics = metrics_binary([r["video_target"] for r in rows], [r["video_score"] for r in rows])
        metrics.update(video_auc=video_metrics["auc"], video_ap=video_metrics["ap"],
                       video_accuracy=100 * float(np.mean([r["video_prediction"] == r["video_target"] for r in rows])))
    diagnostic = {"definition": "Top fraction selected per video; a position is positive if any of its 16 frames is anomalous. Missed means outside top fraction, not necessarily low score.",
                  "top_ratio": args.top_ratio,
                  "top15_anomaly_fraction_micro": selected_positive / selected_total if selected_total else None,
                  "top15_anomaly_fraction_macro": float(np.mean(macro)) if macro else None,
                  "groups": {key: summarize(values) for key, values in groups.items()}}
    write_csv(ev / "per_video.csv", rows)
    write_csv(an / "kernel_evidence.csv", evidence)
    write_csv(an / "frame_scores.csv", frame_rows)
    (ev / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (an / "summary.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(json.dumps(diagnostic, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
