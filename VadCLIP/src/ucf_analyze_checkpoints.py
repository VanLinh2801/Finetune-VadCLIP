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
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from model import CLIPVAD
from model_description import CLIPVADDescription
from model_multi_prompt import CLIPVADMultiPrompt
from model_prompt_replacement import CLIPVADPromptReplacement
from model_prompt_residual import CLIPVADPromptResidual
from model_multi_prompt_residual import CLIPVADMultiPromptResidual
from ucf_evaluate import UCFRelativeTestDataset
from utils.prompt_replacement import build_compact_prompt_groups, build_multi_prompt_groups, build_multi_prompt_weights
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP


LABEL_MAP = {
    "Normal": "Normal",
    "Abuse": "Abuse",
    "Arrest": "Arrest",
    "Arson": "Arson",
    "Assault": "Assault",
    "Burglary": "Burglary",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "RoadAccidents": "RoadAccidents",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Shoplifting": "Shoplifting",
    "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}

PROMPT_REPLACEMENT_LABEL_MAP = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "roadAccidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def build_model(args, model_type, device):
    if model_type == "description":
        model_cls = CLIPVADDescription
    elif model_type == "multi_prompt":
        model_cls = CLIPVADMultiPrompt
    elif model_type == "multi_prompt_residual":
        model_cls = CLIPVADMultiPromptResidual
    elif model_type == "prompt_replacement":
        model_cls = CLIPVADPromptReplacement
    elif model_type == "prompt_residual":
        model_cls = CLIPVADPromptResidual
    else:
        model_cls = CLIPVAD
    return model_cls(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )


def load_model(args, checkpoint, model_type, device):
    model = build_model(args, model_type, device)
    if model_type == "prompt_residual":
        prompt_text = get_prompt_text_for_model_type(args, model_type)
        prompt_weights = get_prompt_weights_for_model_type(args, model_type, device)
        model.to(device)
        model.init_class_prototypes(prompt_text, prompt_weights=prompt_weights)
    if model_type == "multi_prompt_residual":
        prompt_text = get_prompt_text_for_model_type(args, model_type)
        model.to(device)
        model.init_prompt_embeddings(prompt_text)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    if model_type == "prompt_residual":
        model.prototypes_initialized = True
    if model_type == "multi_prompt_residual":
        model.prompts_initialized = True
    model.to(device)
    model.eval()
    return model


def checkpoint_sort_key(path):
    stem = Path(path).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1])
    return 10**9


def discover_checkpoints(args):
    checkpoints = [("baseline", Path(args.baseline_model_path), "baseline")]
    checkpoint_dir = Path(args.epoch_checkpoint_dir)
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.pth"), key=checkpoint_sort_key):
            checkpoints.append((path.stem, path, args.finetuned_model_type))
    if args.description_model_path:
        checkpoints.append(("final", Path(args.description_model_path), args.finetuned_model_type))
    return checkpoints


def get_prompt_text_for_model_type(args, model_type):
    if model_type in ("prompt_replacement", "prompt_residual"):
        prompt_mode = getattr(args, "prompt_mode", None) or args.prompt_replacement_mode
        if args.prototype_source == "compact":
            return build_compact_prompt_groups(
                args.prototype_json,
                PROMPT_REPLACEMENT_LABEL_MAP,
                prompt_mode if prompt_mode in ("prototype_only", "class_plus_prototype") else "class_plus_prototype",
                args.compact_prototype_types,
            )
        return build_multi_prompt_groups(
            args.prototype_json,
            PROMPT_REPLACEMENT_LABEL_MAP,
            args.prototype_source,
            args.compact_prototype_types,
            args.long_prototype_types,
            prompt_mode,
        )
    if model_type in ("multi_prompt", "multi_prompt_residual"):
        return build_multi_prompt_groups(
            args.prototype_json,
            PROMPT_REPLACEMENT_LABEL_MAP,
            args.prototype_source,
            args.compact_prototype_types,
            args.long_prototype_types,
            args.prompt_mode,
        )
    return get_prompt_text(LABEL_MAP)


def get_prompt_weights_for_model_type(args, model_type, device):
    if model_type in ("multi_prompt", "multi_prompt_residual", "prompt_replacement", "prompt_residual"):
        return build_multi_prompt_weights(args.prompt_weights_json, PROMPT_REPLACEMENT_LABEL_MAP, device)
    return None


def run_inference(model, dataloader, args, gt, gtsegments, gtlabels, device, model_type):
    prompt_text = get_prompt_text_for_model_type(args, model_type)
    prompt_weights = get_prompt_weights_for_model_type(args, model_type, device)
    classifier_scores = []
    alignment_scores = []
    element_logits2_stack = []
    video_records = []
    gt_cursor = 0

    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            video_id = item[3][0] if isinstance(item[3], (list, tuple)) else item[3]
            length = int(item[2])
            len_cur = length
            if len_cur < args.visual_length:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)
            length_remaining = length
            lengths = torch.zeros(int(length / args.visual_length) + 1)
            for j in range(int(length / args.visual_length) + 1):
                if j == 0 and length_remaining < args.visual_length:
                    lengths[j] = length_remaining
                elif j == 0 and length_remaining > args.visual_length:
                    lengths[j] = args.visual_length
                    length_remaining -= args.visual_length
                elif length_remaining > args.visual_length:
                    lengths[j] = args.visual_length
                    length_remaining -= args.visual_length
                else:
                    lengths[j] = length_remaining
            lengths = lengths.to(int)

            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            if prompt_weights is not None:
                _, logits1, logits2 = model(
                    visual,
                    padding_mask,
                    prompt_text,
                    lengths,
                    prompt_weights=prompt_weights,
                )
            else:
                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            classifier_clip = torch.sigmoid(logits1[:len_cur].squeeze(-1)).detach().cpu().numpy()
            alignment_clip = (1 - logits2[:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)).detach().cpu().numpy()
            classifier_frame = np.repeat(classifier_clip, 16)
            alignment_frame = np.repeat(alignment_clip, 16)
            gt_frame = gt[gt_cursor: gt_cursor + len(classifier_frame)]
            gt_cursor += len(classifier_frame)

            classifier_scores.append(classifier_frame)
            alignment_scores.append(alignment_frame)
            element_logits2 = logits2[:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2_stack.append(np.repeat(element_logits2, 16, 0))
            video_records.append(
                {
                    "video_id": video_id,
                    "label": label,
                    "gt": gt_frame,
                    "classifier": classifier_frame,
                    "alignment": alignment_frame,
                }
            )

    classifier_scores = np.concatenate(classifier_scores)
    alignment_scores = np.concatenate(alignment_scores)
    if len(classifier_scores) != len(gt):
        raise ValueError(f"Score length {len(classifier_scores)} does not match gt length {len(gt)}")

    metrics = compute_metrics(
        classifier_scores,
        alignment_scores,
        video_records,
        element_logits2_stack,
        gt,
        gtsegments,
        gtlabels,
    )
    return metrics, video_records


def compute_metrics(classifier_scores, alignment_scores, video_records, element_logits2_stack, gt, gtsegments, gtlabels):
    classifier_auc = roc_auc_score(gt, classifier_scores) * 100
    classifier_ap = average_precision_score(gt, classifier_scores) * 100
    alignment_auc = roc_auc_score(gt, alignment_scores) * 100
    alignment_ap = average_precision_score(gt, alignment_scores) * 100
    best_branch = "alignment" if alignment_auc >= classifier_auc else "classifier"
    best_auc = max(classifier_auc, alignment_auc)

    abnormal_gt = []
    abnormal_scores = []
    for record in video_records:
        if record["label"] != "Normal":
            abnormal_gt.append(record["gt"])
            abnormal_scores.append(record[best_branch])
    ano_auc = roc_auc_score(np.concatenate(abnormal_gt), np.concatenate(abnormal_scores)) * 100

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    metrics = {
        "classifier_auc": float(classifier_auc),
        "classifier_ap": float(classifier_ap),
        "alignment_auc": float(alignment_auc),
        "alignment_ap": float(alignment_ap),
        "best_auc": float(best_auc),
        "best_branch": best_branch,
        "ano_auc": float(ano_auc),
        "avg_mAP": float(np.mean(dmap)),
    }
    for threshold, value in zip(iou, dmap):
        metrics[f"mAP@{threshold:.1f}"] = float(value)
    return metrics


def per_class_metrics(video_records, branch):
    rows = []
    for label in sorted({record["label"] for record in video_records}):
        gt = np.concatenate([record["gt"] for record in video_records if record["label"] == label])
        scores = np.concatenate([record[branch] for record in video_records if record["label"] == label])
        if len(np.unique(gt)) < 2:
            auc = math.nan
            ap = math.nan
        else:
            auc = roc_auc_score(gt, scores) * 100
            ap = average_precision_score(gt, scores) * 100
        rows.append({"label": label, "auc": auc, "ap": ap})
    return rows


def save_metrics_csv(path, metrics_rows):
    fieldnames = [
        "checkpoint",
        "model_type",
        "classifier_auc",
        "classifier_ap",
        "alignment_auc",
        "alignment_ap",
        "best_auc",
        "best_branch",
        "ano_auc",
        "mAP@0.1",
        "mAP@0.2",
        "mAP@0.3",
        "mAP@0.4",
        "mAP@0.5",
        "avg_mAP",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_per_class_csv(path, per_class_rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "branch", "label", "auc", "ap"])
        writer.writeheader()
        writer.writerows(per_class_rows)


def metric_array(metrics_rows, metric):
    return [row[metric] for row in metrics_rows]


def save_plot(output_path):
    plt.savefig(output_path, dpi=180)
    plt.close()
    print("Saved plot:", output_path)


def plot_metric_trends(metrics_rows, output_dir):
    checkpoints = [row["checkpoint"] for row in metrics_rows]
    metrics = ["classifier_auc", "alignment_auc", "ano_auc", "classifier_ap", "alignment_ap", "avg_mAP"]
    plt.figure(figsize=(13, 7))
    for metric in metrics:
        plt.plot(checkpoints, metric_array(metrics_rows, metric), marker="o", label=metric)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Metric Trends Across Checkpoints")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir / "01_metric_trends.png")


def plot_delta(metrics_rows, output_dir):
    baseline = metrics_rows[0]
    checkpoints = [row["checkpoint"] for row in metrics_rows[1:]]
    metrics = ["classifier_auc", "alignment_auc", "ano_auc", "avg_mAP"]
    x = np.arange(len(checkpoints))
    width = 0.2
    plt.figure(figsize=(13, 6))
    for idx, metric in enumerate(metrics):
        values = [row[metric] - baseline[metric] for row in metrics_rows[1:]]
        plt.bar(x + (idx - 1.5) * width, values, width, label=metric)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, checkpoints, rotation=35, ha="right")
    plt.ylabel("Delta vs baseline")
    plt.title("Checkpoint Delta vs Baseline")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir / "02_delta_vs_baseline.png")


def plot_map_by_iou(metrics_rows, output_dir):
    checkpoints = [row["checkpoint"] for row in metrics_rows]
    keys = ["mAP@0.1", "mAP@0.2", "mAP@0.3", "mAP@0.4", "mAP@0.5"]
    x = np.arange(len(keys))
    width = min(0.8 / len(metrics_rows), 0.16)
    plt.figure(figsize=(12, 6))
    for idx, row in enumerate(metrics_rows):
        values = [row[key] for key in keys]
        plt.bar(x + (idx - len(metrics_rows) / 2) * width, values, width, label=row["checkpoint"])
    plt.xticks(x, keys)
    plt.ylabel("mAP")
    plt.title("Fine-Grained mAP By IoU Threshold")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_plot(output_dir / "03_map_by_iou.png")


def plot_branch_auc(metrics_rows, output_dir):
    checkpoints = [row["checkpoint"] for row in metrics_rows]
    x = np.arange(len(checkpoints))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, [row["classifier_auc"] for row in metrics_rows], width, label="classifier_auc")
    plt.bar(x + width / 2, [row["alignment_auc"] for row in metrics_rows], width, label="alignment_auc")
    plt.xticks(x, checkpoints, rotation=35, ha="right")
    plt.ylabel("AUC")
    plt.title("Classifier Branch vs Alignment Branch")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir / "04_branch_auc_comparison.png")


def plot_curves(all_outputs, gt, output_dir):
    plt.figure(figsize=(8, 7))
    for checkpoint, outputs in all_outputs.items():
        fpr, tpr, _ = roc_curve(gt, outputs["classifier_scores"])
        plt.plot(fpr, tpr, label=checkpoint)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classifier ROC Curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_plot(output_dir / "05_classifier_roc_curves.png")

    plt.figure(figsize=(8, 7))
    for checkpoint, outputs in all_outputs.items():
        precision, recall, _ = precision_recall_curve(gt, outputs["classifier_scores"])
        plt.plot(recall, precision, label=checkpoint)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Classifier PR Curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_plot(output_dir / "06_classifier_pr_curves.png")


def plot_score_distributions(all_outputs, output_dir):
    for branch, filename in [("classifier_scores", "07_classifier_score_distribution.png"), ("alignment_scores", "08_alignment_score_distribution.png")]:
        plt.figure(figsize=(13, 6))
        for idx, (checkpoint, outputs) in enumerate(all_outputs.items()):
            normal_scores = outputs[branch][outputs["gt"] == 0]
            abnormal_scores = outputs[branch][outputs["gt"] == 1]
            positions = [idx * 3 + 1, idx * 3 + 2]
            plt.boxplot(
                [normal_scores, abnormal_scores],
                positions=positions,
                widths=0.6,
                showfliers=False,
                tick_labels=["N", "A"],
            )
            plt.text(idx * 3 + 1.5, 1.03, checkpoint, ha="center", fontsize=8, rotation=25)
        plt.ylim(-0.02, 1.08)
        plt.ylabel("Score")
        plt.title(branch.replace("_", " ").title() + " Normal vs Abnormal Distribution")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        save_plot(output_dir / filename)


def plot_per_class_heatmap(per_class_rows, output_dir):
    labels = sorted({row["label"] for row in per_class_rows if row["label"] != "Normal"})
    checkpoints = []
    for row in per_class_rows:
        if row["checkpoint"] not in checkpoints and row["branch"] == "classifier":
            checkpoints.append(row["checkpoint"])
    matrix = np.full((len(labels), len(checkpoints)), np.nan)
    lookup = {(row["label"], row["checkpoint"]): row["auc"] for row in per_class_rows if row["branch"] == "classifier"}
    for i, label in enumerate(labels):
        for j, checkpoint in enumerate(checkpoints):
            matrix[i, j] = lookup.get((label, checkpoint), np.nan)

    plt.figure(figsize=(max(10, len(checkpoints) * 1.2), 8))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Per-class classifier AUC")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(len(checkpoints)), checkpoints, rotation=35, ha="right")
    plt.title("Per-Class AUC Heatmap")
    plt.tight_layout()
    save_plot(output_dir / "09_per_class_auc_heatmap.png")


def plot_predicted_duration(all_outputs, output_dir, threshold=0.5):
    labels = []
    classifier_duration = []
    alignment_duration = []
    for checkpoint, outputs in all_outputs.items():
        labels.append(checkpoint)
        classifier_duration.append(float(np.mean(outputs["classifier_scores"] >= threshold) * 100))
        alignment_duration.append(float(np.mean(outputs["alignment_scores"] >= threshold) * 100))

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, classifier_duration, width, label="classifier")
    plt.bar(x + width / 2, alignment_duration, width, label="alignment")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(f"Frames with score >= {threshold} (%)")
    plt.title("Predicted Anomaly Duration")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir / "10_predicted_anomaly_duration.png")


def plot_score_correlation(all_outputs, output_dir):
    if "baseline" not in all_outputs:
        return
    baseline_scores = all_outputs["baseline"]["classifier_scores"]
    labels = []
    correlations = []
    mean_abs_delta = []
    for checkpoint, outputs in all_outputs.items():
        labels.append(checkpoint)
        scores = outputs["classifier_scores"]
        correlations.append(float(np.corrcoef(baseline_scores, scores)[0, 1]))
        mean_abs_delta.append(float(np.mean(np.abs(scores - baseline_scores))))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(labels, correlations, marker="o", color="tab:blue", label="corr with baseline")
    ax1.set_ylabel("Pearson correlation", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(labels, mean_abs_delta, marker="s", color="tab:red", label="mean abs delta")
    ax2.set_ylabel("Mean absolute score delta", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.xticks(rotation=35, ha="right")
    plt.title("Classifier Score Drift vs Baseline")
    fig.tight_layout()
    save_plot(output_dir / "11_score_drift_vs_baseline.png")


def select_timeline_videos(baseline_records, count):
    abnormal = [record for record in baseline_records if record["label"] != "Normal" and record["gt"].sum() > 0]
    normal = [record for record in baseline_records if record["label"] == "Normal"]
    selected = abnormal[: max(1, count - 1)]
    if normal:
        selected.append(normal[0])
    return [record["video_id"] for record in selected[:count]]


def plot_timelines(all_video_records, output_dir, count):
    baseline_records = all_video_records["baseline"]
    selected_ids = select_timeline_videos(baseline_records, count)
    for video_id in selected_ids:
        plt.figure(figsize=(14, 5))
        for checkpoint, records in all_video_records.items():
            record = next((item for item in records if item["video_id"] == video_id), None)
            if record is None:
                continue
            x = np.arange(len(record["classifier"])) / 16.0
            plt.plot(x, record["classifier"], linewidth=1.2, label=f"{checkpoint} classifier")
        baseline_record = next(item for item in baseline_records if item["video_id"] == video_id)
        x = np.arange(len(baseline_record["gt"])) / 16.0
        plt.fill_between(x, 0, baseline_record["gt"], color="tab:red", alpha=0.2, label="GT anomaly")
        plt.ylim(-0.03, 1.05)
        plt.xlabel("Time unit")
        plt.ylabel("Score")
        plt.title(f"Timeline: {video_id} ({baseline_record['label']})")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_id)
        save_plot(output_dir / f"12_timeline_{safe_id}.png")


def save_npz(output_dir, checkpoint, gt, classifier_scores, alignment_scores):
    np.savez_compressed(
        output_dir / f"scores_{checkpoint}.npz",
        gt=gt,
        classifier_scores=classifier_scores,
        alignment_scores=alignment_scores,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics_rows = []
    per_class_rows = []
    all_outputs = {}
    all_video_records = {}

    for checkpoint_name, checkpoint_path, model_type in discover_checkpoints(args):
        print(f"Evaluating {checkpoint_name}: {checkpoint_path}")
        model = load_model(args, checkpoint_path, model_type, device)
        metrics, video_records = run_inference(model, dataloader, args, gt, gtsegments, gtlabels, device, model_type)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        classifier_scores = np.concatenate([record["classifier"] for record in video_records])
        alignment_scores = np.concatenate([record["alignment"] for record in video_records])
        metrics_rows.append({"checkpoint": checkpoint_name, "model_type": model_type, **metrics})
        all_outputs[checkpoint_name] = {
            "gt": gt,
            "classifier_scores": classifier_scores,
            "alignment_scores": alignment_scores,
        }
        all_video_records[checkpoint_name] = video_records
        save_npz(output_dir, checkpoint_name, gt, classifier_scores, alignment_scores)

        for branch in ["classifier", "alignment"]:
            for row in per_class_metrics(video_records, branch):
                per_class_rows.append({"checkpoint": checkpoint_name, "branch": branch, **row})

    save_metrics_csv(output_dir / "metrics_by_checkpoint.csv", metrics_rows)
    save_per_class_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    with open(output_dir / "metrics_by_checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    plot_metric_trends(metrics_rows, output_dir)
    plot_delta(metrics_rows, output_dir)
    plot_map_by_iou(metrics_rows, output_dir)
    plot_branch_auc(metrics_rows, output_dir)
    plot_curves(all_outputs, gt, output_dir)
    plot_score_distributions(all_outputs, output_dir)
    plot_per_class_heatmap(per_class_rows, output_dir)
    plot_predicted_duration(all_outputs, output_dir)
    plot_score_correlation(all_outputs, output_dir)
    plot_timelines(all_video_records, output_dir, args.timeline_count)
    print("Saved diagnostics to:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze VadCLIP checkpoints with metrics and diagnostic plots.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--baseline-model-path", default="../../model_ucf.pth")
    parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints")
    parser.add_argument("--description-model-path", default="model/model_ucf_description.pth")
    parser.add_argument("--finetuned-model-type", default="description", choices=["baseline", "description", "prompt_replacement", "prompt_residual", "multi_prompt", "multi_prompt_residual"])
    parser.add_argument("--output-dir", default="../../code/ucf_checkpoint_diagnostics")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
    parser.add_argument("--timeline-count", default=6, type=int)

    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--prompt-replacement-mode", default="manual_caption", choices=["prototype_only", "class_plus_prototype", "manual_caption"])
    parser.add_argument("--prototype-json", default="../../code/ucf_weighted_class_prompts.json")
    parser.add_argument("--prompt-weights-json", default=None)
    parser.add_argument("--prototype-source", default="manual", choices=["manual", "compact", "long_event_only"])
    parser.add_argument(
        "--prompt-mode",
        default="manual_caption",
        choices=["manual_caption", "class_plus_prototype", "prototype_only"],
    )
    parser.add_argument(
        "--compact-prototype-types",
        default=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
        nargs="+",
        choices=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
    )
    parser.add_argument(
        "--long-prototype-types",
        default=["core_action", "actor_object_interaction", "temporal_progression"],
        nargs="+",
        choices=[
            "core_action",
            "actor_object_interaction",
            "temporal_progression",
            "scene_context",
            "hard_negative_distinction",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
