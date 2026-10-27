import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model import CLIPVAD
from model_description import CLIPVADDescription
from model_multi_prompt import CLIPVADMultiPrompt
from model_prompt_replacement import CLIPVADPromptReplacement
from ucf_evaluate import UCFRelativeTestDataset
from utils.prompt_replacement import (
    build_compact_prompt_groups,
    build_multi_prompt_groups,
    build_multi_prompt_weights,
)
from utils.tools import get_batch_mask, get_prompt_text


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

PROMPT_LABEL_MAP = {
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

BRANCHES = ["classifier", "alignment"]


def build_model(args, model_type, device):
    if model_type == "description":
        model_cls = CLIPVADDescription
    elif model_type == "multi_prompt":
        model_cls = CLIPVADMultiPrompt
    elif model_type == "prompt_replacement":
        model_cls = CLIPVADPromptReplacement
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


def load_model(args, config, device):
    model_type = config.get("model_type", "baseline")
    model = build_model(args, model_type, device)
    model_path = Path(config["model_path"])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_prompt_text_for_config(config):
    model_type = config.get("model_type", "baseline")
    if model_type == "prompt_replacement":
        return build_compact_prompt_groups(
            config["prototype_json"],
            PROMPT_LABEL_MAP,
            config.get("prompt_replacement_mode", "class_plus_prototype"),
            config.get(
                "compact_prototype_types",
                ["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
            ),
        )
    if model_type == "multi_prompt":
        return build_multi_prompt_groups(
            config["prototype_json"],
            PROMPT_LABEL_MAP,
            config.get("prototype_source", "manual"),
            config.get(
                "compact_prototype_types",
                ["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
            ),
            config.get(
                "long_prototype_types",
                ["core_action", "actor_object_interaction", "temporal_progression"],
            ),
            config.get("prompt_mode", "manual_caption"),
        )
    return get_prompt_text(LABEL_MAP)


def get_prompt_weights_for_config(config, device):
    if config.get("model_type", "baseline") != "multi_prompt":
        return None
    return build_multi_prompt_weights(config.get("prompt_weights_json"), PROMPT_LABEL_MAP, device)


def make_lengths(length, visual_length):
    length_remaining = int(length)
    lengths = torch.zeros(int(length / visual_length) + 1)
    for idx in range(int(length / visual_length) + 1):
        if idx == 0 and length_remaining < visual_length:
            lengths[idx] = length_remaining
        elif idx == 0 and length_remaining > visual_length:
            lengths[idx] = visual_length
            length_remaining -= visual_length
        elif length_remaining > visual_length:
            lengths[idx] = visual_length
            length_remaining -= visual_length
        else:
            lengths[idx] = length_remaining
    return lengths.to(int)


def run_inference(model, dataloader, args, config, gt, device):
    prompt_text = get_prompt_text_for_config(config)
    prompt_weights = get_prompt_weights_for_config(config, device)

    video_records = []
    all_classifier = []
    all_alignment = []
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

            lengths = make_lengths(length, args.visual_length)
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

            classifier_snippet = torch.sigmoid(logits1[:len_cur].squeeze(-1)).detach().cpu().numpy()
            alignment_snippet = (
                1 - logits2[:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)
            ).detach().cpu().numpy()

            classifier_frame = np.repeat(classifier_snippet, 16)
            alignment_frame = np.repeat(alignment_snippet, 16)
            gt_frame = gt[gt_cursor : gt_cursor + len(classifier_frame)]
            gt_cursor += len(classifier_frame)

            video_records.append(
                {
                    "video_id": str(video_id),
                    "label": str(label),
                    "gt": gt_frame,
                    "classifier": classifier_frame,
                    "alignment": alignment_frame,
                }
            )
            all_classifier.append(classifier_frame)
            all_alignment.append(alignment_frame)

    all_classifier = np.concatenate(all_classifier)
    all_alignment = np.concatenate(all_alignment)
    if len(all_classifier) != len(gt):
        raise ValueError(f"Score length {len(all_classifier)} does not match gt length {len(gt)}")

    return video_records, {"classifier": all_classifier, "alignment": all_alignment}


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score) * 100)


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score) * 100)


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def topk_overlap(scores, gt, top_k_ratio):
    scores = np.asarray(scores, dtype=np.float32)
    gt = np.asarray(gt)
    if scores.size == 0:
        return np.nan, 0, 0
    top_k = max(1, int(scores.size * top_k_ratio))
    top_k = min(top_k, scores.size)
    top_indices = np.argsort(scores)[-top_k:]
    hits = int(np.sum(gt[top_indices] == 1))
    return float(hits / top_k), hits, int(top_k)


def top_bottom_gap(scores, top_k_ratio):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size < 2:
        return np.nan
    top_k = max(1, int(scores.size * top_k_ratio))
    top_k = min(top_k, scores.size)
    if top_k * 2 > scores.size:
        return np.nan
    sorted_scores = np.sort(scores)
    return float(np.mean(sorted_scores[-top_k:]) - np.mean(sorted_scores[:top_k]))


def quantiles(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {f"p{q}": np.nan for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
    return {
        f"p{q}": float(np.percentile(values, q))
        for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }


def summarize(values):
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float32)
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def analyze_model(model_name, video_records, all_scores, gt, top_k_ratio):
    rows_per_video = []
    for record in video_records:
        row = {
            "model": model_name,
            "video_id": record["video_id"],
            "label": record["label"],
            "num_frames": int(len(record["gt"])),
            "positive_frames": int(np.sum(record["gt"] == 1)),
        }
        for branch in BRANCHES:
            overlap, hits, top_k = topk_overlap(record[branch], record["gt"], top_k_ratio)
            row[f"{branch}_topk_overlap"] = overlap
            row[f"{branch}_topk_hits"] = hits
            row[f"{branch}_topk_total"] = top_k
            row[f"{branch}_top_bottom_gap"] = top_bottom_gap(record[branch], top_k_ratio)
            row[f"{branch}_mean"] = float(np.mean(record[branch]))
            row[f"{branch}_max"] = float(np.max(record[branch]))
        row["classifier_alignment_corr"] = safe_corr(record["classifier"], record["alignment"])
        rows_per_video.append(row)

    rows_quantile = []
    rows_summary = []
    rows_corr = []
    rows_per_class = []

    for branch in BRANCHES:
        scores = all_scores[branch]
        pos_scores = scores[gt == 1]
        neg_scores = scores[gt == 0]
        q_pos = quantiles(pos_scores)
        q_neg = quantiles(neg_scores)
        rows_quantile.append(
            {
                "model": model_name,
                "branch": branch,
                "group": "positive",
                "mean": float(np.mean(pos_scores)) if pos_scores.size else np.nan,
                **q_pos,
            }
        )
        rows_quantile.append(
            {
                "model": model_name,
                "branch": branch,
                "group": "negative",
                "mean": float(np.mean(neg_scores)) if neg_scores.size else np.nan,
                **q_neg,
            }
        )

        abnormal_rows = [row for row in rows_per_video if row["label"] != "Normal"]
        overlap_summary = summarize([row[f"{branch}_topk_overlap"] for row in abnormal_rows])
        gap_summary = summarize([row[f"{branch}_top_bottom_gap"] for row in abnormal_rows])
        rows_summary.append(
            {
                "model": model_name,
                "branch": branch,
                "auc": safe_auc(gt, scores),
                "ap": safe_ap(gt, scores),
                "positive_mean": float(np.mean(pos_scores)) if pos_scores.size else np.nan,
                "negative_mean": float(np.mean(neg_scores)) if neg_scores.size else np.nan,
                "global_gap": float(np.mean(pos_scores) - np.mean(neg_scores))
                if pos_scores.size and neg_scores.size
                else np.nan,
                "negative_p95": float(np.percentile(neg_scores, 95)) if neg_scores.size else np.nan,
                "negative_p99": float(np.percentile(neg_scores, 99)) if neg_scores.size else np.nan,
                "positive_p10": float(np.percentile(pos_scores, 10)) if pos_scores.size else np.nan,
                "positive_p50": float(np.percentile(pos_scores, 50)) if pos_scores.size else np.nan,
                "topk_overlap_mean": overlap_summary["mean"],
                "topk_overlap_median": overlap_summary["median"],
                "intra_video_gap_mean": gap_summary["mean"],
                "intra_video_gap_median": gap_summary["median"],
            }
        )

    rows_corr.append(
        {
            "model": model_name,
            "scope": "all_frames",
            "classifier_alignment_corr": safe_corr(all_scores["classifier"], all_scores["alignment"]),
        }
    )
    rows_corr.append(
        {
            "model": model_name,
            "scope": "positive_frames",
            "classifier_alignment_corr": safe_corr(
                all_scores["classifier"][gt == 1],
                all_scores["alignment"][gt == 1],
            ),
        }
    )
    rows_corr.append(
        {
            "model": model_name,
            "scope": "negative_frames",
            "classifier_alignment_corr": safe_corr(
                all_scores["classifier"][gt == 0],
                all_scores["alignment"][gt == 0],
            ),
        }
    )

    for class_name in sorted({row["label"] for row in rows_per_video}):
        class_records = [row for row in video_records if row["label"] == class_name]
        class_gt = np.concatenate([row["gt"] for row in class_records])
        for branch in BRANCHES:
            class_scores = np.concatenate([row[branch] for row in class_records])
            overlaps = [
                topk_overlap(row[branch], row["gt"], top_k_ratio)[0]
                for row in class_records
                if row["label"] != "Normal"
            ]
            gaps = [top_bottom_gap(row[branch], top_k_ratio) for row in class_records]
            rows_per_class.append(
                {
                    "model": model_name,
                    "class_name": class_name,
                    "branch": branch,
                    "num_videos": len(class_records),
                    "positive_frames": int(np.sum(class_gt == 1)),
                    "auc": safe_auc(class_gt, class_scores),
                    "ap": safe_ap(class_gt, class_scores),
                    "topk_overlap_mean": summarize(overlaps)["mean"],
                    "intra_video_gap_mean": summarize(gaps)["mean"],
                    "score_mean": float(np.mean(class_scores)),
                    "score_max": float(np.max(class_scores)),
                }
            )

    return rows_summary, rows_quantile, rows_corr, rows_per_video, rows_per_class


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_model_summary(rows_summary, rows_corr):
    print("\nScore separation summary")
    for row in rows_summary:
        print(
            f"  {row['model']} | {row['branch']} | "
            f"AUC={row['auc']:.2f} AP={row['ap']:.2f} | "
            f"global_gap={row['global_gap']:.4f} | "
            f"topk_overlap={row['topk_overlap_mean']:.4f} | "
            f"intra_gap={row['intra_video_gap_mean']:.4f} | "
            f"neg_p95={row['negative_p95']:.4f} neg_p99={row['negative_p99']:.4f} | "
            f"pos_p10={row['positive_p10']:.4f}"
        )
    print("\nClassifier-alignment correlation")
    for row in rows_corr:
        print(f"  {row['model']} | {row['scope']} | corr={row['classifier_alignment_corr']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze score separation for trained VadCLIP checkpoints.")
    parser.add_argument("--models-json", required=True, help="JSON file containing a list of model configs.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--output-dir", default="../../code/results/analyze/score_separation")
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
    parser.add_argument("--batch-size", default=1, type=int)

    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
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
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 to keep scores aligned with UCF-Crime ground truth.")

    with open(args.models_json, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    if not isinstance(model_configs, list):
        raise ValueError("--models-json must contain a list of model configs.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = np.load(args.gt_path)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_quantiles = []
    all_corr = []
    all_video_rows = []
    all_class_rows = []

    print("Score separation diagnostics")
    print(f"Device: {device}")
    print(f"Models: {len(model_configs)}")
    print(f"Feature root: {args.feature_root}")
    print(f"Test list: {args.test_list}")
    print(f"Top-k ratio: {args.top_k_ratio}")

    for config in model_configs:
        name = config["name"]
        print("\n" + "=" * 80)
        print(f"Analyzing model: {name}")
        print(f"  type: {config.get('model_type', 'baseline')}")
        print(f"  path: {config['model_path']}")
        model = load_model(args, config, device)
        video_records, all_scores = run_inference(model, dataloader, args, config, gt, device)
        rows_summary, rows_quantile, rows_corr, rows_per_video, rows_per_class = analyze_model(
            name,
            video_records,
            all_scores,
            gt,
            args.top_k_ratio,
        )
        print_model_summary(rows_summary, rows_corr)
        all_summary.extend(rows_summary)
        all_quantiles.extend(rows_quantile)
        all_corr.extend(rows_corr)
        all_video_rows.extend(rows_per_video)
        all_class_rows.extend(rows_per_class)

    write_csv(output_dir / "summary_by_model.csv", all_summary)
    write_csv(output_dir / "score_quantiles_by_model.csv", all_quantiles)
    write_csv(output_dir / "logits1_logits2_correlation.csv", all_corr)
    write_csv(output_dir / "per_video_topk_overlap.csv", all_video_rows)
    write_csv(output_dir / "per_class_score_separation.csv", all_class_rows)
    with (output_dir / "models_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_configs, f, indent=2, ensure_ascii=False)

    print("\nSaved outputs:")
    for filename in [
        "summary_by_model.csv",
        "score_quantiles_by_model.csv",
        "logits1_logits2_correlation.csv",
        "per_video_topk_overlap.csv",
        "per_class_score_separation.csv",
        "models_config.json",
    ]:
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
