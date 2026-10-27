"""Measure how local snippet order affects baseline VadCLIP scores."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text


VISUAL_LENGTH = 256
EMBED_DIM = 512
VISUAL_WIDTH = 512
VISUAL_HEAD = 1
VISUAL_LAYERS = 2
ATTN_WINDOW = 8
PROMPT_PREFIX = 10
PROMPT_POSTFIX = 10
CLASS_COUNT = 14
FRAMES_PER_SNIPPET = 16

# Contiguous top-15% classifier regions selected from baseline_self_train diagnostics.
SEGMENTS = {
    "Abuse030_x264": {"false_high": (25, 30), "true_high": (81, 84)},
    "Arrest039_x264": {"false_high": (876, 938), "true_high": (592, 627)},
    "Arrest024_x264": {"false_high": (21, 37), "true_high": (154, 170)},
    "Arson007_x264": {"false_high": (372, 389), "true_high": (193, 204)},
    "Arson010_x264": {"false_high": (33, 54), "true_high": (55, 62)},
    "Assault010_x264": {"false_high": (937, 963), "true_high": (712, 729)},
    "Assault011_x264": {"false_high": (60, 67), "true_high": (48, 59)},
    "Burglary061_x264": {"false_high": (112, 136), "true_high": (263, 316)},
    "Burglary032_x264": {"false_high": (715, 737), "true_high": (203, 229)},
    "Explosion022_x264": {"false_high": (183, 191), "true_high": (139, 151)},
    "Explosion013_x264": {"false_high": (200, 206), "true_high": (162, 184)},
    "Fighting047_x264": {"false_high": (163, 167), "true_high": (88, 108)},
    "RoadAccidents016_x264": {"false_high": (45, 54), "true_high": (37, 44)},
    "RoadAccidents127_x264": {"false_high": (32, 41), "true_high": (138, 143)},
    "Shooting047_x264": {"false_high": (43, 59), "true_high": (154, 178)},
    "Shooting022_x264": {"false_high": (214, 225), "true_high": (178, 206)},
    "Shoplifting021_x264": {"false_high": (13, 28), "true_high": (129, 138)},
    "Shoplifting004_x264": {"false_high": (127, 135), "true_high": (157, 193)},
    "Stealing079_x264": {"false_high": (330, 356), "true_high": (259, 281)},
    "Vandalism028_x264": {"false_high": (99, 113), "true_high": (115, 122)},
}

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


def load_video_rows(test_list):
    with Path(test_list).open(newline="", encoding="utf-8-sig") as handle:
        return {row["video_id"]: row for row in csv.DictReader(handle)}


def chunked_forward(model, features, prompt_text, device):
    count = len(features)
    chunk_count = (count + VISUAL_LENGTH - 1) // VISUAL_LENGTH
    padded = np.zeros((chunk_count, VISUAL_LENGTH, features.shape[1]), dtype=np.float32)
    lengths = np.empty(chunk_count, dtype=np.int64)
    for chunk_index in range(chunk_count):
        start = chunk_index * VISUAL_LENGTH
        end = min(start + VISUAL_LENGTH, count)
        lengths[chunk_index] = end - start
        padded[chunk_index, : end - start] = features[start:end]

    visual = torch.from_numpy(padded).to(device)
    lengths_tensor = torch.from_numpy(lengths).to(device=device, dtype=torch.long)
    padding_mask = get_batch_mask(lengths_tensor.cpu(), VISUAL_LENGTH).to(device)
    with torch.no_grad():
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths_tensor)
    valid_logits1 = logits1.reshape(-1, logits1.shape[-1])[:count, 0]
    valid_logits2 = logits2.reshape(-1, logits2.shape[-1])[:count]
    classifier = torch.sigmoid(valid_logits1).cpu().numpy()
    alignment = (1 - valid_logits2.softmax(dim=-1)[:, 0]).cpu().numpy()
    return classifier, alignment


def shuffle_within_model_windows(features, seed):
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(features))
    for start in range(0, len(features), VISUAL_LENGTH):
        end = min(start + VISUAL_LENGTH, len(features))
        permutation[start:end] = rng.permutation(permutation[start:end])
    return features[permutation], permutation


def segment_mean(values, interval):
    start, end = interval
    return float(np.mean(values[start : end + 1]))


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
        raise RuntimeError("This VadCLIP implementation requires CUDA for graph adjacency.")
    feature_root = Path(args.feature_root)
    rows_by_video = load_video_rows(args.test_list)
    gt_frames = np.load(args.gt_path, mmap_mode="r")

    model = CLIPVAD(
        CLASS_COUNT, EMBED_DIM, VISUAL_LENGTH, VISUAL_WIDTH, VISUAL_HEAD,
        VISUAL_LAYERS, ATTN_WINDOW, PROMPT_PREFIX, PROMPT_POSTFIX, device,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    prompt_text = get_prompt_text(LABEL_MAP)

    # Verify the selected snippets against frame labels and locate each video in GT.
    cursor = 0
    gt_by_video = {}
    with Path(args.test_list).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            features = np.load(feature_root / row["path"], mmap_mode="r")
            frame_count = len(features) * FRAMES_PER_SNIPPET
            gt_by_video[row["video_id"]] = np.asarray(
                gt_frames[cursor : cursor + frame_count]
            ).reshape(-1, FRAMES_PER_SNIPPET).max(axis=1)
            cursor += frame_count
    if cursor != len(gt_frames):
        raise ValueError(f"Consumed {cursor} GT frames, expected {len(gt_frames)}.")

    records = []
    for video_index, (video_id, groups) in enumerate(SEGMENTS.items()):
        if video_id not in rows_by_video:
            raise KeyError(f"Video missing from test list: {video_id}")
        features = np.asarray(
            np.load(feature_root / rows_by_video[video_id]["path"]), dtype=np.float32
        )
        labels = gt_by_video[video_id]
        for group, (start, end) in groups.items():
            expected = 0 if group == "false_high" else 1
            if end >= len(features) or not np.all(labels[start : end + 1] == expected):
                raise ValueError(f"Selected {group} range failed GT check: {video_id} {start}-{end}")

        original_classifier, original_alignment = chunked_forward(
            model, features, prompt_text, device
        )
        shuffled_classifier_runs = []
        shuffled_alignment_runs = []
        for repeat in range(args.shuffle_repeats):
            shuffled, permutation = shuffle_within_model_windows(
                features, seed=2026 + video_index * 100 + repeat
            )
            classifier, alignment = chunked_forward(model, shuffled, prompt_text, device)
            # Restore scores to the original snippet indices for a paired comparison.
            classifier_restored = np.empty_like(classifier)
            alignment_restored = np.empty_like(alignment)
            classifier_restored[permutation] = classifier
            alignment_restored[permutation] = alignment
            shuffled_classifier_runs.append(classifier_restored)
            shuffled_alignment_runs.append(alignment_restored)

        shuffled_classifier_mean = np.mean(shuffled_classifier_runs, axis=0)
        shuffled_alignment_mean = np.mean(shuffled_alignment_runs, axis=0)
        for group, interval in groups.items():
            start, end = interval
            record = {
                "video_id": video_id,
                "class_name": rows_by_video[video_id]["label"],
                "group": group,
                "snippet_start": start,
                "snippet_end_inclusive": end,
                "snippet_count": end - start + 1,
                "original_classifier_mean": segment_mean(original_classifier, interval),
                "shuffled_classifier_mean": segment_mean(shuffled_classifier_mean, interval),
                "classifier_change": segment_mean(shuffled_classifier_mean - original_classifier, interval),
                "original_alignment_mean": segment_mean(original_alignment, interval),
                "shuffled_alignment_mean": segment_mean(shuffled_alignment_mean, interval),
                "alignment_change": segment_mean(shuffled_alignment_mean - original_alignment, interval),
            }
            records.append(record)
            print(
                f"{video_id} | {group} | classifier {record['original_classifier_mean']:.4f} "
                f"-> {record['shuffled_classifier_mean']:.4f} "
                f"(delta {record['classifier_change']:+.4f}) | alignment "
                f"{record['original_alignment_mean']:.4f} -> {record['shuffled_alignment_mean']:.4f} "
                f"(delta {record['alignment_change']:+.4f})",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_count = len(SEGMENTS)
    output_csv = output_dir / f"temporal_order_shuffle_{video_count}_videos.csv"
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    paired = {}
    for video_id in SEGMENTS:
        video_records = {row["group"]: row for row in records if row["video_id"] == video_id}
        for branch in ("classifier", "alignment"):
            original_gap = (
                video_records["true_high"][f"original_{branch}_mean"]
                - video_records["false_high"][f"original_{branch}_mean"]
            )
            shuffled_gap = (
                video_records["true_high"][f"shuffled_{branch}_mean"]
                - video_records["false_high"][f"shuffled_{branch}_mean"]
            )
            paired[f"{video_id}_{branch}"] = {
                "original_true_minus_false_gap": original_gap,
                "shuffled_true_minus_false_gap": shuffled_gap,
                "gap_change": shuffled_gap - original_gap,
            }
            print(
                f"{video_id} | {branch} true-minus-false gap: "
                f"{original_gap:+.4f} -> {shuffled_gap:+.4f} "
                f"(delta {shuffled_gap - original_gap:+.4f})",
                flush=True,
            )
    summary = {
        "model_path": str(args.model_path),
        "shuffle_repeats": args.shuffle_repeats,
        "shuffle_scope": "within each 256-snippet model window; features preserved; scores mapped back to original snippet indices",
        "selected_segments": SEGMENTS,
        "paired_true_minus_false_gaps": paired,
        "csv_path": str(output_csv),
    }
    (output_dir / f"temporal_order_shuffle_{video_count}_videos_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Saved:", output_csv)
    print("Saved:", output_dir / f"temporal_order_shuffle_{video_count}_videos_summary.json")


if __name__ == "__main__":
    main()
