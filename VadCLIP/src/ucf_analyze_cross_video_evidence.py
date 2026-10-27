import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import CLIPVAD
from ucf_evaluate import UCFRelativeTestDataset
from utils import tools
from utils.tools import get_batch_mask


FRAMES_PER_SNIPPET = 16
STAGES = ("raw_clip", "classifier_input")
QUERY_GROUPS = ("true_high", "false_high", "missed_anomaly", "true_low")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether cross-video neighbors distinguish reliable anomaly evidence."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-ratio", default=0.15, type=float)
    parser.add_argument("--high-snippets-per-chunk", default=4, type=int)
    parser.add_argument("--uniform-snippets-per-chunk", default=4, type=int)
    parser.add_argument("--max-bank-snippets-per-video", default=40, type=int)
    parser.add_argument("--max-queries-per-class-group", default=100, type=int)
    parser.add_argument("--neighbors", default=20, type=int)
    parser.add_argument("--centroid-neighbors", default=200, type=int)
    parser.add_argument("--num-clusters", default=64, type=int)
    parser.add_argument("--train-batch-size", default=16, type=int)
    parser.add_argument("--similarity-batch-size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
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


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(roc_auc_score(labels, scores) * 100)


def safe_ap(labels, scores):
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return np.nan
    return float(average_precision_score(labels, scores) * 100)


def entropy(labels):
    counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
    if counts.size <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum() / np.log(len(counts)))


def describe(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def load_state_dict(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    return payload


def build_model(args, device):
    model = CLIPVAD(
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
    model.load_state_dict(load_state_dict(args.model_path, device), strict=True)
    return model.to(device).eval()


class RelativeTrainDataset(Dataset):
    def __init__(self, train_list, feature_root, visual_length):
        self.feature_root = Path(feature_root)
        self.visual_length = int(visual_length)
        with Path(train_list).open("r", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        missing = [row["path"] for row in self.rows if not self.resolve_path(row["path"]).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} train features. First: {missing[:5]}")

    def resolve_path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.feature_root / path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = np.load(self.resolve_path(row["path"]))
        feature, length = tools.process_feat(feature, self.visual_length)
        video_id = row.get("video_id") or Path(row["path"]).stem.split("__")[0]
        return torch.tensor(feature), int(length), str(video_id), row["label"], row["path"]


def evenly_spaced_indices(length, count):
    count = min(max(1, int(count)), int(length))
    return np.unique(np.linspace(0, int(length) - 1, count, dtype=np.int64))


def select_candidate_indices(scores, length, high_count, uniform_count):
    high_count = min(max(1, int(high_count)), int(length))
    high = np.argsort(-scores[:length], kind="stable")[:high_count]
    uniform = evenly_spaced_indices(length, uniform_count)
    indices = np.unique(np.concatenate([high, uniform])).astype(np.int64)
    high_set = set(high.tolist())
    source = np.asarray(["high" if index in high_set else "uniform" for index in indices], dtype=object)
    return indices, source


def collect_train_candidates(model, dataloader, args, device):
    per_video = defaultdict(list)
    with torch.no_grad():
        for batch_index, item in enumerate(dataloader):
            visual, lengths, video_ids, class_names, paths = item
            visual = visual.to(device, non_blocking=True)
            lengths = lengths.to(torch.int64)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            final_visual = model.encode_video(visual, padding_mask, lengths)
            classifier_input = final_visual + model.mlp2(final_visual)
            scores = torch.sigmoid(model.classifier(classifier_input).squeeze(-1))
            for item_index, length in enumerate(lengths.tolist()):
                indices, sources = select_candidate_indices(
                    scores[item_index].detach().cpu().numpy(),
                    length,
                    args.high_snippets_per_chunk,
                    args.uniform_snippets_per_chunk,
                )
                raw = visual[item_index, indices].detach().cpu().to(torch.float16).numpy()
                classifier = classifier_input[item_index, indices].detach().cpu().to(torch.float16).numpy()
                current_scores = scores[item_index, indices].detach().cpu().numpy()
                for position, snippet_index in enumerate(indices):
                    per_video[str(video_ids[item_index])].append(
                        {
                            "video_id": str(video_ids[item_index]),
                            "class_name": str(class_names[item_index]),
                            "chunk_path": str(paths[item_index]),
                            "snippet_index": int(snippet_index),
                            "candidate_source": str(sources[position]),
                            "classifier_score": float(current_scores[position]),
                            "raw_clip": raw[position],
                            "classifier_input": classifier[position],
                        }
                    )
            if (batch_index + 1) % 100 == 0:
                print(f"  Train batches: {batch_index + 1}/{len(dataloader)}")

    selected = []
    rng = np.random.default_rng(args.seed)
    cap = args.max_bank_snippets_per_video
    for video_id in sorted(per_video):
        records = per_video[video_id]
        high = sorted(
            (record for record in records if record["candidate_source"] == "high"),
            key=lambda record: record["classifier_score"],
            reverse=True,
        )[: max(1, cap // 2)]
        remaining = cap - len(high)
        uniform = [record for record in records if record["candidate_source"] == "uniform"]
        if len(uniform) > remaining:
            chosen = np.sort(rng.choice(len(uniform), remaining, replace=False))
            uniform = [uniform[index] for index in chosen]
        selected.extend(high + uniform[:remaining])

    metadata = {
        key: np.asarray([record[key] for record in selected], dtype=object)
        for key in ("video_id", "class_name", "chunk_path", "candidate_source")
    }
    metadata["snippet_index"] = np.asarray([record["snippet_index"] for record in selected], dtype=np.int64)
    metadata["classifier_score"] = np.asarray([record["classifier_score"] for record in selected], dtype=np.float32)
    features = {
        stage: normalize(np.stack([record[stage] for record in selected]).astype(np.float32))
        for stage in STAGES
    }
    return metadata, features


def make_lengths(length, visual_length):
    remaining = int(length)
    values = []
    while remaining > visual_length:
        values.append(visual_length)
        remaining -= visual_length
    values.append(remaining)
    return torch.tensor(values, dtype=torch.int64)


def topk_mask(scores, ratio):
    count = min(len(scores), max(1, int(math.ceil(len(scores) * ratio))))
    indices = np.argsort(-np.asarray(scores), kind="stable")[:count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[indices] = True
    return mask


def snippet_ground_truth(gt_frames, snippet_count):
    expected = snippet_count * FRAMES_PER_SNIPPET
    values = np.asarray(gt_frames[:expected], dtype=np.int64)
    if len(values) < expected:
        values = np.pad(values, (0, expected - len(values)))
    reshaped = values.reshape(snippet_count, FRAMES_PER_SNIPPET)
    return (reshaped.max(axis=1) > 0).astype(np.int64), reshaped.mean(axis=1).astype(np.float32)


def assign_groups(gt_any, high_mask):
    groups = np.full(len(gt_any), "true_low", dtype=object)
    groups[(gt_any == 1) & high_mask] = "true_high"
    groups[(gt_any == 0) & high_mask] = "false_high"
    groups[(gt_any == 1) & ~high_mask] = "missed_anomaly"
    return groups


def collect_test_queries(model, dataloader, gt, args, device):
    records = []
    gt_cursor = 0
    with torch.no_grad():
        for batch_index, item in enumerate(dataloader):
            visual = item[0].squeeze(0)
            class_name = str(item[1][0] if isinstance(item[1], (list, tuple)) else item[1])
            video_id = str(item[3][0] if isinstance(item[3], (list, tuple)) else item[3])
            snippet_count = int(item[2])
            if snippet_count < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = make_lengths(snippet_count, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            final_visual = model.encode_video(visual, padding_mask, lengths)
            classifier_input = final_visual + model.mlp2(final_visual)
            logits = model.classifier(classifier_input)
            valid_raw = visual.reshape(-1, visual.shape[-1])[:snippet_count]
            valid_classifier = classifier_input.reshape(-1, classifier_input.shape[-1])[:snippet_count]
            scores = torch.sigmoid(logits.reshape(-1)[:snippet_count]).cpu().numpy()

            frame_count = snippet_count * FRAMES_PER_SNIPPET
            gt_any, gt_fraction = snippet_ground_truth(gt[gt_cursor : gt_cursor + frame_count], snippet_count)
            gt_cursor += frame_count
            groups = assign_groups(gt_any, topk_mask(scores, args.top_k_ratio))
            raw = valid_raw.cpu().to(torch.float16).numpy()
            classifier = valid_classifier.cpu().to(torch.float16).numpy()
            for index in range(snippet_count):
                records.append(
                    {
                        "video_id": video_id,
                        "class_name": class_name,
                        "snippet_index": index,
                        "group": str(groups[index]),
                        "gt_any": int(gt_any[index]),
                        "gt_fraction": float(gt_fraction[index]),
                        "classifier_score": float(scores[index]),
                        "raw_clip": raw[index],
                        "classifier_input": classifier[index],
                    }
                )
            if (batch_index + 1) % 50 == 0:
                print(f"  Test videos: {batch_index + 1}/{len(dataloader)}")
    if gt_cursor != len(gt):
        raise ValueError(f"Consumed {gt_cursor} GT frames, but GT contains {len(gt)}")
    return records


def sample_test_queries(records, max_per_class_group, seed):
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        if record["class_name"] == "Normal" and record["group"] != "false_high":
            continue
        grouped[(record["class_name"], record["group"])].append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted(grouped):
        indices = np.asarray(grouped[key], dtype=np.int64)
        if len(indices) > max_per_class_group:
            indices = rng.choice(indices, max_per_class_group, replace=False)
        selected.extend(indices.tolist())
    return [records[index] for index in sorted(selected)]


def retrieve_neighbors(query_features, query_video_ids, bank_features, bank_video_ids, neighbors, batch_size, device):
    neighbors = min(int(neighbors), len(bank_features))
    search_count = min(len(bank_features), neighbors + 64)
    bank_tensor = torch.from_numpy(bank_features).to(device)
    all_indices = []
    all_values = []
    for start in range(0, len(query_features), batch_size):
        query_tensor = torch.from_numpy(query_features[start : start + batch_size]).to(device)
        values, indices = torch.topk(query_tensor @ bank_tensor.T, search_count, dim=1)
        values = values.cpu().numpy()
        indices = indices.cpu().numpy()
        for row, query_video_id in enumerate(query_video_ids[start : start + batch_size]):
            keep = bank_video_ids[indices[row]] != query_video_id
            kept_indices = indices[row][keep][:neighbors]
            kept_values = values[row][keep][:neighbors]
            if len(kept_indices) < neighbors:
                raise ValueError("Not enough cross-video neighbors after excluding the query video")
            all_indices.append(kept_indices)
            all_values.append(kept_values)
    return np.stack(all_indices), np.stack(all_values)


def neighbor_metrics(query_records, neighbor_indices, neighbor_values, bank_metadata, stage):
    rows = []
    bank_classes = bank_metadata["class_name"]
    bank_videos = bank_metadata["video_id"]
    for query, indices, similarities in zip(query_records, neighbor_indices, neighbor_values):
        labels = bank_classes[indices]
        same_ratio = float(np.mean(labels == query["class_name"]))
        normal_ratio = float(np.mean(labels == "Normal"))
        other_labels = labels[(labels != query["class_name"]) & (labels != "Normal")]
        other_counts = Counter(other_labels.tolist())
        dominant_other, dominant_other_count = other_counts.most_common(1)[0] if other_counts else ("", 0)
        dominant_other_ratio = dominant_other_count / len(labels)
        rows.append(
            {
                "stage": stage,
                "video_id": query["video_id"],
                "class_name": query["class_name"],
                "snippet_index": query["snippet_index"],
                "group": query["group"],
                "gt_fraction": query["gt_fraction"],
                "classifier_score": query["classifier_score"],
                "neighbor_similarity_mean": float(similarities.mean()),
                "neighbor_similarity_top1": float(similarities[0]),
                "same_class_ratio": same_ratio,
                "normal_ratio": normal_ratio,
                "other_anomaly_ratio": float(np.mean((labels != query["class_name"]) & (labels != "Normal"))),
                "dominant_other_class": dominant_other,
                "dominant_other_ratio": float(dominant_other_ratio),
                "class_entropy": entropy(labels.tolist()),
                "unique_neighbor_videos": int(len(set(bank_videos[indices].tolist()))),
                "evidence_quality": float(same_ratio - max(normal_ratio, dominant_other_ratio)),
                "neighbor_indices": " ".join(map(str, indices.tolist())),
            }
        )
    return rows


def summarize_retrieval(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["class_name"], row["group"])].append(row)
    output = []
    metrics = (
        "neighbor_similarity_mean",
        "same_class_ratio",
        "normal_ratio",
        "other_anomaly_ratio",
        "class_entropy",
        "unique_neighbor_videos",
        "evidence_quality",
    )
    for (stage, class_name, group), current in sorted(grouped.items()):
        row = {"stage": stage, "class_name": class_name, "group": group, "count": len(current)}
        for metric in metrics:
            values = np.asarray([item[metric] for item in current], dtype=np.float32)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        output.append(row)
    return output


def summarize_neighbor_classes(rows, bank_metadata):
    grouped = defaultdict(Counter)
    query_counts = Counter()
    bank_classes = bank_metadata["class_name"]
    for row in rows:
        key = (row["stage"], row["class_name"], row["group"])
        indices = np.fromstring(row["neighbor_indices"], sep=" ", dtype=np.int64)
        grouped[key].update(bank_classes[indices].tolist())
        query_counts[key] += 1

    output = []
    for (stage, query_class, group), counts in sorted(grouped.items()):
        total = sum(counts.values())
        for neighbor_class, count in counts.most_common():
            output.append(
                {
                    "stage": stage,
                    "query_class": query_class,
                    "query_group": group,
                    "query_count": query_counts[(stage, query_class, group)],
                    "neighbor_class": neighbor_class,
                    "neighbor_count": count,
                    "neighbor_ratio": float(count / total),
                }
            )
    return output


def separation_metrics(rows):
    output = []
    comparisons = (("true_high", "false_high"), ("missed_anomaly", "false_high"))
    for stage in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage and row["class_name"] != "Normal"]
        scopes = [("all_abnormal", stage_rows)]
        scopes.extend(
            (class_name, [row for row in stage_rows if row["class_name"] == class_name])
            for class_name in sorted({row["class_name"] for row in stage_rows})
        )
        for scope, scope_rows in scopes:
            for positive_group, negative_group in comparisons:
                current = [row for row in scope_rows if row["group"] in (positive_group, negative_group)]
                labels = np.asarray([row["group"] == positive_group for row in current], dtype=np.int64)
                for metric in ("evidence_quality", "same_class_ratio", "neighbor_similarity_mean"):
                    scores = np.asarray([row[metric] for row in current], dtype=np.float32)
                    positive = scores[labels == 1]
                    negative = scores[labels == 0]
                    output.append(
                        {
                            "stage": stage,
                            "scope": scope,
                            "comparison": f"{positive_group}_vs_{negative_group}",
                            "metric": metric,
                            "positive_count": int((labels == 1).sum()),
                            "negative_count": int((labels == 0).sum()),
                            "auc": safe_auc(labels, scores),
                            "ap": safe_ap(labels, scores),
                            "positive_mean": float(positive.mean()) if positive.size else np.nan,
                            "negative_mean": float(negative.mean()) if negative.size else np.nan,
                            "mean_gap": float(positive.mean() - negative.mean()) if positive.size and negative.size else np.nan,
                        }
                    )
    return output


def centroid_neighbor_overlap(query_records, query_features, bank_features, bank_metadata, args, device, stage):
    rows = []
    classes = sorted({record["class_name"] for record in query_records if record["class_name"] != "Normal"})
    for class_name in classes:
        group_indices = {
            group: np.asarray(
                [index for index, record in enumerate(query_records) if record["class_name"] == class_name and record["group"] == group],
                dtype=np.int64,
            )
            for group in ("true_high", "false_high", "missed_anomaly")
        }
        centroids = {}
        neighbor_sets = {}
        for group, indices in group_indices.items():
            if len(indices) == 0:
                continue
            centroid = normalize(query_features[indices].mean(axis=0, keepdims=True))[0]
            centroids[group] = centroid
            neighbor_indices, _ = retrieve_neighbors(
                centroid[None, :],
                np.asarray([f"__centroid_{class_name}_{group}"]),
                bank_features,
                bank_metadata["video_id"],
                args.centroid_neighbors,
                1,
                device,
            )
            neighbor_sets[group] = set(neighbor_indices[0].tolist())
        for group_a, group_b in (("true_high", "false_high"), ("missed_anomaly", "false_high"), ("true_high", "missed_anomaly")):
            if group_a not in centroids or group_b not in centroids:
                continue
            intersection = neighbor_sets[group_a] & neighbor_sets[group_b]
            union = neighbor_sets[group_a] | neighbor_sets[group_b]
            rows.append(
                {
                    "stage": stage,
                    "class_name": class_name,
                    "group_a": group_a,
                    "group_b": group_b,
                    "group_a_count": len(group_indices[group_a]),
                    "group_b_count": len(group_indices[group_b]),
                    "centroid_cosine": float(centroids[group_a] @ centroids[group_b]),
                    "neighbor_intersection": len(intersection),
                    "neighbor_jaccard": float(len(intersection) / max(1, len(union))),
                }
            )
    return rows


def cluster_diagnostics(bank_features, bank_metadata, query_features, query_records, num_clusters, seed, stage):
    cluster_count = min(int(num_clusters), len(bank_features))
    estimator = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        batch_size=2048,
        n_init=1,
        max_iter=100,
    )
    train_clusters = estimator.fit_predict(bank_features)
    query_clusters = estimator.predict(query_features)
    cluster_rows = []
    for cluster_id in range(cluster_count):
        mask = train_clusters == cluster_id
        labels = bank_metadata["class_name"][mask]
        videos = bank_metadata["video_id"][mask]
        counts = Counter(labels.tolist())
        dominant_class, dominant_count = counts.most_common(1)[0]
        cluster_rows.append(
            {
                "stage": stage,
                "cluster_id": cluster_id,
                "count": int(mask.sum()),
                "unique_videos": len(set(videos.tolist())),
                "normal_ratio": float(np.mean(labels == "Normal")),
                "dominant_class": dominant_class,
                "dominant_class_ratio": float(dominant_count / len(labels)),
                "class_entropy": entropy(labels.tolist()),
            }
        )

    distribution_rows = []
    grouped_counts = defaultdict(Counter)
    for record, cluster_id in zip(query_records, query_clusters):
        grouped_counts[(record["class_name"], record["group"])][int(cluster_id)] += 1
    for (class_name, group), counts in sorted(grouped_counts.items()):
        total = sum(counts.values())
        for cluster_id, count in counts.items():
            distribution_rows.append(
                {
                    "stage": stage,
                    "class_name": class_name,
                    "group": group,
                    "cluster_id": cluster_id,
                    "count": count,
                    "ratio": float(count / total),
                }
            )
    return cluster_rows, distribution_rows


def print_summary(bank_metadata, query_records, group_rows, separation_rows, overlap_rows):
    print("\n" + "=" * 88)
    print("CROSS-VIDEO CLASS EVIDENCE DIAGNOSTICS")
    print(f"Train bank: {len(bank_metadata['video_id'])} snippets | {len(set(bank_metadata['video_id']))} videos")
    print(f"Test queries: {len(query_records)}")
    for stage in STAGES:
        print("\n" + "-" * 88)
        print(f"Feature stage: {stage}")
        current = [row for row in group_rows if row["stage"] == stage and row["class_name"] != "Normal"]
        for group in ("true_high", "false_high", "missed_anomaly"):
            rows = [row for row in current if row["group"] == group]
            if not rows:
                continue
            count = sum(row["count"] for row in rows)
            same = np.average([row["same_class_ratio_mean"] for row in rows], weights=[row["count"] for row in rows])
            normal = np.average([row["normal_ratio_mean"] for row in rows], weights=[row["count"] for row in rows])
            quality = np.average([row["evidence_quality_mean"] for row in rows], weights=[row["count"] for row in rows])
            print(f"  {group:16s} n={count:5d} | same-class={same:.3f} | Normal={normal:.3f} | quality={quality:+.3f}")
        for row in separation_rows:
            if row["stage"] == stage and row["scope"] == "all_abnormal" and row["metric"] == "evidence_quality":
                print(f"  {row['comparison']:38s} AUC={row['auc']:.2f} | gap={row['mean_gap']:+.4f}")
        per_class = [
            row for row in separation_rows
            if row["stage"] == stage
            and row["scope"] != "all_abnormal"
            and row["comparison"] == "true_high_vs_false_high"
            and row["metric"] == "evidence_quality"
            and np.isfinite(row["auc"])
        ]
        if per_class:
            ordered = sorted(per_class, key=lambda row: row["auc"])
            print("  Weakest class separation:", ", ".join(f"{row['scope']}={row['auc']:.1f}" for row in ordered[:5]))
            print("  Strongest class separation:", ", ".join(f"{row['scope']}={row['auc']:.1f}" for row in ordered[-5:][::-1]))
        overlaps = [row for row in overlap_rows if row["stage"] == stage and row["group_a"] == "true_high" and row["group_b"] == "false_high"]
        if overlaps:
            print(
                "  True-high/false-high neighborhood: "
                f"mean centroid cosine={np.mean([row['centroid_cosine'] for row in overlaps]):.3f} | "
                f"mean Jaccard={np.mean([row['neighbor_jaccard'] for row in overlaps]):.3f}"
            )


def build_decision_summary(separation_rows, overlap_rows):
    stages = {}
    for stage in STAGES:
        true_false = next(
            row for row in separation_rows
            if row["stage"] == stage
            and row["scope"] == "all_abnormal"
            and row["comparison"] == "true_high_vs_false_high"
            and row["metric"] == "evidence_quality"
        )
        missed_false = next(
            row for row in separation_rows
            if row["stage"] == stage
            and row["scope"] == "all_abnormal"
            and row["comparison"] == "missed_anomaly_vs_false_high"
            and row["metric"] == "evidence_quality"
        )
        per_class = [
            row for row in separation_rows
            if row["stage"] == stage
            and row["scope"] != "all_abnormal"
            and row["comparison"] == "true_high_vs_false_high"
            and row["metric"] == "evidence_quality"
            and np.isfinite(row["auc"])
        ]
        class_pass_count = sum(row["auc"] >= 55 and row["mean_gap"] > 0 for row in per_class)
        class_pass_ratio = class_pass_count / max(1, len(per_class))
        overlaps = [
            row for row in overlap_rows
            if row["stage"] == stage and row["group_a"] == "true_high" and row["group_b"] == "false_high"
        ]
        mean_jaccard = float(np.mean([row["neighbor_jaccard"] for row in overlaps])) if overlaps else np.nan
        passes = (
            true_false["auc"] >= 60
            and missed_false["auc"] >= 55
            and true_false["mean_gap"] > 0
            and missed_false["mean_gap"] > 0
            and class_pass_ratio >= 0.6
        )
        stages[stage] = {
            "true_high_vs_false_high_evidence_quality_auc": true_false["auc"],
            "missed_anomaly_vs_false_high_evidence_quality_auc": missed_false["auc"],
            "true_high_vs_false_high_mean_gap": true_false["mean_gap"],
            "missed_anomaly_vs_false_high_mean_gap": missed_false["mean_gap"],
            "true_high_false_high_mean_neighbor_jaccard": mean_jaccard,
            "classes_with_valid_true_high_false_high_test": len(per_class),
            "classes_passing_true_high_false_high_gate": class_pass_count,
            "class_pass_ratio": class_pass_ratio,
            "passes_initial_cross_video_gate": bool(passes),
        }
    recommended = bool(stages["classifier_input"]["passes_initial_cross_video_gate"])
    return {
        "decision": "continue_to_training_experiment" if recommended else "do_not_use_cross_video_evidence_for_training_yet",
        "reason": (
            "Classifier-input neighborhoods separate both true-high and missed anomaly from false-high."
            if recommended
            else "Cross-video evidence quality does not reliably separate false-high from both useful evidence groups."
        ),
        "stages": stages,
    }


def main():
    args = parse_args()
    if not 0 < args.top_k_ratio < 1:
        raise ValueError("--top-k-ratio must be in (0, 1)")
    if min(args.high_snippets_per_chunk, args.uniform_snippets_per_chunk, args.max_bank_snippets_per_video, args.neighbors) <= 0:
        raise ValueError("Candidate and neighbor limits must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Cross-video class evidence analysis only; no training and no model updates")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Top-k ratio used to define test groups: {args.top_k_ratio:.2f}")
    model = build_model(args, device)

    train_dataset = RelativeTrainDataset(args.train_list, args.feature_root, args.visual_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    print(f"\nBuilding train bank from {len(train_dataset)} chunks...")
    bank_metadata, bank_features = collect_train_candidates(model, train_loader, args, device)
    print(
        f"Bank candidates retained: {len(bank_metadata['video_id'])} | "
        f"videos={len(set(bank_metadata['video_id']))} | classes={len(set(bank_metadata['class_name']))}"
    )

    test_dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    gt = np.load(args.gt_path)
    print(f"\nRunning test inference for diagnostic labels on {len(test_dataset)} videos...")
    all_test_records = collect_test_queries(model, test_loader, gt, args, device)
    query_records = sample_test_queries(all_test_records, args.max_queries_per_class_group, args.seed)
    print(f"Balanced diagnostic queries retained: {len(query_records)}")

    query_metadata = {
        key: np.asarray([record[key] for record in query_records], dtype=object)
        for key in ("video_id", "class_name", "group")
    }
    query_features = {
        stage: normalize(np.stack([record[stage] for record in query_records]).astype(np.float32))
        for stage in STAGES
    }

    retrieval_rows = []
    overlap_rows = []
    cluster_rows = []
    cluster_distribution_rows = []
    for stage_index, stage in enumerate(STAGES):
        print(f"\nRetrieving cross-video neighbors in {stage} space...")
        neighbor_indices, neighbor_values = retrieve_neighbors(
            query_features[stage],
            query_metadata["video_id"],
            bank_features[stage],
            bank_metadata["video_id"],
            args.neighbors,
            args.similarity_batch_size,
            device,
        )
        retrieval_rows.extend(
            neighbor_metrics(query_records, neighbor_indices, neighbor_values, bank_metadata, stage)
        )
        overlap_rows.extend(
            centroid_neighbor_overlap(
                query_records,
                query_features[stage],
                bank_features[stage],
                bank_metadata,
                args,
                device,
                stage,
            )
        )
        print(f"Clustering train candidates in {stage} space...")
        current_clusters, current_distribution = cluster_diagnostics(
            bank_features[stage],
            bank_metadata,
            query_features[stage],
            query_records,
            args.num_clusters,
            args.seed + stage_index,
            stage,
        )
        cluster_rows.extend(current_clusters)
        cluster_distribution_rows.extend(current_distribution)

    group_rows = summarize_retrieval(retrieval_rows)
    class_confusion_rows = summarize_neighbor_classes(retrieval_rows, bank_metadata)
    separation_rows = separation_metrics(retrieval_rows)
    decision = build_decision_summary(separation_rows, overlap_rows)
    print_summary(bank_metadata, query_records, group_rows, separation_rows, overlap_rows)
    print("\nDecision:", decision["decision"])
    print("Reason:", decision["reason"])

    bank_rows = []
    for index in range(len(bank_metadata["video_id"])):
        bank_rows.append(
            {
                "bank_index": index,
                "video_id": bank_metadata["video_id"][index],
                "class_name": bank_metadata["class_name"][index],
                "chunk_path": bank_metadata["chunk_path"][index],
                "snippet_index": bank_metadata["snippet_index"][index],
                "candidate_source": bank_metadata["candidate_source"][index],
                "classifier_score": bank_metadata["classifier_score"][index],
            }
        )
    write_csv(output_dir / "train_bank_index.csv", bank_rows)
    write_csv(output_dir / "query_neighbor_metrics.csv", retrieval_rows)
    write_csv(output_dir / "group_retrieval_summary.csv", group_rows)
    write_csv(output_dir / "neighbor_class_distribution.csv", class_confusion_rows)
    write_csv(output_dir / "cross_video_separation.csv", separation_rows)
    write_csv(output_dir / "centroid_neighbor_overlap.csv", overlap_rows)
    write_csv(output_dir / "train_cluster_summary.csv", cluster_rows)
    write_csv(output_dir / "query_cluster_distribution.csv", cluster_distribution_rows)
    with (output_dir / "decision_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": str(args.model_path),
                "feature_root": str(args.feature_root),
                "train_chunks": len(train_dataset),
                "train_bank_snippets": len(bank_metadata["video_id"]),
                "train_bank_videos": len(set(bank_metadata["video_id"])),
                "test_videos": len(test_dataset),
                "balanced_test_queries": len(query_records),
                "top_k_ratio": args.top_k_ratio,
                "neighbors": args.neighbors,
                "candidate_policy": {
                    "high_snippets_per_chunk": args.high_snippets_per_chunk,
                    "uniform_snippets_per_chunk": args.uniform_snippets_per_chunk,
                    "max_bank_snippets_per_video": args.max_bank_snippets_per_video,
                },
                "test_ground_truth_usage": "Diagnostic grouping only; never used to build the train bank.",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    np.savez_compressed(
        output_dir / "cross_video_feature_bank.npz",
        raw_clip=bank_features["raw_clip"].astype(np.float16),
        classifier_input=bank_features["classifier_input"].astype(np.float16),
    )
    print(f"\nSaved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
