import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.tools import process_feat


# Match the original UCF train prompts; its evaluator uses display-case prompts.
TRAIN_LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}
TEST_LABEL_MAP = {name: name for name in TRAIN_LABEL_MAP}


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UCFCNNFeatureDataset(Dataset):
    def __init__(self, feature_root, list_path, visual_length, train=False, normal=None):
        self.feature_root = Path(feature_root)
        self.visual_length = visual_length
        self.train = train
        with Path(list_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if train and normal is not None:
            rows = [row for row in rows if (row["label"] == "Normal") == normal]
        if not rows:
            raise ValueError(f"No matching feature rows in {list_path}")
        for row in rows:
            if row["label"] not in TRAIN_LABEL_MAP:
                raise ValueError(f"Unknown class: {row['label']}")
            path = Path(row["path"])
            if path.is_absolute():
                row["feature_path"] = path
            else:
                row["feature_path"] = self.feature_root / path
            if not row["feature_path"].is_file():
                raise FileNotFoundError(row["feature_path"])
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = np.load(row["feature_path"])
        if feature.ndim != 2 or feature.shape[1] != 512 or feature.shape[0] == 0:
            raise ValueError(f"Invalid feature shape {feature.shape}: {row['feature_path']}")
        if self.train:
            feature, length = process_feat(feature, self.visual_length)
            return torch.from_numpy(np.asarray(feature, dtype=np.float32)), row["label"], length

        length = feature.shape[0]
        chunks = []
        chunk_lengths = []
        for start in range(0, length, self.visual_length):
            chunk = np.asarray(feature[start:start + self.visual_length], dtype=np.float32)
            chunk_lengths.append(len(chunk))
            if len(chunk) < self.visual_length:
                chunk = np.pad(chunk, ((0, self.visual_length - len(chunk)), (0, 0)))
            chunks.append(chunk)
        return torch.from_numpy(np.stack(chunks)), row["label"], length, row["video_id"], torch.tensor(chunk_lengths)


def aggregate_video_logits(logits, lengths, method="topk"):
    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise ValueError("Expected logits [B,T,1]")
    if method not in ("topk", "max"):
        raise ValueError("Aggregation must be topk or max")
    pooled = []
    for index, length in enumerate(lengths):
        valid = int(length)
        if valid < 1 or valid > logits.shape[1]:
            raise ValueError(f"Invalid sequence length: {valid}")
        scores = logits[index, :valid, 0]
        k = max(1, (valid + 15) // 16) if method == "topk" else 1
        pooled.append(scores.topk(k).values.mean())
    return torch.stack(pooled)


def alignment_mil_loss(logits, labels, lengths):
    selected = []
    for index, length in enumerate(lengths):
        valid = int(length)
        k = valid // 16 + 1
        selected.append(logits[index, :valid].topk(k, dim=0).values.mean(dim=0))
    class_logits = torch.stack(selected)
    labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(labels * F.log_softmax(class_logits, dim=1)).sum(dim=1).mean()


def text_regularization_loss(text_features):
    text = F.normalize(text_features.float(), dim=-1)
    return (text[0:1] @ text[1:].T).abs().mean() * 0.1
