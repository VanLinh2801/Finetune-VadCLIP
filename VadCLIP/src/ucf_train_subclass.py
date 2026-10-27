"""Train VadCLIP subclass MIL without test-set checkpoint selection."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

from model_subclass import (
    CLASS_NAMES, CLIPVADSubclass, aggregate_subclass_logits,
    load_subclass_bank, subclass_group_mask, text_regularization,
)
from utils.tools import process_feat


class RelativeTrainDataset(Dataset):
    def __init__(self, csv_path, feature_root, normal_only, visual_length):
        self.feature_root = Path(feature_root)
        self.visual_length = visual_length
        with Path(csv_path).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [row for row in rows if (row["label"] == "Normal") == normal_only]
        if not self.rows:
            raise ValueError("Empty normal or abnormal training split")
        missing = [row["path"] for row in self.rows if not (self.feature_root / row["path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} training features; first: {missing[:3]}")
        self.label_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        unknown = set(row["label"] for row in self.rows) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(unknown)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = np.load(self.feature_root / row["path"])
        if feature.ndim != 2 or feature.shape[1] != 512:
            raise ValueError(f"Unexpected feature shape for {row['path']}: {feature.shape}")
        feature, length = process_feat(feature, self.visual_length)
        return torch.from_numpy(np.asarray(feature, dtype=np.float32)), self.label_index[row["label"]], length


def classifier_mil_loss(logits1, labels, lengths):
    scores = torch.sigmoid(logits1.squeeze(-1))
    instances = []
    for score, length in zip(scores, lengths):
        valid = int(length)
        k = min(valid, valid // 16 + 1)
        instances.append(score[:valid].topk(k).values.mean())
    return F.binary_cross_entropy(torch.stack(instances), (labels != 0).float())


def class_mil_loss(subclass_logits, labels, lengths, mode, temperature, subclass_mask=None):
    class_logits, frame_weights = aggregate_subclass_logits(
        subclass_logits, mode, temperature, subclass_mask, return_weights=True
    )
    class_scores = []
    subclass_usage = []
    for sample_logits, sample_weights, length in zip(class_logits, frame_weights, lengths):
        valid = int(length)
        if not 1 <= valid <= sample_logits.shape[0]:
            raise ValueError(f"Invalid snippet length: {valid}")
        k = min(valid, valid // 16 + 1)
        top_values, top_indices = sample_logits[:valid].topk(k, dim=0)
        class_scores.append(top_values.mean(dim=0))

        # Summarize subclass contributions only at the snippets selected by
        # the class-level Temporal Top-K. This is diagnostic, not a new loss.
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, sample_weights.shape[-1])
        selected_weights = sample_weights[:valid].gather(0, gather_index)
        subclass_usage.append(selected_weights.mean(dim=0))
    class_scores = torch.stack(class_scores)
    subclass_usage = torch.stack(subclass_usage)
    return F.cross_entropy(class_scores, labels), subclass_usage


def make_model(device, visual_length=256):
    return CLIPVADSubclass(14, 512, visual_length, 512, 1, 2, 8, 10, 10, device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subclass-json", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--initialization", choices=["scratch", "checkpoint"], required=True)
    parser.add_argument("--pretrained-model-path")
    parser.add_argument("--aggregation", choices=["max", "soft"], required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke test only: limit steps per epoch")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select a GPU runtime and verify CUDA-enabled PyTorch before training.")
    if args.temperature <= 0 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid temperature, batch size, or number of workers")
    if args.initialization == "checkpoint" and not args.pretrained_model_path:
        raise ValueError("Checkpoint initialization requires --pretrained-model-path")
    epochs = args.max_epoch if args.max_epoch is not None else (10 if args.initialization == "scratch" else 3)
    lr = args.lr if args.lr is not None else 2e-5
    if epochs < 1 or lr <= 0 or args.max_steps < 0:
        raise ValueError("Epochs and learning rate must be positive; max-steps cannot be negative")
    milestones = [4, 8] if args.initialization == "scratch" else [2]

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    groups = load_subclass_bank(args.subclass_json)
    group_mask = subclass_group_mask(groups, device)
    normal = RelativeTrainDataset(args.train_list, args.feature_root, True, args.visual_length)
    abnormal = RelativeTrainDataset(args.train_list, args.feature_root, False, args.visual_length)
    loader_args = dict(batch_size=args.batch_size, shuffle=True, drop_last=True,
                       num_workers=args.num_workers, pin_memory=True)
    normal_loader = DataLoader(normal, **loader_args)
    abnormal_loader = DataLoader(abnormal, **loader_args)
    model = make_model(device, args.visual_length)
    if args.initialization == "checkpoint":
        state = torch.load(args.pretrained_model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print("Loaded self-trained baseline checkpoint:", args.pretrained_model_path, flush=True)
    model.to(device)
    for parameter in model.clipmodel.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    config = vars(args) | {"resolved_epochs": epochs, "resolved_lr": lr, "milestones": milestones,
                           "normal_chunks": len(normal), "abnormal_chunks": len(abnormal),
                           "aggregation_order": "subclass_then_temporal_topk",
                           "subclass_counts": [len(group) for group in groups],
                           "max_subclasses_per_class": int(group_mask.shape[1]),
                           "subclass_sha256": hashlib.sha256(Path(args.subclass_json).read_bytes()).hexdigest()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Config:", config, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        selections = np.zeros((len(CLASS_NAMES), group_mask.shape[1]), dtype=np.int64)
        steps = 0
        for normal_batch, abnormal_batch in zip(normal_loader, abnormal_loader):
            features = torch.cat([normal_batch[0], abnormal_batch[0]]).to(device, non_blocking=True)
            labels = torch.cat([normal_batch[1], abnormal_batch[1]]).to(device, non_blocking=True)
            lengths = torch.cat([normal_batch[2], abnormal_batch[2]]).to(device, non_blocking=True)
            class_text, logits1, subclass_logits = model(features, None, groups, lengths)
            loss_cls = classifier_mil_loss(logits1, labels, lengths)
            loss_mil, subclass_scores = class_mil_loss(
                subclass_logits, labels, lengths, args.aggregation, args.temperature, group_mask
            )
            loss_text = text_regularization(class_text)
            loss = loss_cls + loss_mil + loss_text
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {steps + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += [loss_cls.item(), loss_mil.item(), loss_text.item(), loss.item()]
            best = subclass_scores.detach().argmax(dim=-1)
            for label, index in zip(labels.cpu().tolist(), best[torch.arange(len(labels), device=device), labels].cpu().tolist()):
                selections[label, index] += 1
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        scheduler.step()
        checkpoint = output_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), checkpoint)
        summary = {"epoch": epoch, "steps": steps, "loss_cls": totals[0] / steps,
                   "loss_mil": totals[1] / steps, "loss_text_reg": totals[2] / steps,
                   "loss_total": totals[3] / steps, "lr_next_epoch": scheduler.get_last_lr()[0],
                   "subclass_top1_train_counts": selections.tolist()}
        (output_dir / f"epoch_{epoch:03d}_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Epoch summary:", summary, "checkpoint:", checkpoint, flush=True)
    print("Final fixed-epoch checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    main()
