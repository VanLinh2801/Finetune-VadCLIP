import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

import ucf_option_video_context
from model_video_context import CLIPVADVideoContext
from utils import tools


LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def crop_index_from_path(path):
    match = re.search(r"__(\d+)$", Path(path).stem)
    if not match:
        raise ValueError(f"Feature path has no numeric crop suffix __i: {path}")
    return int(match.group(1))


class CropViewVideoDataset(Dataset):
    """Group spatial crop/flip views by video; never treat view index as time."""

    def __init__(self, csv_path, feature_root, visual_length, normal):
        self.feature_root = Path(feature_root)
        self.visual_length = visual_length
        self.videos = {}
        grouped = defaultdict(list)
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                video_id = row.get("video_id") or Path(row["path"]).stem.rsplit("__", 1)[0]
                if (row["label"] == "Normal") != normal:
                    continue
                grouped[video_id].append(row)

        expected = set(range(10))
        for video_id, rows in grouped.items():
            labels = {row["label"] for row in rows}
            if len(labels) != 1:
                raise ValueError(f"Conflicting labels for video_id={video_id}: {labels}")
            indices = [crop_index_from_path(row["path"]) for row in rows]
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate spatial crop index for video_id={video_id}")
            if set(indices) != expected:
                raise ValueError(
                    f"Expected crop views 0..9 for video_id={video_id}, got {sorted(indices)}"
                )
            rows.sort(key=lambda row: crop_index_from_path(row["path"]))
            for row in rows:
                path = Path(row["path"])
                path = path if path.is_absolute() else self.feature_root / path
                if not path.is_file():
                    raise FileNotFoundError(f"Missing feature for {video_id}: {path}")
                row["resolved_path"] = path
            self.videos[video_id] = {"label": rows[0]["label"], "views": rows}

        if not self.videos:
            raise ValueError(f"No {'normal' if normal else 'abnormal'} videos found in {csv_path}")
        self.video_ids = sorted(self.videos)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        record = self.videos[video_id]
        features, lengths = [], []
        for row in record["views"]:
            feature = np.load(row["resolved_path"], mmap_mode="r")
            if feature.ndim != 2 or feature.shape[1] != 512 or feature.shape[0] == 0:
                raise ValueError(f"Invalid feature shape for {row['resolved_path']}: {feature.shape}")
            processed, valid_length = tools.process_feat(
                np.asarray(feature, dtype=np.float32), self.visual_length
            )
            features.append(torch.from_numpy(np.asarray(processed, dtype=np.float32)))
            lengths.append(int(valid_length))
        return {
            "video_id": video_id,
            "label": record["label"],
            "features": features,
            "lengths": lengths,
        }


def collate_videos(videos):
    flat_features = []
    flat_lengths = []
    view_owner = []
    video_labels = []
    video_ids = []
    for video_index, video in enumerate(videos):
        video_ids.append(video["video_id"])
        video_labels.append(video["label"])
        for feature, length in zip(video["features"], video["lengths"]):
            flat_features.append(feature)
            flat_lengths.append(length)
            view_owner.append(video_index)
    return {
        "features": torch.stack(flat_features),
        "lengths": torch.tensor(flat_lengths, dtype=torch.long),
        "view_owner": torch.tensor(view_owner, dtype=torch.long),
        "video_labels": video_labels,
        "video_ids": video_ids,
    }


def combine_video_batches(normal, abnormal):
    video_offset = len(normal["video_labels"])
    return {
        "features": torch.cat((normal["features"], abnormal["features"]), dim=0),
        "lengths": torch.cat((normal["lengths"], abnormal["lengths"]), dim=0),
        "view_owner": torch.cat(
            (normal["view_owner"], abnormal["view_owner"] + video_offset), dim=0
        ),
        "video_labels": normal["video_labels"] + abnormal["video_labels"],
        "video_ids": normal["video_ids"] + abnormal["video_ids"],
    }


def macro_video_mean(view_values, view_owner, video_count):
    per_video = []
    for index in range(video_count):
        selected = view_values[view_owner == index]
        if not selected.numel():
            raise ValueError(f"Video {index} has no crop views in the batch.")
        per_video.append(selected.mean())
    return torch.stack(per_video).mean()


def vadclip_losses_per_view(logits1, logits2, labels, lengths):
    classifier_losses = []
    alignment_losses = []
    scores = torch.sigmoid(logits1.float()).squeeze(-1)
    for index in range(scores.shape[0]):
        valid = max(1, min(int(lengths[index]), scores.shape[1]))
        k = min(valid, int(valid / 16 + 1))
        classifier_score = scores[index, :valid].topk(k).values.mean()
        classifier_target = 1.0 - labels[index, 0]
        classifier_losses.append(
            F.binary_cross_entropy(classifier_score, classifier_target.float())
        )

        class_target = labels[index] / labels[index].sum().clamp_min(1.0)
        class_scores = logits2[index, :valid].topk(k, largest=True, dim=0).values.mean(0)
        alignment_losses.append(
            -(class_target * F.log_softmax(class_scores.float(), dim=0)).sum()
        )
    return torch.stack(classifier_losses), torch.stack(alignment_losses)


def text_regularization_loss(text_features):
    normalized = F.normalize(text_features.float(), dim=-1)
    normal = normalized[0]
    return torch.stack([torch.abs(normal @ normalized[i]) for i in range(1, len(normalized))]).mean() * 0.1


def save_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Video-context training requires a CUDA runtime.")

    output_path = Path(args.output_model_path)
    checkpoint_dir = Path(args.epoch_checkpoint_dir)
    results_dir = Path(args.results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    normal_dataset = CropViewVideoDataset(
        args.train_list, args.feature_root, args.visual_length, normal=True
    )
    abnormal_dataset = CropViewVideoDataset(
        args.train_list, args.feature_root, args.visual_length, normal=False
    )
    loader_options = {
        "batch_size": args.videos_per_class_batch,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": True,
        "collate_fn": collate_videos,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    normal_loader = DataLoader(normal_dataset, shuffle=True, **loader_options)
    abnormal_loader = DataLoader(abnormal_dataset, shuffle=True, **loader_options)
    steps = min(len(normal_loader), len(abnormal_loader))
    if steps == 0:
        raise ValueError("Not enough full video batches for balanced training.")

    model = CLIPVADVideoContext(
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
        context_bin_size=args.context_bin_size,
        context_hidden_width=args.context_hidden_width,
        max_context_delta=args.max_context_delta,
    ).to(device)
    # Every trainable VadCLIP component and the context branch start from scratch.
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr
    )
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_rate
    )
    prompt_text = tools.get_prompt_text(LABEL_MAP)
    history = []

    print("Training from scratch; CLIP image/text encoder remains frozen.")
    print(
        f"Videos: normal={len(normal_dataset)} abnormal={len(abnormal_dataset)} | "
        f"views/video: 10 | videos/batch: {args.videos_per_class_batch} per label"
    )
    print(
        f"Context bins={args.context_bin_size} snippets | GRU hidden={args.context_hidden_width} | "
        f"lambda_video={args.lambda_video} warmup={args.video_loss_warmup_epochs} epochs"
    )
    print(f"LR={args.lr} epochs={args.max_epoch} milestones={args.scheduler_milestones}")

    for epoch in range(args.max_epoch):
        model.train()
        totals = {key: 0.0 for key in ("loss_cls", "loss_mil", "loss_text", "loss_video", "delta_abs")}
        normal_iter, abnormal_iter = iter(normal_loader), iter(abnormal_loader)
        video_lambda = args.lambda_video * min(
            1.0, (epoch + 1) / max(args.video_loss_warmup_epochs, 1)
        )

        for _ in range(steps):
            batch = combine_video_batches(next(normal_iter), next(abnormal_iter))
            visual = batch["features"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            owner = batch["view_owner"].to(device)
            video_count = len(batch["video_labels"])
            labels_text = [batch["video_labels"][int(i)] for i in batch["view_owner"]]
            labels = tools.get_batch_label(labels_text, prompt_text, LABEL_MAP).to(device)
            video_targets = torch.tensor(
                [0.0 if label == "Normal" else 1.0 for label in batch["video_labels"]],
                dtype=torch.float32,
                device=device,
            )
            view_block_counts = [1] * visual.shape[0]
            padding_mask = tools.get_batch_mask(
                lengths.detach().cpu(), args.visual_length
            ).to(device)

            text_features, logits1, logits2, details = model(
                visual,
                padding_mask,
                prompt_text,
                lengths,
                view_block_counts,
                return_context_details=True,
            )
            cls_per_view, mil_per_view = vadclip_losses_per_view(
                logits1, logits2, labels, lengths
            )
            loss_cls = macro_video_mean(cls_per_view, owner, video_count)
            loss_mil = macro_video_mean(mil_per_view, owner, video_count)
            loss_text = text_regularization_loss(text_features)

            video_logits = details["video_logits_per_view"]
            video_targets_per_view = video_targets[owner]
            bce_per_view = F.binary_cross_entropy_with_logits(
                video_logits.float(), video_targets_per_view, reduction="none"
            )
            loss_video = macro_video_mean(bce_per_view, owner, video_count)
            loss = loss_cls + loss_mil + loss_text + video_lambda * loss_video

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            valid_delta = details["context_delta"].squeeze(-1)[details["valid_mask"]]
            totals["loss_cls"] += loss_cls.item()
            totals["loss_mil"] += loss_mil.item()
            totals["loss_text"] += loss_text.item()
            totals["loss_video"] += loss_video.item()
            totals["delta_abs"] += valid_delta.detach().abs().mean().item()

        scheduler.step()
        row = {"epoch": epoch + 1, **{key: value / steps for key, value in totals.items()}}
        row["lambda_video"] = video_lambda
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        print(
            f"epoch {epoch + 1}/{args.max_epoch} | cls={row['loss_cls']:.5f} "
            f"mil={row['loss_mil']:.5f} text={row['loss_text']:.5f} "
            f"video_bce={row['loss_video']:.5f} lambda_video={video_lambda:.4f} "
            f"mean_abs_delta={row['delta_abs']:.6f}"
        )
        state = model.state_dict()
        torch.save(state, checkpoint_dir / f"epoch_{epoch + 1:03d}.pth")
        torch.save(state, output_path)
        save_history(results_dir / "training_history.csv", history)

    print(f"Saved final epoch model: {output_path}")
    print(f"Saved epoch checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    train(ucf_option_video_context.parser.parse_args())
