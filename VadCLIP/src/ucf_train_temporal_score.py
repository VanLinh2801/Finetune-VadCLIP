import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_temporal_score
from model_temporal_score import CLIPVADTemporalScore
from utils.dataset_public import UCFDataset
from utils.tools import get_batch_label, get_prompt_text


LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def classifier_mil_loss(logits, labels, lengths, device):
    instance_logits = torch.zeros(0, device=device)
    targets = (1 - labels[:, 0]).to(device)
    scores = torch.sigmoid(logits.float()).squeeze(-1)
    for index in range(scores.shape[0]):
        length = max(1, min(int(lengths[index]), scores.shape[1]))
        k = min(length, int(length / 16 + 1))
        instance_logits = torch.cat(
            [instance_logits, scores[index, :length].topk(k).values.mean().view(1)]
        )
    return F.binary_cross_entropy(instance_logits, targets)


def alignment_mil_loss(logits, labels, lengths, device):
    instance_logits = torch.zeros((0, logits.shape[-1]), device=device)
    targets = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    targets = targets.to(device)
    for index in range(logits.shape[0]):
        length = max(1, min(int(lengths[index]), logits.shape[1]))
        k = min(length, int(length / 16 + 1))
        selected = logits[index, :length].topk(k, largest=True, dim=0).values
        instance_logits = torch.cat(
            [instance_logits, selected.mean(0, keepdim=True)], dim=0
        )
    return -(targets * F.log_softmax(instance_logits.float(), dim=1)).sum(1).mean()


def text_regularization_loss(text_features, device):
    normal = F.normalize(text_features[0].float(), dim=-1)
    loss = torch.zeros((), device=device)
    for index in range(1, text_features.shape[0]):
        abnormal = F.normalize(text_features[index].float(), dim=-1)
        loss = loss + torch.abs(normal @ abnormal)
    return loss / 13 * 1e-1


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
        raise RuntimeError("Temporal score training requires CUDA; select a GPU runtime.")

    checkpoint_dir = Path(args.epoch_checkpoint_dir)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output_model_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, LABEL_MAP, normal=True
    )
    abnormal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, LABEL_MAP, normal=False
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    abnormal_loader = DataLoader(
        abnormal_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    if len(normal_loader) == 0 or len(abnormal_loader) == 0:
        raise ValueError("Train list must contain at least one full normal and abnormal batch.")

    model = CLIPVADTemporalScore(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix,
        args.prompt_postfix, device, temporal_mode=args.temporal_mode,
        temporal_hidden_width=args.temporal_hidden_width,
        max_temporal_delta=args.max_temporal_delta,
    ).to(device)
    # Deliberately no checkpoint-loading path: each variant starts from scratch.
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_rate
    )
    prompt_text = get_prompt_text(LABEL_MAP)
    rows = []

    print(f"Training mode: {args.temporal_mode} (from scratch)")
    print(f"Train videos/chunks: normal={len(normal_dataset)}, abnormal={len(abnormal_dataset)}")
    print(f"Batch composition: {args.batch_size} normal + {args.batch_size} abnormal")
    print(f"LR={args.lr}, epochs={args.max_epoch}, milestones={args.scheduler_milestones}")

    for epoch in range(args.max_epoch):
        model.train()
        total_cls = total_mil = total_text = total_delta_abs = 0.0
        steps = min(len(normal_loader), len(abnormal_loader))
        normal_iter, abnormal_iter = iter(normal_loader), iter(abnormal_loader)

        for _ in range(steps):
            normal_features, normal_labels, normal_lengths = next(normal_iter)
            abnormal_features, abnormal_labels, abnormal_lengths = next(abnormal_iter)
            visual = torch.cat((normal_features, abnormal_features), dim=0).to(
                device, non_blocking=True
            )
            labels_text = list(normal_labels) + list(abnormal_labels)
            lengths = torch.cat((normal_lengths, abnormal_lengths), dim=0).to(device)
            labels = get_batch_label(labels_text, prompt_text, LABEL_MAP).to(device)

            text_features, logits1, logits2, details = model(
                visual, None, prompt_text, lengths, return_temporal_details=True
            )
            loss_cls = classifier_mil_loss(logits1, labels, lengths, device)
            loss_mil = alignment_mil_loss(logits2, labels, lengths, device)
            loss_text = text_regularization_loss(text_features, device)
            loss = loss_cls + loss_mil + loss_text

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_cls += loss_cls.item()
            total_mil += loss_mil.item()
            total_text += loss_text.item()
            valid_delta = details["temporal_delta"].squeeze(-1)[details["valid_mask"]]
            total_delta_abs += valid_delta.detach().abs().mean().item()

        scheduler.step()
        denominator = max(steps, 1)
        row = {
            "epoch": epoch + 1,
            "loss_cls": total_cls / denominator,
            "loss_mil": total_mil / denominator,
            "loss_text_reg": total_text / denominator,
            "temporal_delta_abs_mean": total_delta_abs / denominator,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        print(
            f"epoch {epoch + 1}/{args.max_epoch} | "
            f"loss_cls={row['loss_cls']:.6f} | loss_mil={row['loss_mil']:.6f} | "
            f"loss_text_reg={row['loss_text_reg']:.6f} | "
            f"mean_abs_delta={row['temporal_delta_abs_mean']:.6f}"
        )
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch + 1:03d}.pth")
        torch.save(model.state_dict(), output_path)
        save_history(results_dir / "training_history.csv", rows)

    print(f"Saved final model: {output_path}")
    print(f"Saved epoch checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    train(ucf_option_temporal_score.parser.parse_args())
