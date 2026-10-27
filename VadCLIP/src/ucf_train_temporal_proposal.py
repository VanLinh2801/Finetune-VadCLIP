import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_temporal_proposal
from model_temporal_proposal import CLIPVADTemporalProposal
from ucf_test import test
from ucf_train_class_prototype import UCFPrototypeDataset, text_regularization_loss
from utils.tools import get_batch_label, get_prompt_text


LABEL_MAP = {
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_bag_logits(proposal_logits, temperature):
    temperature = max(float(temperature), 1e-4)
    proposal_count = proposal_logits.shape[1]
    return temperature * (
        torch.logsumexp(proposal_logits / temperature, dim=1)
        - math.log(proposal_count)
    )


def proposal_bag_loss(proposal_logits, labels, temperature):
    abnormal_target = (1 - labels[:, 0]).float()
    bag_logits = soft_bag_logits(proposal_logits.float(), temperature)
    return F.binary_cross_entropy_with_logits(bag_logits, abnormal_target), bag_logits


def normal_proposal_loss(proposal_logits, labels, device):
    normal_mask = labels[:, 0] > 0.5
    if not normal_mask.any():
        return torch.zeros((), device=device)
    normal_logits = proposal_logits[normal_mask].float()
    return F.binary_cross_entropy_with_logits(normal_logits, torch.zeros_like(normal_logits))


def proposal_class_loss(proposal_logits, proposal_class_logits, labels, temperature):
    temperature = max(float(temperature), 1e-4)
    weights = F.softmax(proposal_logits.float() / temperature, dim=1)
    video_class_logits = torch.einsum(
        "bm,bmc->bc", weights, proposal_class_logits.float()
    )
    class_target = labels.argmax(dim=1)
    return F.cross_entropy(video_class_logits, class_target), video_class_logits


def proposal_width_loss(widths, proposal_logits, labels, temperature, device):
    abnormal_mask = labels[:, 0] < 0.5
    if not abnormal_mask.any():
        return torch.zeros((), device=device)
    weights = F.softmax(
        proposal_logits[abnormal_mask].float() / max(float(temperature), 1e-4), dim=1
    )
    return (weights * widths[abnormal_mask].float()).sum(dim=1).mean()


def proposal_coverage_loss(
    snippet_probability, labels, lengths, min_coverage, max_coverage, device
):
    abnormal_indices = torch.where(labels[:, 0] < 0.5)[0]
    if abnormal_indices.numel() == 0:
        return torch.zeros((), device=device)
    losses = []
    for index in abnormal_indices:
        valid_length = max(1, min(int(lengths[index]), snippet_probability.shape[1]))
        coverage = snippet_probability[index, :valid_length].float().mean()
        losses.append(
            F.relu(float(min_coverage) - coverage)
            + F.relu(coverage - float(max_coverage))
        )
    return torch.stack(losses).mean()


def proposal_overlap_loss(centers, widths, proposal_logits, labels, device):
    abnormal_mask = labels[:, 0] < 0.5
    if not abnormal_mask.any() or centers.shape[1] < 2:
        return torch.zeros((), device=device)

    centers = centers[abnormal_mask].float()
    widths = widths[abnormal_mask].float()
    confidence = torch.sigmoid(proposal_logits[abnormal_mask].float())
    left = centers - widths / 2
    right = centers + widths / 2
    intersection = (
        torch.minimum(right.unsqueeze(2), right.unsqueeze(1))
        - torch.maximum(left.unsqueeze(2), left.unsqueeze(1))
    ).clamp_min(0)
    union = widths.unsqueeze(2) + widths.unsqueeze(1) - intersection
    iou = intersection / union.clamp_min(1e-6)
    confidence_pair = confidence.unsqueeze(2) * confidence.unsqueeze(1)
    upper_triangle = torch.triu(
        torch.ones_like(iou, dtype=torch.bool), diagonal=1
    )
    weighted_iou = iou * confidence_pair
    return weighted_iou[upper_triangle].mean()


def proposal_boundary_loss(
    boundary_distance, proposal_logits, labels, margin, temperature, device
):
    abnormal_mask = labels[:, 0] < 0.5
    if not abnormal_mask.any():
        return torch.zeros((), device=device)
    weights = F.softmax(
        proposal_logits[abnormal_mask].float() / max(float(temperature), 1e-4), dim=1
    )
    penalties = F.relu(float(margin) - boundary_distance[abnormal_mask].float())
    return (weights * penalties).sum(dim=1).mean()


def proposal_statistics(proposal_outputs, labels):
    confidence = proposal_outputs["proposal_confidence"].detach().float()
    widths = proposal_outputs["widths"].detach().float()
    centers = proposal_outputs["centers"].detach().float()
    abnormal_mask = labels[:, 0] < 0.5
    normal_mask = ~abnormal_mask
    return {
        "proposal_confidence_mean": float(confidence.mean()),
        "normal_confidence_mean": float(confidence[normal_mask].mean())
        if normal_mask.any()
        else np.nan,
        "abnormal_confidence_mean": float(confidence[abnormal_mask].mean())
        if abnormal_mask.any()
        else np.nan,
        "width_mean": float(widths.mean()),
        "width_min": float(widths.min()),
        "width_max": float(widths.max()),
        "center_min": float(centers.min()),
        "center_max": float(centers.max()),
    }


def ensure_output_directories(args):
    for path in [
        Path(args.output_model_path),
        Path(args.checkpoint_path),
        Path(args.save_cur_path),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)


def save_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(model, normal_loader, anomaly_loader, test_loader, args, device):
    ensure_output_directories(args)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    best_auc = -float("inf")
    epochs_without_improvement = 0
    history = []
    num_steps = min(len(normal_loader), len(anomaly_loader))

    print("Temporal proposal training from scratch")
    print("  proposals:", args.num_proposals)
    print(
        "  width range:",
        f"[{args.proposal_min_width:.3f}, {args.proposal_max_width:.3f}]",
        "initial:",
        f"{args.proposal_initial_width:.3f}",
    )
    print("  mask temperature:", args.mask_temperature)
    print("  train steps per epoch:", num_steps)

    for epoch in range(args.max_epoch):
        model.train()
        totals = {
            "loss_total": 0.0,
            "loss_bag": 0.0,
            "loss_normal": 0.0,
            "loss_class": 0.0,
            "loss_width": 0.0,
            "loss_coverage": 0.0,
            "loss_overlap": 0.0,
            "loss_boundary": 0.0,
            "loss_text": 0.0,
        }
        statistic_totals = {}
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for _ in range(num_steps):
            normal_features, normal_labels, normal_lengths, _ = next(normal_iter)
            anomaly_features, anomaly_labels, anomaly_lengths, _ = next(anomaly_iter)
            visual = torch.cat([normal_features, anomaly_features], dim=0).to(
                device, non_blocking=True
            )
            lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            label_names = list(normal_labels) + list(anomaly_labels)
            labels = get_batch_label(label_names, prompt_text, LABEL_MAP).to(device)

            text_features, _, _, proposal_outputs = model(
                visual, None, prompt_text, lengths, return_proposals=True
            )
            proposal_logits = proposal_outputs["proposal_logits"]
            proposal_class_logits = proposal_outputs["proposal_class_logits"]

            loss_bag, _ = proposal_bag_loss(
                proposal_logits, labels, args.bag_temperature
            )
            loss_normal = normal_proposal_loss(proposal_logits, labels, device)
            loss_class, _ = proposal_class_loss(
                proposal_logits,
                proposal_class_logits,
                labels,
                args.class_selection_temperature,
            )
            loss_width = proposal_width_loss(
                proposal_outputs["widths"],
                proposal_logits,
                labels,
                args.class_selection_temperature,
                device,
            )
            loss_coverage = proposal_coverage_loss(
                proposal_outputs["snippet_probability"],
                labels,
                lengths,
                args.proposal_min_coverage,
                args.proposal_max_coverage,
                device,
            )
            loss_overlap = proposal_overlap_loss(
                proposal_outputs["centers"],
                proposal_outputs["widths"],
                proposal_logits,
                labels,
                device,
            )
            loss_boundary = proposal_boundary_loss(
                proposal_outputs["boundary_distance"],
                proposal_logits,
                labels,
                args.boundary_margin,
                args.class_selection_temperature,
                device,
            )
            loss_text = text_regularization_loss(text_features.float(), device).squeeze()
            lambda_boundary = (
                0.0 if epoch < args.boundary_warmup_epochs else args.lambda_boundary
            )
            loss = (
                loss_bag
                + args.lambda_normal * loss_normal
                + args.lambda_class * loss_class
                + args.lambda_width * loss_width
                + args.lambda_coverage * loss_coverage
                + args.lambda_overlap * loss_overlap
                + lambda_boundary * loss_boundary
                + loss_text
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            current_losses = {
                "loss_total": loss,
                "loss_bag": loss_bag,
                "loss_normal": loss_normal,
                "loss_class": loss_class,
                "loss_width": loss_width,
                "loss_coverage": loss_coverage,
                "loss_overlap": loss_overlap,
                "loss_boundary": loss_boundary,
                "loss_text": loss_text,
            }
            for name, value in current_losses.items():
                totals[name] += float(value.detach())
            for name, value in proposal_statistics(proposal_outputs, labels).items():
                statistic_totals[name] = statistic_totals.get(name, 0.0) + value

        scheduler.step()
        divisor = max(1, num_steps)
        epoch_row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{name: value / divisor for name, value in totals.items()},
            **{name: value / divisor for name, value in statistic_totals.items()},
        }
        print(
            f"epoch: {epoch + 1}",
            f"| total: {epoch_row['loss_total']:.4f}",
            f"| bag: {epoch_row['loss_bag']:.4f}",
            f"| normal: {epoch_row['loss_normal']:.4f}",
            f"| class: {epoch_row['loss_class']:.4f}",
            f"| coverage: {epoch_row['loss_coverage']:.4f}",
            f"| width: {epoch_row['width_mean']:.4f}",
            f"| confidence normal/abnormal: "
            f"{epoch_row['normal_confidence_mean']:.4f}/{epoch_row['abnormal_confidence_mean']:.4f}",
        )

        epoch_path = (
            Path(args.epoch_checkpoint_dir)
            / f"model_epoch_{epoch + 1:03d}_temporal_proposal.pth"
        )
        torch.save(model.state_dict(), epoch_path)
        torch.save(model.state_dict(), args.save_cur_path)
        print("Saved epoch checkpoint:", epoch_path)

        auc, ap = test(
            model,
            test_loader,
            args.visual_length,
            prompt_text,
            gt,
            gtsegments,
            gtlabels,
            device,
        )
        epoch_row["classifier_auc"] = float(auc)
        epoch_row["classifier_ap"] = float(ap)
        history.append(epoch_row)
        save_history(Path(args.results_dir) / "training_history.csv", history)

        if auc > best_auc:
            best_auc = float(auc)
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
            print("Saved best checkpoint:", args.checkpoint_path, "AUC:", best_auc)
        else:
            epochs_without_improvement += 1
            print(
                "Early stopping counter:",
                epochs_without_improvement,
                "/",
                args.early_stopping_patience,
            )
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print("Early stopping triggered")
                break

    best_state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output_model_path)
    (Path(args.results_dir) / "training_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (Path(args.results_dir) / "best_result.json").write_text(
        json.dumps(
            {"best_classifier_auc": best_auc, "epochs_ran": len(history)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved best temporal proposal model:", args.output_model_path)


def main():
    args = ucf_option_temporal_proposal.parser.parse_args()
    if not 0 <= args.proposal_min_coverage < args.proposal_max_coverage <= 1:
        raise ValueError(
            "Coverage bounds must satisfy 0 <= min_coverage < max_coverage <= 1"
        )
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = bool(args.pin_memory and device == "cuda")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    normal_dataset = UCFPrototypeDataset(
        args.visual_length, args.train_list, False, args.feature_root, True
    )
    anomaly_dataset = UCFPrototypeDataset(
        args.visual_length, args.train_list, False, args.feature_root, False
    )
    test_dataset = UCFPrototypeDataset(
        args.visual_length, args.test_list, True, args.feature_root, False
    )
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, **loader_kwargs
    )

    model = CLIPVADTemporalProposal(
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
        num_proposals=args.num_proposals,
        proposal_heads=args.proposal_heads,
        min_width=args.proposal_min_width,
        max_width=args.proposal_max_width,
        initial_width=args.proposal_initial_width,
        mask_temperature=args.mask_temperature,
        context_scale=args.context_scale,
    )
    train(model, normal_loader, anomaly_loader, test_loader, args, device)


if __name__ == "__main__":
    main()
