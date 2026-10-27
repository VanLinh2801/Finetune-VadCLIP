import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_multi_prompt
from model_multi_prompt import CLIPVADMultiPrompt
from ucf_test_description import test
from ucf_train_class_prototype import (
    CLAS2,
    CLASM,
    UCFPrototypeDataset,
    length_to_int,
    text_regularization_loss,
)
from utils.prompt_replacement import build_multi_prompt_groups, build_multi_prompt_weights
from utils.topk_tracker import TopKSelectionTracker
from utils.tools import get_batch_label, get_prompt_text


def get_autocast_context(device, use_amp):
    if device == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def multi_prompt_consistency_loss(prompt_logits, logits1, labels, lengths, top_k_ratio, device):
    losses = []
    num_prompts = prompt_logits.shape[-1]
    if num_prompts <= 1:
        return torch.zeros((), device=device)

    for i, length in enumerate(lengths):
        valid_length = max(1, min(int(length), prompt_logits.shape[1]))
        top_k = max(2, int(valid_length * top_k_ratio))
        top_k = min(top_k, valid_length)
        class_index = int(labels[i].argmax().item())

        anomaly_scores = torch.sigmoid(logits1[i, :valid_length, 0].float())
        top_indices = torch.topk(anomaly_scores, k=top_k, largest=True).indices
        selected_logits = prompt_logits[i, top_indices, class_index, :].float()

        prompt_time_dist = F.softmax(selected_logits.transpose(0, 1), dim=-1).clamp_min(1e-8)
        mean_time_dist = prompt_time_dist.mean(dim=0, keepdim=True).detach().expand_as(prompt_time_dist)
        loss = F.kl_div(prompt_time_dist.log(), mean_time_dist, reduction="batchmean")
        losses.append(loss)

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def get_anomaly_score_branches(logits1, logits2, score_branch):
    branches = []
    if score_branch in ("classifier", "both"):
        branches.append(("classifier", torch.sigmoid(logits1[..., 0].float())))
    if score_branch in ("alignment", "both"):
        branches.append(("alignment", 1.0 - F.softmax(logits2.float(), dim=-1)[..., 0]))
    if not branches:
        raise ValueError(f"Unknown score branch: {score_branch}")
    return branches


def valid_topk_count(valid_length, top_k_ratio):
    top_k = max(1, int(valid_length * top_k_ratio))
    return min(top_k, valid_length)


def normal_topk_suppression_loss(logits1, logits2, labels, lengths, top_k_ratio, score_branch, device):
    losses = []
    branches = get_anomaly_score_branches(logits1, logits2, score_branch)
    for _, scores in branches:
        branch_losses = []
        for i, length in enumerate(lengths):
            class_index = int(labels[i].argmax().item())
            if class_index != 0:
                continue
            valid_length = max(1, min(int(length), scores.shape[1]))
            top_k = valid_topk_count(valid_length, top_k_ratio)
            branch_losses.append(torch.topk(scores[i, :valid_length], k=top_k, largest=True).values.mean())
        if branch_losses:
            losses.append(torch.stack(branch_losses).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def abnormal_background_suppression_loss(
    logits1, logits2, labels, lengths, top_k_ratio, margin, score_branch, device
):
    """Intra-video high/low top-k separation margin on abnormal videos.

    For each abnormal sample, select k highest and k lowest scores (indices from
    detached scores), then enforce:
        min(high_k) >= max(low_k) + margin
    via hinge loss:
        relu(max(low_k) - min(high_k) + margin)
    """
    losses = []
    margin = float(margin)
    branches = get_anomaly_score_branches(logits1, logits2, score_branch)
    for _, scores in branches:
        branch_losses = []
        for i, length in enumerate(lengths):
            class_index = int(labels[i].argmax().item())
            if class_index == 0:
                continue
            valid_length = max(1, min(int(length), scores.shape[1]))
            top_k = valid_topk_count(valid_length, top_k_ratio)
            if 2 * top_k > valid_length:
                continue
            valid_scores = scores[i, :valid_length]
            high_indices = torch.topk(valid_scores.detach(), k=top_k, largest=True).indices
            low_indices = torch.topk(valid_scores.detach(), k=top_k, largest=False).indices
            high_vals = valid_scores[high_indices]
            low_vals = valid_scores[low_indices]
            branch_losses.append(F.relu(low_vals.max() - high_vals.min() + margin))
        if branch_losses:
            losses.append(torch.stack(branch_losses).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def ranking_margin_loss(logits1, logits2, labels, lengths, top_k_ratio, margin, score_branch, device):
    losses = []
    branches = get_anomaly_score_branches(logits1, logits2, score_branch)
    for _, scores in branches:
        normal_topk_scores = []
        abnormal_topk_scores = []
        for i, length in enumerate(lengths):
            valid_length = max(1, min(int(length), scores.shape[1]))
            top_k = valid_topk_count(valid_length, top_k_ratio)
            topk_mean = torch.topk(scores[i, :valid_length], k=top_k, largest=True).values.mean()
            class_index = int(labels[i].argmax().item())
            if class_index == 0:
                normal_topk_scores.append(topk_mean)
            else:
                abnormal_topk_scores.append(topk_mean)
        if normal_topk_scores and abnormal_topk_scores:
            normal_mean = torch.stack(normal_topk_scores).mean()
            abnormal_mean = torch.stack(abnormal_topk_scores).mean()
            losses.append(F.relu(torch.tensor(margin, device=device) - abnormal_mean + normal_mean))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_cur_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    if args.use_pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device), strict=False)
        print("Loaded pretrained model:", args.pretrained_model_path)
    else:
        print("Skipped VadCLIP checkpoint load. Training from scratch for VadCLIP-specific layers.")

    for clip_param in model.clipmodel.parameters():
        clip_param.requires_grad = False
    model.text_prompt_embeddings.weight.requires_grad = args.train_learnable_prompt

    class_prompt_text = get_prompt_text(label_map)
    prompt_groups = build_multi_prompt_groups(
        args.prototype_json,
        label_map,
        args.prototype_source,
        args.compact_prototype_types,
        args.long_prototype_types,
        args.prompt_mode,
    )
    prompt_weights = build_multi_prompt_weights(args.prompt_weights_json, label_map, device)
    if prompt_weights is not None and prompt_weights.shape != (len(prompt_groups), len(prompt_groups[0])):
        raise ValueError(
            f"Prompt weights shape {tuple(prompt_weights.shape)} does not match prompt groups "
            f"shape {(len(prompt_groups), len(prompt_groups[0]))}"
        )

    print("Train learnable prefix/postfix:", args.train_learnable_prompt)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    use_amp = bool(args.use_amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = 0
    epochs_without_improvement = 0
    topk_tracker = (
        TopKSelectionTracker(
            args.topk_tracking_output_dir,
            branches=args.topk_tracking_branches,
            scope=args.topk_tracking_scope,
        )
        if args.track_topk
        else None
    )

    print(
        "Multi-prompt config:",
        "| source:", args.prototype_source,
        "| compact_types:", args.compact_prototype_types,
        "| long_types:", args.long_prototype_types,
        "| prompt_mode:", args.prompt_mode,
        "| prompts_per_class:", sorted(set(len(group) for group in prompt_groups)),
        "| prompt_weights_json:", args.prompt_weights_json,
        "| prompt_weight_range:",
        None if prompt_weights is None else (float(prompt_weights.min().item()), float(prompt_weights.max().item())),
        "| lambda_consistency:", args.lambda_consistency,
        "| regularize_score_branch:", args.regularize_score_branch,
        "| regularize_top_k_ratio:", args.regularize_top_k_ratio,
        "| lambda_normal_topk:", args.lambda_normal_topk,
        "| lambda_abnormal_bg:", args.lambda_abnormal_bg,
        "| abnormal_bg_margin:", args.abnormal_bg_margin,
        "| abnormal_bg_warmup_epochs:", args.abnormal_bg_warmup_epochs,
        "| lambda_ranking_margin:", args.lambda_ranking_margin,
        "| ranking_margin:", args.ranking_margin,
        "| prototype_json:", args.prototype_json,
    )

    for e in range(args.max_epoch):
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_cons = 0
        loss_total_normal = 0
        loss_total_abn_bg = 0
        loss_total_margin = 0

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            normal_batch = next(normal_iter)
            anomaly_batch = next(anomaly_iter)
            if topk_tracker is not None:
                normal_features, normal_label, normal_lengths, _, normal_sample_ids = normal_batch
                anomaly_features, anomaly_label, anomaly_lengths, _, anomaly_sample_ids = anomaly_batch
            else:
                normal_features, normal_label, normal_lengths, _ = normal_batch
                anomaly_features, anomaly_label, anomaly_lengths, _ = anomaly_batch
                normal_sample_ids = []
                anomaly_sample_ids = []

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device, non_blocking=True)
            text_labels_raw = list(normal_label) + list(anomaly_label)
            feat_lengths_tensor = torch.cat([normal_lengths, anomaly_lengths], dim=0)
            feat_lengths = [length_to_int(length, args.visual_length) for length in feat_lengths_tensor]
            text_labels = get_batch_label(text_labels_raw, class_prompt_text, label_map).to(device, non_blocking=True)

            with get_autocast_context(device, use_amp):
                text_features, logits1, logits2, prompt_logits = model(
                    visual_features,
                    None,
                    prompt_groups,
                    feat_lengths,
                    return_prompt_logits=True,
                    prompt_weights=prompt_weights,
                )
            if topk_tracker is not None:
                topk_tracker.update_from_batch(
                    e + 1,
                    list(normal_sample_ids) + list(anomaly_sample_ids),
                    text_labels_raw,
                    text_labels,
                    feat_lengths,
                    logits1,
                    logits2,
                )

            loss1 = CLAS2(logits1.float(), text_labels.float(), feat_lengths, device)
            loss2 = CLASM(logits2.float(), text_labels.float(), feat_lengths, device)
            loss3 = text_regularization_loss(text_features.float(), device)
            loss_consistency = multi_prompt_consistency_loss(
                prompt_logits.float(),
                logits1.float(),
                text_labels.float(),
                feat_lengths,
                args.consistency_top_k_ratio,
                device,
            )
            loss_normal = normal_topk_suppression_loss(
                logits1.float(),
                logits2.float(),
                text_labels.float(),
                feat_lengths,
                args.regularize_top_k_ratio,
                args.regularize_score_branch,
                device,
            )
            loss_abn_bg = abnormal_background_suppression_loss(
                logits1.float(),
                logits2.float(),
                text_labels.float(),
                feat_lengths,
                args.regularize_top_k_ratio,
                args.abnormal_bg_margin,
                args.regularize_score_branch,
                device,
            )
            loss_margin = ranking_margin_loss(
                logits1.float(),
                logits2.float(),
                text_labels.float(),
                feat_lengths,
                args.regularize_top_k_ratio,
                args.ranking_margin,
                args.regularize_score_branch,
                device,
            )
            lambda_abnormal_bg = (
                0.0 if e < args.abnormal_bg_warmup_epochs else args.lambda_abnormal_bg
            )
            loss = (
                loss1
                + loss2
                + loss3
                + args.lambda_consistency * loss_consistency
                + args.lambda_normal_topk * loss_normal
                + lambda_abnormal_bg * loss_abn_bg
                + args.lambda_ranking_margin * loss_margin
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total3 += loss3.item()
            loss_total_cons += loss_consistency.item()
            loss_total_normal += loss_normal.item()
            loss_total_abn_bg += loss_abn_bg.item()
            loss_total_margin += loss_margin.item()

            step = i * normal_loader.batch_size * 2
            if args.eval_steps > 0 and step % args.eval_steps == 0 and step != 0:
                print(
                    "epoch:", e + 1,
                    "| step:", step,
                    "| loss1:", loss_total1 / (i + 1),
                    "| loss2:", loss_total2 / (i + 1),
                    "| loss3:", loss_total3 / (i + 1),
                    "| loss_consistency:", loss_total_cons / (i + 1),
                    "| loss_normal_topk:", loss_total_normal / (i + 1),
                    "| loss_abnormal_bg:", loss_total_abn_bg / (i + 1),
                    "| loss_ranking_margin:", loss_total_margin / (i + 1),
                )
                auc, ap = test(model, testloader, args.visual_length, prompt_groups, gt, gtsegments, gtlabels, device)
                if auc > best_score:
                    best_score = auc
                    torch.save(model.state_dict(), args.checkpoint_path)

        scheduler.step()
        num_steps = max(1, min(len(normal_loader), len(anomaly_loader)))
        print(
            "epoch:", e + 1,
            "| loss1:", loss_total1 / num_steps,
            "| loss2:", loss_total2 / num_steps,
            "| loss3:", loss_total3 / num_steps,
            "| loss_consistency:", loss_total_cons / num_steps,
            "| loss_normal_topk:", loss_total_normal / num_steps,
            "| loss_abnormal_bg:", loss_total_abn_bg / num_steps,
            "| loss_ranking_margin:", loss_total_margin / num_steps,
        )
        if topk_tracker is not None:
            topk_tracker.end_epoch(e + 1)
        torch.save(model.state_dict(), args.save_cur_path)
        epoch_checkpoint_path = Path(args.epoch_checkpoint_dir) / f"model_epoch_{e + 1:03d}_multi_prompt.pth"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print("Saved epoch checkpoint:", epoch_checkpoint_path)

        if args.early_stopping_patience > 0:
            auc, ap = test(model, testloader, args.visual_length, prompt_groups, gt, gtsegments, gtlabels, device)
            print(
                "epoch:", e + 1,
                "| validation_auc:", auc,
                "| validation_ap:", ap,
                "| best_auc:", best_score,
            )
            if auc > best_score:
                best_score = auc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.checkpoint_path)
                print("Saved best checkpoint:", args.checkpoint_path)
            else:
                epochs_without_improvement += 1
                print(
                    "Early stopping counter:",
                    epochs_without_improvement,
                    "/",
                    args.early_stopping_patience,
                )
                if epochs_without_improvement >= args.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

    if best_score > 0:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
    else:
        model.load_state_dict(torch.load(args.save_cur_path, map_location=device), strict=False)
    torch.save(model.state_dict(), args.output_model_path)
    print("Saved fine-tuned model:", args.output_model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_multi_prompt.parser.parse_args()
    setup_seed(args.seed)

    label_map = {
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

    pin_memory = bool(args.pin_memory and device == "cuda")
    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    normal_dataset = UCFPrototypeDataset(
        args.visual_length,
        args.train_list,
        False,
        args.feature_root,
        True,
        return_sample_id=args.track_topk,
    )
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )
    anomaly_dataset = UCFPrototypeDataset(
        args.visual_length,
        args.train_list,
        False,
        args.feature_root,
        False,
        return_sample_id=args.track_topk,
    )
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )
    test_dataset = UCFPrototypeDataset(args.visual_length, args.test_list, True, args.feature_root, False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **dataloader_kwargs)

    model = CLIPVADMultiPrompt(
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

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
