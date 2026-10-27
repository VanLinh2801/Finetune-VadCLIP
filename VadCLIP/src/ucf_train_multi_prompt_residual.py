import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_multi_prompt_residual
from model_multi_prompt_residual import CLIPVADMultiPromptResidual
from ucf_test_description import test
from ucf_train_class_prototype import (
    CLAS2,
    CLASM,
    UCFPrototypeDataset,
    length_to_int,
    text_regularization_loss,
)
from ucf_train_multi_prompt import (
    abnormal_background_suppression_loss,
    multi_prompt_consistency_loss,
    normal_topk_suppression_loss,
    ranking_margin_loss,
)
from utils.prompt_replacement import build_multi_prompt_groups, build_multi_prompt_weights
from utils.topk_tracker import TopKSelectionTracker
from utils.tools import get_batch_label, get_prompt_text


def get_autocast_context(device, use_amp):
    if device == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def freeze_all_except_residuals(model, residual_param_names=("prompt_residuals",)):
    """Freeze VAD/CLIP; keep residual + learnable soft prompts trainable."""
    for clip_param in model.clipmodel.parameters():
        clip_param.requires_grad = False
    if hasattr(model, "prompt_embeddings") and model.prompt_embeddings is not None:
        model.prompt_embeddings.requires_grad_(False)
    allowed = set(residual_param_names) | {"text_prompt_embeddings.weight"}
    trainable = []
    for name, param in model.named_parameters():
        if name in allowed:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    return trainable


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_cur_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    train_mode = getattr(args, "train_mode", "from_scratch")
    warmstart_delta_only = train_mode == "warmstart_delta_only"
    warmstart_continue = train_mode == "warmstart_continue"
    needs_base = warmstart_delta_only or warmstart_continue

    if needs_base:
        base_path = getattr(args, "base_model_path", None)
        if not base_path:
            raise ValueError(f"--base-model-path is required when --train-mode {train_mode}")
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Base Hướng 2a checkpoint not found: {base_path}")
        missing, unexpected = model.load_state_dict(
            torch.load(base_path, map_location=device),
            strict=False,
        )
        print(f"Warm-start ({train_mode}) from Hướng 2a base:", base_path)
        print("  missing keys:", len(missing), "| unexpected keys:", len(unexpected))
    elif args.use_pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device), strict=False)
        print("Loaded pretrained model:", args.pretrained_model_path)
    else:
        print("Skipped VadCLIP checkpoint load. Training from scratch for VadCLIP-specific layers.")

    for clip_param in model.clipmodel.parameters():
        clip_param.requires_grad = False

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

    # Snapshot e_k for logging; forward re-encodes each step so soft prompts stay trainable.
    model.init_prompt_embeddings(prompt_groups)
    if model.prompt_embeddings is not None:
        model.prompt_embeddings.requires_grad_(False)
    with torch.no_grad():
        model.prompt_residuals.zero_()
    print("Initialized prompt embedding snapshot:", tuple(model.prompt_embeddings.shape))
    print("Residual Delta init stats:", model.residual_stats())
    print("Learnable soft prompt requires_grad:", model.text_prompt_embeddings.weight.requires_grad)

    if warmstart_delta_only:
        trainable_names = freeze_all_except_residuals(model, residual_param_names=("prompt_residuals",))
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            "Train mode: warmstart_delta_only | trainable (Delta + soft prompt):",
            trainable_names,
            "| count:",
            n_train,
        )
    else:
        # from_scratch and warmstart_continue: train VAD + soft prompt + Delta (CLIP frozen).
        model.text_prompt_embeddings.weight.requires_grad = True
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Train mode: {train_mode} | trainable param groups:", len(trainable_names), "| count:", n_train)

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
        "Multi-prompt residual config:",
        "| train_mode:", train_mode,
        "| source:", args.prototype_source,
        "| prompt_mode:", args.prompt_mode,
        "| prompts_per_class:", sorted(set(len(group) for group in prompt_groups)),
        "| prompt_weights_json:", args.prompt_weights_json,
        "| lambda_residual:", args.lambda_residual,
        "| lambda_consistency:", args.lambda_consistency,
        "| lambda_normal_topk:", args.lambda_normal_topk,
        "| lambda_abnormal_bg:", args.lambda_abnormal_bg,
        "| lambda_ranking_margin:", args.lambda_ranking_margin,
        "| prototype_json:", args.prototype_json,
        "| use_pretrained_model:", args.use_pretrained_model,
        "| base_model_path:", getattr(args, "base_model_path", None),
    )

    for e in range(args.max_epoch):
        model.train()
        for clip_param in model.clipmodel.parameters():
            clip_param.requires_grad = False
        if model.prompt_embeddings is not None:
            model.prompt_embeddings.requires_grad_(False)
        if warmstart_delta_only:
            freeze_all_except_residuals(model, residual_param_names=("prompt_residuals",))
        else:
            model.text_prompt_embeddings.weight.requires_grad = True

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_cons = 0
        loss_total_normal = 0
        loss_total_abn_bg = 0
        loss_total_margin = 0
        loss_total_res = 0

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
                loss_res = model.residual_l2_loss()
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
                    + args.lambda_residual * loss_res
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
            loss_total_res += loss_res.item()

        scheduler.step()
        num_steps = max(1, min(len(normal_loader), len(anomaly_loader)))
        stats = model.residual_stats()
        print(
            "epoch:", e + 1,
            "| loss1:", loss_total1 / num_steps,
            "| loss2:", loss_total2 / num_steps,
            "| loss3:", loss_total3 / num_steps,
            "| loss_consistency:", loss_total_cons / num_steps,
            "| loss_res:", loss_total_res / num_steps,
            "| mean||Delta||:", f"{stats['mean_norm']:.6f}",
            "| max||Delta||:", f"{stats['max_norm']:.6f}",
            "| mean cos(t,e):", f"{stats['mean_cosine_to_e']:.6f}",
        )
        if topk_tracker is not None:
            topk_tracker.end_epoch(e + 1)
        torch.save(model.state_dict(), args.save_cur_path)
        epoch_checkpoint_path = (
            Path(args.epoch_checkpoint_dir) / f"model_epoch_{e + 1:03d}_multi_prompt_residual.pth"
        )
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print("Saved epoch checkpoint:", epoch_checkpoint_path)

        if args.early_stopping_patience > 0:
            auc, ap = test(
                model,
                testloader,
                args.visual_length,
                prompt_groups,
                gt,
                gtsegments,
                gtlabels,
                device,
                prompt_weights=prompt_weights,
            )
            print(f"epoch {e + 1} eval AUC={auc:.4f} AP={ap:.4f}")
            if auc > best_score:
                best_score = auc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.checkpoint_path)
                print("New best checkpoint:", args.checkpoint_path)
            else:
                epochs_without_improvement += 1
                print(
                    f"No AUC improvement ({epochs_without_improvement}/{args.early_stopping_patience})"
                )
                if epochs_without_improvement >= args.early_stopping_patience:
                    print("Early stopping.")
                    break

    if best_score > 0 and Path(args.checkpoint_path).exists():
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
    else:
        model.load_state_dict(torch.load(args.save_cur_path, map_location=device), strict=False)
    model.prompts_initialized = True
    torch.save(model.state_dict(), args.output_model_path)
    print("Saved fine-tuned model:", args.output_model_path)
    print("Final residual stats:", model.residual_stats())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_multi_prompt_residual.parser.parse_args()
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

    model = CLIPVADMultiPromptResidual(
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
