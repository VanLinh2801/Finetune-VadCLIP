import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from pathlib import Path

from model import CLIPVAD
from model_multi_classifier import (
    CLIPVADAdapterMultiClassifier,
    CLIPVADMultiClassifier,
)
from ucf_test import test
from utils.dataset import UCFDataset
from utils.topk_tracker import TopKSelectionTracker
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def get_anomaly_score_branches(logits1, logits2, score_branch):
    branches = []
    if score_branch in ('classifier', 'both'):
        branches.append(torch.sigmoid(logits1[..., 0].float()))
    if score_branch in ('alignment', 'both'):
        branches.append(1.0 - F.softmax(logits2.float(), dim=-1)[..., 0])
    if not branches:
        raise ValueError(f'Unknown score branch: {score_branch}')
    return branches

def normal_topk_suppression_loss(logits1, logits2, labels, lengths, top_k_ratio, score_branch, device):
    losses = []
    branches = get_anomaly_score_branches(logits1, logits2, score_branch)
    for scores in branches:
        branch_losses = []
        for i, length in enumerate(lengths):
            class_index = int(labels[i].argmax().item())
            if class_index != 0:
                continue
            valid_length = max(1, min(int(length), scores.shape[1]))
            top_k = max(1, int(valid_length * top_k_ratio))
            top_k = min(top_k, valid_length)
            branch_losses.append(torch.topk(scores[i, :valid_length], k=top_k, largest=True).values.mean())
        if branch_losses:
            losses.append(torch.stack(branch_losses).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()

def abnormal_background_suppression_loss(
    logits1, logits2, labels, lengths, top_k_ratio, margin, score_branch, device
):
    losses = []
    margin = float(margin)
    branches = get_anomaly_score_branches(logits1, logits2, score_branch)
    for scores in branches:
        branch_losses = []
        for i, length in enumerate(lengths):
            class_index = int(labels[i].argmax().item())
            if class_index == 0:
                continue
            valid_length = max(1, min(int(length), scores.shape[1]))
            top_k = max(1, int(valid_length * top_k_ratio))
            top_k = min(top_k, valid_length)
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
    for scores in get_anomaly_score_branches(logits1, logits2, score_branch):
        normal_scores = []
        abnormal_scores = []
        for i, length in enumerate(lengths):
            valid_length = min(int(length), scores.shape[1])
            if valid_length <= 0:
                continue
            top_k = max(1, int(valid_length * top_k_ratio))
            topk_mean = scores[i, :valid_length].topk(top_k).values.mean()
            if int(labels[i].argmax().item()) == 0:
                normal_scores.append(topk_mean)
            else:
                abnormal_scores.append(topk_mean)
        if normal_scores and abnormal_scores:
            normal_mean = torch.stack(normal_scores).mean()
            abnormal_mean = torch.stack(abnormal_scores).mean()
            losses.append(F.relu(normal_mean + margin - abnormal_mean))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def hard_pair_soft_ranking_loss(
    logits1, logits2, labels, lengths, margin, temperature, hard_normal_fraction, score_branch
):
    branch_losses = []
    active_fractions = []
    for scores in get_anomaly_score_branches(logits1, logits2, score_branch):
        normal_scores = []
        abnormal_scores = []
        for index, length in enumerate(lengths):
            valid_length = min(int(length), scores.shape[1])
            if valid_length <= 0:
                continue
            # Match the original MIL selection rather than selecting 15% of snippets.
            k = min(valid_length, int(valid_length / 16 + 1))
            video_score = scores[index, :valid_length].topk(k).values.mean()
            if int(labels[index].argmax().item()) == 0:
                normal_scores.append(video_score)
            else:
                abnormal_scores.append(video_score)

        if not normal_scores or not abnormal_scores:
            continue
        normal_scores = torch.stack(normal_scores)
        abnormal_scores = torch.stack(abnormal_scores)
        hard_count = max(1, int(normal_scores.numel() * hard_normal_fraction))
        hard_normals = normal_scores.topk(hard_count).values
        violations = margin + hard_normals.unsqueeze(0) - abnormal_scores.unsqueeze(1)
        branch_losses.append(temperature * F.softplus(violations / temperature).mean())
        active_fractions.append((violations.detach() > 0).float().mean())

    if not branch_losses:
        raise ValueError('Hard-pair ranking requires normal and abnormal samples in the batch.')
    return torch.stack(branch_losses).mean(), torch.stack(active_fractions).mean()


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    topk_tracker = create_topk_tracker(args)
    epoch = 0
    eval_steps = int(getattr(args, 'eval_steps', 1280))
    eval_each_epoch = bool(getattr(args, 'eval_each_epoch', False))

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    def save_checkpoint(current_epoch, current_ap):
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ap': current_ap,
        }
        torch.save(checkpoint, args.checkpoint_path)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total_normal = 0
        loss_total_abn_bg = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths, normal_sample_ids = unpack_training_batch(
                next(normal_iter), topk_tracker
            )
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_sample_ids = unpack_training_batch(
                next(anomaly_iter), topk_tracker
            )

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            label_names = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(label_names, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths) 
            if topk_tracker is not None:
                topk_tracker.update_from_batch(
                    e + 1,
                    normal_sample_ids + anomaly_sample_ids,
                    label_names,
                    text_labels,
                    feat_lengths,
                    logits1,
                    logits2,
                )
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            loss_normal = normal_topk_suppression_loss(
                logits1,
                logits2,
                text_labels,
                feat_lengths,
                args.regularize_top_k_ratio,
                args.regularize_score_branch,
                device,
            )
            loss_total_normal += loss_normal.item()

            loss_abn_bg = abnormal_background_suppression_loss(
                logits1,
                logits2,
                text_labels,
                feat_lengths,
                args.regularize_top_k_ratio,
                args.abnormal_bg_margin,
                args.regularize_score_branch,
                device,
            )
            loss_total_abn_bg += loss_abn_bg.item()

            lambda_normal_topk = (
                0.0 if e < args.normal_topk_warmup_epochs else args.lambda_normal_topk
            )
            lambda_abnormal_bg = (
                0.0 if e < args.abnormal_bg_warmup_epochs else args.lambda_abnormal_bg
            )
            loss = (
                loss1
                + loss2
                + loss3
                + lambda_normal_topk * loss_normal
                + lambda_abnormal_bg * loss_abn_bg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if eval_steps > 0 and step % eval_steps == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(), '| loss_normal_topk: ', loss_total_normal / (i+1), '| lambda_normal_topk: ', lambda_normal_topk, '| loss_abnormal_bg: ', loss_total_abn_bg / (i+1), '| lambda_abnormal_bg: ', lambda_abnormal_bg)
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    save_checkpoint(e, ap_best)

        scheduler.step()

        print('epoch: ', e+1, '| loss1: ', loss_total1 / max(1, len(normal_loader)), '| loss2: ', loss_total2 / max(1, len(normal_loader)), '| loss3: ', loss3.item(), '| loss_normal_topk: ', loss_total_normal / max(1, len(normal_loader)), '| lambda_normal_topk: ', lambda_normal_topk, '| loss_abnormal_bg: ', loss_total_abn_bg / max(1, len(normal_loader)), '| lambda_abnormal_bg: ', lambda_abnormal_bg)
        if topk_tracker is not None:
            topk_tracker.end_epoch(e + 1)
        if eval_each_epoch:
            AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
            AP = AUC
            if AP > ap_best:
                ap_best = AP
                save_checkpoint(e, ap_best)

        if getattr(args, 'save_epoch_checkpoints', False):
            epoch_checkpoint_dir = Path(args.epoch_checkpoint_dir)
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = epoch_checkpoint_dir / f'model_epoch_{e + 1:03d}.pth'
            torch.save(model.state_dict(), epoch_path)
            print('Saved epoch checkpoint:', epoch_path)

        torch.save(model.state_dict(), 'model/model_cur.pth')
        if Path(args.checkpoint_path).exists():
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

    if Path(args.checkpoint_path).exists():
        checkpoint = torch.load(args.checkpoint_path)
        torch.save(checkpoint['model_state_dict'], args.model_path)
    else:
        torch.save(model.state_dict(), args.model_path)

def train_ranking_margin_epoch_eval(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    if args.use_checkpoint:
        raise ValueError('Ranking experiments train from scratch; use_checkpoint must be false.')
    if not 0 < args.regularize_top_k_ratio <= 1:
        raise ValueError('regularize_top_k_ratio must be in (0, 1].')
    if args.ranking_margin < 0 or args.lambda_ranking_margin < 0 or args.ranking_warmup_epochs < 0:
        raise ValueError('ranking margin, lambda, and warmup epochs must be non-negative.')
    hard_pair_mode = args.training_mode == 'ranking_hard_pair_epoch_eval'
    if hard_pair_mode and (
        args.ranking_soft_temperature <= 0
        or not 0 < args.ranking_hard_normal_fraction <= 1
    ):
        raise ValueError('Hard-pair temperature must be positive and normal fraction must be in (0, 1].')

    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    tracker = create_topk_tracker(args)
    epoch_dir = Path(args.epoch_checkpoint_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for output_path in (args.model_path, args.checkpoint_path, args.save_cur_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.max_epoch):
        model.train()
        totals = {'cls': 0.0, 'mil': 0.0, 'text': 0.0, 'ranking': 0.0, 'active_pairs': 0.0}
        num_steps = min(len(normal_loader), len(anomaly_loader))
        if num_steps == 0:
            raise ValueError('Training requires at least one full normal and abnormal batch.')
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        ranking_weight = 0.0 if epoch < args.ranking_warmup_epochs else args.lambda_ranking_margin

        for _ in range(num_steps):
            normal_features, normal_labels, normal_lengths, normal_ids = unpack_training_batch(
                next(normal_iter), tracker
            )
            abnormal_features, abnormal_labels, abnormal_lengths, abnormal_ids = unpack_training_batch(
                next(anomaly_iter), tracker
            )
            visual = torch.cat((normal_features, abnormal_features), dim=0).to(device)
            names = list(normal_labels) + list(abnormal_labels)
            lengths = torch.cat((normal_lengths, abnormal_lengths), dim=0).to(device)
            labels = get_batch_label(names, prompt_text, label_map).to(device)
            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
            if tracker is not None:
                tracker.update_from_batch(
                    epoch + 1, normal_ids + abnormal_ids, names, labels, lengths, logits1, logits2
                )

            loss_cls = CLAS2(logits1, labels, lengths, device)
            loss_mil = CLASM(logits2, labels, lengths, device)
            loss_text = text_regularization_loss(text_features, device)
            if hard_pair_mode:
                loss_ranking, active_pairs = hard_pair_soft_ranking_loss(
                    logits1, logits2, labels, lengths, args.ranking_margin,
                    args.ranking_soft_temperature, args.ranking_hard_normal_fraction,
                    args.regularize_score_branch,
                )
            else:
                loss_ranking = ranking_margin_loss(
                    logits1, logits2, labels, lengths, args.regularize_top_k_ratio,
                    args.ranking_margin, args.regularize_score_branch, device,
                )
                active_pairs = None
            loss = loss_cls + loss_mil + loss_text + ranking_weight * loss_ranking
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch + 1}.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, value in (
                ('cls', loss_cls), ('mil', loss_mil), ('text', loss_text), ('ranking', loss_ranking)
            ):
                totals[name] += value.item()
            if active_pairs is not None:
                totals['active_pairs'] += active_pairs.item()

        scheduler.step()
        if tracker is not None:
            tracker.end_epoch(epoch + 1)
        print(
            f'epoch: {epoch + 1} | loss_cls: {totals["cls"] / num_steps:.6f}'
            f' | loss_mil: {totals["mil"] / num_steps:.6f}'
            f' | loss_text: {totals["text"] / num_steps:.6f}'
            f' | loss_ranking: {totals["ranking"] / num_steps:.6f}'
            f' | lambda_ranking: {ranking_weight:.4f}'
            + (f' | active_pair_fraction: {totals["active_pairs"] / num_steps:.4f}'
               if hard_pair_mode else ''),
            flush=True,
        )
        epoch_path = epoch_dir / f'model_epoch_{epoch + 1:03d}.pth'
        torch.save(model.state_dict(), epoch_path)
        torch.save(model.state_dict(), args.save_cur_path)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, args.checkpoint_path)
        print('Saved epoch checkpoint:', epoch_path, flush=True)
        test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

    torch.save(model.state_dict(), args.model_path)
    print('Final epoch model:', args.model_path, flush=True)


def train_abnormal_topk_epoch_eval(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    topk_tracker = create_topk_tracker(args)

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", checkpoint['epoch'] + 1, " ap:", ap_best)

    def save_checkpoint(current_epoch, current_ap):
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ap': current_ap,
        }
        torch.save(checkpoint, args.checkpoint_path)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total_normal = 0
        loss_total_abn_bg = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_steps = min(len(normal_loader), len(anomaly_loader))

        for i in range(num_steps):
            normal_features, normal_label, normal_lengths, normal_sample_ids = unpack_training_batch(
                next(normal_iter), topk_tracker
            )
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_sample_ids = unpack_training_batch(
                next(anomaly_iter), topk_tracker
            )

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            label_names = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(label_names, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)
            if topk_tracker is not None:
                topk_tracker.update_from_batch(
                    e + 1,
                    normal_sample_ids + anomaly_sample_ids,
                    label_names,
                    text_labels,
                    feat_lengths,
                    logits1,
                    logits2,
                )

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            loss_normal = normal_topk_suppression_loss(
                logits1,
                logits2,
                text_labels,
                feat_lengths,
                args.regularize_top_k_ratio,
                args.regularize_score_branch,
                device,
            )
            loss_total_normal += loss_normal.item()

            loss_abn_bg = abnormal_background_suppression_loss(
                logits1,
                logits2,
                text_labels,
                feat_lengths,
                args.regularize_top_k_ratio,
                args.abnormal_bg_margin,
                args.regularize_score_branch,
                device,
            )
            loss_total_abn_bg += loss_abn_bg.item()

            lambda_abnormal_bg = (
                0.0 if e < args.abnormal_bg_warmup_epochs else args.lambda_abnormal_bg
            )
            loss = loss1 + loss2 + loss3 + lambda_abnormal_bg * loss_abn_bg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(
            'epoch: ', e + 1,
            '| loss1: ', loss_total1 / max(1, num_steps),
            '| loss2: ', loss_total2 / max(1, num_steps),
            '| loss3: ', loss3.item(),
            '| loss_normal_topk_monitor_only: ', loss_total_normal / max(1, num_steps),
            '| loss_abnormal_bg: ', loss_total_abn_bg / max(1, num_steps),
            '| lambda_abnormal_bg: ', lambda_abnormal_bg,
        )
        if topk_tracker is not None:
            topk_tracker.end_epoch(e + 1)

        AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        AP = AUC
        if AP > ap_best:
            ap_best = AP
            save_checkpoint(e, ap_best)

        if getattr(args, 'save_epoch_checkpoints', False):
            epoch_checkpoint_dir = Path(args.epoch_checkpoint_dir)
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = epoch_checkpoint_dir / f'model_epoch_{e + 1:03d}.pth'
            torch.save(model.state_dict(), epoch_path)
            print('Saved epoch checkpoint:', epoch_path)

        torch.save(model.state_dict(), 'model/model_cur_abnormal_topk.pth')

    if Path(args.checkpoint_path).exists():
        checkpoint = torch.load(args.checkpoint_path)
        torch.save(checkpoint['model_state_dict'], args.model_path)
    else:
        torch.save(model.state_dict(), args.model_path)


def _best_contiguous_window_start(scores, segment_length, largest=True):
    """Select a contiguous window from detached snippet scores."""
    windows = scores.unfold(0, segment_length, 1).mean(dim=-1)
    return int((windows.argmax() if largest else windows.argmin()).item())


def _context_compatible_starts(valid_length, segment_length, context_width):
    first = max(0, int(context_width))
    last = valid_length - segment_length - max(0, int(context_width))
    if last < first:
        return []
    return list(range(first, last + 1))


def _mean_cosine_similarity(left, right):
    if left.shape[0] == 0 or right.shape[0] == 0:
        return None
    return F.cosine_similarity(
        left.float().mean(dim=0, keepdim=True),
        right.float().mean(dim=0, keepdim=True),
        dim=-1,
    )[0]


def build_hard_negative_transplants(
    normal_features,
    anomaly_features,
    normal_lengths,
    anomaly_lengths,
    normal_classifier_scores,
    normal_alignment_scores,
    anomaly_classifier_scores,
    anomaly_alignment_scores,
    min_segment_length,
    max_segment_length,
    context_width,
    high_exclusion_ratio,
    boundary_blend_width,
):
    """Insert a hard Normal segment only where local visual context is compatible."""
    transplanted = anomaly_features.clone()
    inserted_mask = torch.zeros(
        anomaly_features.shape[:2], dtype=torch.bool, device=anomaly_features.device
    )

    normal_count = normal_features.shape[0]
    for i in range(anomaly_features.shape[0]):
        source_i = i % normal_count
        source_length = max(1, min(int(normal_lengths[source_i]), normal_features.shape[1]))
        target_length = max(1, min(int(anomaly_lengths[i]), anomaly_features.shape[1]))
        available = min(source_length, target_length, int(max_segment_length))
        if available < 1:
            continue
        lower = min(max(1, int(min_segment_length)), available)
        segment_length = int(
            torch.randint(lower, available + 1, (1,), device=anomaly_features.device).item()
        )

        source_starts = _context_compatible_starts(
            source_length, segment_length, context_width
        )
        target_starts = _context_compatible_starts(
            target_length, segment_length, context_width
        )
        if not source_starts or not target_starts:
            continue

        source_evidence = 0.5 * (
            normal_classifier_scores[source_i, :source_length]
            + normal_alignment_scores[source_i, :source_length]
        )
        source_window_scores = torch.stack(
            [source_evidence[start:start + segment_length].mean() for start in source_starts]
        )
        source_start = source_starts[int(source_window_scores.argmax().item())]

        classifier_scores = anomaly_classifier_scores[i, :target_length]
        alignment_scores = anomaly_alignment_scores[i, :target_length]
        top_k = max(1, int(target_length * float(high_exclusion_ratio)))
        top_k = min(top_k, target_length)
        high_indices = torch.cat(
            [
                torch.topk(classifier_scores, k=top_k, largest=True).indices,
                torch.topk(alignment_scores, k=top_k, largest=True).indices,
            ]
        ).unique()
        protected = torch.zeros(target_length, dtype=torch.bool, device=anomaly_features.device)
        for high_index in high_indices.tolist():
            protected[
                max(0, high_index - context_width):min(target_length, high_index + context_width + 1)
            ] = True

        classifier_median = classifier_scores.median()
        alignment_median = alignment_scores.median()
        safe_starts = []
        for start in target_starts:
            end = start + segment_length
            if protected[start:end].any():
                continue
            if (
                classifier_scores[start:end].mean() <= classifier_median
                and alignment_scores[start:end].mean() <= alignment_median
            ):
                safe_starts.append(start)
        # Skip this sample rather than relaxing either low-score agreement or
        # the protected high-evidence neighborhood.
        if not safe_starts:
            continue

        source_left = normal_features[
            source_i, source_start - context_width:source_start
        ]
        source_right = normal_features[
            source_i,
            source_start + segment_length:source_start + segment_length + context_width,
        ]
        source_segment = normal_features[
            source_i, source_start:source_start + segment_length
        ]

        match_scores = []
        for start in safe_starts:
            end = start + segment_length
            target_left = anomaly_features[i, start - context_width:start]
            target_right = anomaly_features[i, end:end + context_width]
            target_segment = anomaly_features[i, start:end]
            left_similarity = _mean_cosine_similarity(source_left, target_left)
            right_similarity = _mean_cosine_similarity(source_right, target_right)
            segment_similarity = _mean_cosine_similarity(source_segment, target_segment)
            if left_similarity is None or right_similarity is None or segment_similarity is None:
                match_scores.append(torch.tensor(float('-inf'), device=anomaly_features.device))
            else:
                match_scores.append(
                    0.4 * left_similarity + 0.4 * right_similarity + 0.2 * segment_similarity
                )
        target_start = safe_starts[int(torch.stack(match_scores).argmax().item())]
        target_end = target_start + segment_length
        target_original = anomaly_features[i, target_start:target_end]
        blended_segment = source_segment.clone()
        blend_width = min(
            max(0, int(boundary_blend_width)), max(0, (segment_length - 1) // 2)
        )
        for boundary_offset in range(blend_width):
            source_weight = float(boundary_offset + 1) / float(blend_width + 1)
            blended_segment[boundary_offset] = (
                source_weight * source_segment[boundary_offset]
                + (1.0 - source_weight) * target_original[boundary_offset]
            )
            right_offset = segment_length - 1 - boundary_offset
            blended_segment[right_offset] = (
                source_weight * source_segment[right_offset]
                + (1.0 - source_weight) * target_original[right_offset]
            )
        transplanted[i, target_start:target_end] = blended_segment

        pure_start = target_start + blend_width
        pure_end = target_end - blend_width
        if pure_end > pure_start:
            inserted_mask[i, pure_start:pure_end] = True

    return transplanted, inserted_mask


def build_evidence_erased_batch(
    anomaly_features,
    anomaly_lengths,
    anomaly_scores,
    min_segment_length,
    max_segment_length,
):
    """Replace the strongest contiguous run with surrounding context features."""
    erased = anomaly_features.clone()
    erased_mask = torch.zeros(
        anomaly_features.shape[:2], dtype=torch.bool, device=anomaly_features.device
    )

    for i in range(anomaly_features.shape[0]):
        valid_length = max(1, min(int(anomaly_lengths[i]), anomaly_features.shape[1]))
        available = min(valid_length, int(max_segment_length))
        if available < 1:
            continue
        lower = min(max(1, int(min_segment_length)), available)
        segment_length = int(
            torch.randint(lower, available + 1, (1,), device=anomaly_features.device).item()
        )
        start = _best_contiguous_window_start(
            anomaly_scores[i, :valid_length], segment_length, largest=True
        )
        end = start + segment_length

        context_width = max(1, segment_length // 2)
        left = anomaly_features[i, max(0, start - context_width):start]
        right = anomaly_features[i, end:min(valid_length, end + context_width)]
        context = torch.cat([left, right], dim=0)
        if context.shape[0] == 0:
            replacement = anomaly_features[i, :valid_length].mean(dim=0)
        else:
            replacement = context.mean(dim=0)
        erased[i, start:end] = replacement
        erased_mask[i, start:end] = True

    return erased, erased_mask


def transplanted_negative_loss(logits1, logits2, inserted_mask, score_branch, device):
    losses = []
    for scores in get_anomaly_score_branches(logits1, logits2, score_branch):
        selected = scores[inserted_mask]
        if selected.numel() > 0:
            losses.append(selected.mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def text_regularization_loss(text_features, device):
    loss = torch.zeros((), device=device)
    text_feature_normal = F.normalize(text_features[0].float(), dim=-1)
    for j in range(1, text_features.shape[0]):
        text_feature_abnormal = F.normalize(text_features[j].float(), dim=-1)
        loss = loss + torch.abs(text_feature_normal @ text_feature_abnormal)
    return loss / max(1, text_features.shape[0] - 1) * 1e-1


def create_topk_tracker(args):
    if not getattr(args, 'track_topk', False):
        return None
    return TopKSelectionTracker(
        args.topk_tracking_output_dir,
        branches=args.topk_tracking_branches,
        scope=args.topk_tracking_scope,
    )


def unpack_training_batch(batch, tracker):
    if tracker is None:
        features, labels, lengths = batch
        return features, labels, lengths, []
    features, labels, lengths, sample_ids = batch
    return features, labels, lengths, list(sample_ids)


def train_transplant_erasing_epoch_eval(
    model, normal_loader, anomaly_loader, testloader, args, label_map, device
):
    """Train original VadCLIP with normal transplantation and evidence erasing."""
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    best_auc = float('-inf')
    stale_epochs = 0
    epoch_checkpoint_dir = Path(args.epoch_checkpoint_dir)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_cur_path).parent.mkdir(parents=True, exist_ok=True)
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
        'Transplant + evidence erasing config:',
        '| warmup_epochs:', args.augmentation_warmup_epochs,
        '| lambda_transplant:', args.lambda_transplant,
        '| lambda_evidence_erasing:', args.lambda_evidence_erasing,
        '| segment_length:',
        f'[{args.augmentation_min_segment_length}, {args.augmentation_max_segment_length}]',
        '| transplant_branch:', args.transplant_score_branch,
        '| context_width:', args.transplant_context_width,
        '| high_exclusion_ratio:', args.transplant_high_exclusion_ratio,
        '| boundary_blend_width:', args.transplant_boundary_blend_width,
    )

    num_steps = min(len(normal_loader), len(anomaly_loader))
    for epoch in range(args.max_epoch):
        model.train()
        totals = {
            'cls': 0.0,
            'mil': 0.0,
            'text': 0.0,
            'transplant': 0.0,
            'erase': 0.0,
            'transplant_samples': 0,
            'transplant_snippets': 0,
            'augmentation_samples': 0,
        }
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        augmentation_active = epoch >= args.augmentation_warmup_epochs

        for _ in range(num_steps):
            normal_batch = next(normal_iter)
            anomaly_batch = next(anomaly_iter)
            if topk_tracker is not None:
                normal_features, normal_label, normal_lengths, normal_sample_ids = normal_batch
                anomaly_features, anomaly_label, anomaly_lengths, anomaly_sample_ids = anomaly_batch
            else:
                normal_features, normal_label, normal_lengths = normal_batch
                anomaly_features, anomaly_label, anomaly_lengths = anomaly_batch
                normal_sample_ids = []
                anomaly_sample_ids = []

            normal_features = normal_features.to(device)
            anomaly_features = anomaly_features.to(device)
            normal_lengths = normal_lengths.to(device)
            anomaly_lengths = anomaly_lengths.to(device)
            visual_features = torch.cat([normal_features, anomaly_features], dim=0)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0)
            label_names = list(normal_label) + list(anomaly_label)
            labels = get_batch_label(label_names, prompt_text, label_map).to(device)
            anomaly_labels = labels[normal_features.shape[0]:]

            optimizer.zero_grad()
            text_features, logits1, logits2 = model(
                visual_features, None, prompt_text, feat_lengths
            )
            if topk_tracker is not None:
                topk_tracker.update_from_batch(
                    epoch + 1,
                    list(normal_sample_ids) + list(anomaly_sample_ids),
                    label_names,
                    labels,
                    feat_lengths,
                    logits1,
                    logits2,
                )
            loss_cls = CLAS2(logits1, labels, feat_lengths, device)
            loss_mil = CLASM(logits2, labels, feat_lengths, device)
            loss_text = text_regularization_loss(text_features, device)
            original_loss = loss_cls + loss_mil + loss_text
            original_loss.backward()

            totals['cls'] += loss_cls.item()
            totals['mil'] += loss_mil.item()
            totals['text'] += loss_text.item()

            loss_transplant = torch.zeros((), device=device)
            loss_erase = torch.zeros((), device=device)
            if augmentation_active:
                normal_count = normal_features.shape[0]
                totals['augmentation_samples'] += anomaly_features.shape[0]
                classifier_scores = torch.sigmoid(logits1[..., 0].detach())
                alignment_scores = 1.0 - F.softmax(logits2.detach(), dim=-1)[..., 0]
                normal_classifier_scores = classifier_scores[:normal_count]
                anomaly_classifier_scores = classifier_scores[normal_count:]
                normal_alignment_scores = alignment_scores[:normal_count]
                anomaly_alignment_scores = alignment_scores[normal_count:]
                augmented_batches = []
                augmented_lengths = []
                inserted_mask = None

                if args.lambda_transplant > 0:
                    transplanted, inserted_mask = build_hard_negative_transplants(
                        normal_features,
                        anomaly_features,
                        normal_lengths,
                        anomaly_lengths,
                        normal_classifier_scores,
                        normal_alignment_scores,
                        anomaly_classifier_scores,
                        anomaly_alignment_scores,
                        args.augmentation_min_segment_length,
                        args.augmentation_max_segment_length,
                        args.transplant_context_width,
                        args.transplant_high_exclusion_ratio,
                        args.transplant_boundary_blend_width,
                    )
                    augmented_batches.append(('transplant', transplanted))
                    augmented_lengths.append(anomaly_lengths)
                    totals['transplant_samples'] += int(inserted_mask.any(dim=1).sum().item())
                    totals['transplant_snippets'] += int(inserted_mask.sum().item())

                if args.lambda_evidence_erasing > 0:
                    erased, _ = build_evidence_erased_batch(
                        anomaly_features,
                        anomaly_lengths,
                        anomaly_classifier_scores,
                        args.augmentation_min_segment_length,
                        args.augmentation_max_segment_length,
                    )
                    augmented_batches.append(('erase', erased))
                    augmented_lengths.append(anomaly_lengths)

                if augmented_batches:
                    batch_size = anomaly_features.shape[0]
                    augmented_visual = torch.cat(
                        [item[1] for item in augmented_batches], dim=0
                    )
                    all_augmented_lengths = torch.cat(augmented_lengths, dim=0)
                    _, augmented_logits1, augmented_logits2 = model(
                        augmented_visual, None, prompt_text, all_augmented_lengths
                    )
                    offset = 0
                    for augmentation_name, _ in augmented_batches:
                        next_offset = offset + batch_size
                        current_logits1 = augmented_logits1[offset:next_offset]
                        current_logits2 = augmented_logits2[offset:next_offset]
                        if augmentation_name == 'transplant':
                            loss_transplant = transplanted_negative_loss(
                                current_logits1,
                                current_logits2,
                                inserted_mask,
                                args.transplant_score_branch,
                                device,
                            )
                        else:
                            loss_erase = CLAS2(
                                current_logits1,
                                anomaly_labels,
                                anomaly_lengths,
                                device,
                            ) + CLASM(
                                current_logits2,
                                anomaly_labels,
                                anomaly_lengths,
                                device,
                            )
                        offset = next_offset

                    augmentation_loss = (
                        args.lambda_transplant * loss_transplant
                        + args.lambda_evidence_erasing * loss_erase
                    )
                    augmentation_loss.backward()

            totals['transplant'] += loss_transplant.item()
            totals['erase'] += loss_erase.item()
            optimizer.step()

        scheduler.step()
        divisor = max(1, num_steps)
        print(
            'epoch:', epoch + 1,
            '| augmentation_active:', augmentation_active,
            '| loss_cls:', totals['cls'] / divisor,
            '| loss_mil:', totals['mil'] / divisor,
            '| loss_text:', totals['text'] / divisor,
            '| loss_transplant:', totals['transplant'] / divisor,
            '| loss_erase:', totals['erase'] / divisor,
            '| transplant_sample_coverage:', (
                totals['transplant_samples'] / max(1, totals['augmentation_samples'])
            ),
            '| supervised_transplant_snippets:', totals['transplant_snippets'],
        )
        if topk_tracker is not None:
            topk_tracker.end_epoch(epoch + 1)

        epoch_path = epoch_checkpoint_dir / f'model_epoch_{epoch + 1:03d}.pth'
        torch.save(model.state_dict(), epoch_path)
        torch.save(model.state_dict(), args.save_cur_path)

        auc, _ = test(
            model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device
        )
        auc = float(auc)
        if auc > best_auc:
            best_auc = auc
            stale_epochs = 0
            torch.save(model.state_dict(), args.model_path)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ap': auc,
                },
                args.checkpoint_path,
            )
            print(f'New best classifier AUC: {best_auc:.6f}')
        else:
            stale_epochs += 1
            print(
                f'No classifier AUC improvement: {stale_epochs}/'
                f'{args.early_stopping_patience}'
            )
            if stale_epochs >= args.early_stopping_patience:
                print('Early stopping triggered.')
                break

    print('Best classifier AUC:', best_auc)
    print('Best model:', args.model_path)


def multi_classifier_regularization_losses(
    head_logits, labels, lengths, temperature, device
):
    """Compute abnormal-only temporal diversity and head-scale balance."""
    diversity_losses = []
    balance_losses = []
    num_heads = head_logits.shape[-1]
    temperature = max(float(temperature), 1e-6)

    for i, length in enumerate(lengths):
        if int(labels[i].argmax().item()) == 0:
            continue
        valid_length = max(1, min(int(length), head_logits.shape[1]))
        valid_logits = head_logits[i, :valid_length].float().transpose(0, 1)
        attention = F.softmax(valid_logits / temperature, dim=-1)

        pair_losses = []
        for first in range(num_heads - 1):
            for second in range(first + 1, num_heads):
                pair_losses.append(
                    F.cosine_similarity(
                        attention[first].unsqueeze(0),
                        attention[second].unsqueeze(0),
                        dim=-1,
                    ).squeeze(0)
                )
        if pair_losses:
            diversity_losses.append(torch.stack(pair_losses).mean())

        score_maps = torch.sigmoid(valid_logits)
        average_map = score_maps.mean(dim=0)
        average_norm = torch.linalg.vector_norm(average_map)
        head_norms = torch.linalg.vector_norm(score_maps, dim=-1)
        balance_losses.append(torch.abs(head_norms - average_norm).mean())

    diversity = (
        torch.stack(diversity_losses).mean()
        if diversity_losses
        else torch.zeros((), device=device)
    )
    balance = (
        torch.stack(balance_losses).mean()
        if balance_losses
        else torch.zeros((), device=device)
    )
    return diversity, balance


def diversity_weight_scale(epoch, warmup_epochs, ramp_epochs):
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))


def train_multi_classifier_epoch_eval(
    model, normal_loader, anomaly_loader, testloader, args, label_map, device
):
    """Train parallel classifier heads from scratch with epoch-level evaluation."""
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    output_paths = [args.model_path, args.checkpoint_path, args.save_cur_path]
    for output_path in output_paths:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epoch_checkpoint_dir = Path(args.epoch_checkpoint_dir)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    aggregate_tracker = create_topk_tracker(args)
    head_trackers = []
    if args.track_topk:
        tracker_root = Path(args.topk_tracking_output_dir)
        for head_index in range(args.num_classifier_heads):
            head_trackers.append(
                TopKSelectionTracker(
                    tracker_root / f'head_{head_index + 1}',
                    branches='classifier',
                    scope=args.topk_tracking_scope,
                )
            )

    aggregate_weight = float(args.multi_head_cls_aggregate_weight)
    if not 0.0 <= aggregate_weight <= 1.0:
        raise ValueError('--multi-head-cls-aggregate-weight must be in [0, 1]')

    best_auc = float('-inf')
    stale_epochs = 0
    num_steps = min(len(normal_loader), len(anomaly_loader))
    print(
        'Multi-classifier config:',
        '| heads:', args.num_classifier_heads,
        '| adapters:', isinstance(model, CLIPVADAdapterMultiClassifier),
        '| adapter_bottleneck:', args.head_adapter_bottleneck,
        '| adapter_dropout:', args.head_adapter_dropout,
        '| adapter_residual_scale:', args.head_adapter_residual_scale,
        '| aggregate: mean raw logits',
        '| aggregate_cls_weight:', aggregate_weight,
        '| lambda_diversity:', args.lambda_head_diversity,
        '| lambda_balance:', args.lambda_head_balance,
        '| temperature:', args.head_diversity_temperature,
        '| warmup_epochs:', args.head_diversity_warmup_epochs,
        '| ramp_epochs:', args.head_diversity_ramp_epochs,
    )

    for epoch in range(args.max_epoch):
        model.train()
        totals = {
            'cls_aggregate': 0.0,
            'cls_individual': 0.0,
            'mil': 0.0,
            'text': 0.0,
            'diversity': 0.0,
            'balance': 0.0,
        }
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        weight_scale = diversity_weight_scale(
            epoch,
            args.head_diversity_warmup_epochs,
            args.head_diversity_ramp_epochs,
        )
        lambda_diversity = float(args.lambda_head_diversity) * weight_scale
        lambda_balance = float(args.lambda_head_balance) * weight_scale

        for _ in range(num_steps):
            normal_features, normal_label, normal_lengths, normal_ids = unpack_training_batch(
                next(normal_iter), aggregate_tracker
            )
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_ids = unpack_training_batch(
                next(anomaly_iter), aggregate_tracker
            )
            visual = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            label_names = list(normal_label) + list(anomaly_label)
            labels = get_batch_label(label_names, prompt_text, label_map).to(device)
            sample_ids = normal_ids + anomaly_ids

            text_features, logits1, logits2, head_logits = model(
                visual,
                None,
                prompt_text,
                lengths,
                return_head_logits=True,
            )
            if aggregate_tracker is not None:
                aggregate_tracker.update_from_batch(
                    epoch + 1,
                    sample_ids,
                    label_names,
                    labels,
                    lengths,
                    logits1,
                    logits2,
                )
                for head_index, tracker in enumerate(head_trackers):
                    tracker.update_from_batch(
                        epoch + 1,
                        sample_ids,
                        label_names,
                        labels,
                        lengths,
                        head_logits[..., head_index:head_index + 1],
                        logits2,
                    )

            loss_cls_aggregate = CLAS2(logits1, labels, lengths, device)
            individual_losses = [
                CLAS2(
                    head_logits[..., index:index + 1], labels, lengths, device
                )
                for index in range(args.num_classifier_heads)
            ]
            loss_cls_individual = torch.stack(individual_losses).mean()
            loss_cls = (
                aggregate_weight * loss_cls_aggregate
                + (1.0 - aggregate_weight) * loss_cls_individual
            )
            loss_mil = CLASM(logits2, labels, lengths, device)
            loss_text = text_regularization_loss(text_features, device)
            loss_diversity, loss_balance = multi_classifier_regularization_losses(
                head_logits,
                labels,
                lengths,
                args.head_diversity_temperature,
                device,
            )
            loss = (
                loss_cls
                + loss_mil
                + loss_text
                + lambda_diversity * loss_diversity
                + lambda_balance * loss_balance
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals['cls_aggregate'] += loss_cls_aggregate.item()
            totals['cls_individual'] += loss_cls_individual.item()
            totals['mil'] += loss_mil.item()
            totals['text'] += loss_text.item()
            totals['diversity'] += loss_diversity.item()
            totals['balance'] += loss_balance.item()

        scheduler.step()
        divisor = max(1, num_steps)
        print(
            'epoch:', epoch + 1,
            '| loss_cls_aggregate:', totals['cls_aggregate'] / divisor,
            '| loss_cls_individual:', totals['cls_individual'] / divisor,
            '| loss_mil:', totals['mil'] / divisor,
            '| loss_text:', totals['text'] / divisor,
            '| loss_diversity:', totals['diversity'] / divisor,
            '| loss_balance:', totals['balance'] / divisor,
            '| lambda_diversity:', lambda_diversity,
            '| lambda_balance:', lambda_balance,
        )

        if aggregate_tracker is not None:
            aggregate_tracker.end_epoch(epoch + 1)
            for tracker in head_trackers:
                tracker.end_epoch(epoch + 1)

        epoch_path = epoch_checkpoint_dir / f'model_epoch_{epoch + 1:03d}.pth'
        torch.save(model.state_dict(), epoch_path)
        torch.save(model.state_dict(), args.save_cur_path)

        auc, _ = test(
            model,
            testloader,
            args.visual_length,
            prompt_text,
            gt,
            gtsegments,
            gtlabels,
            device,
        )
        auc = float(auc)
        if auc > best_auc:
            best_auc = auc
            stale_epochs = 0
            torch.save(model.state_dict(), args.model_path)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ap': auc,
                },
                args.checkpoint_path,
            )
            print(f'New best multi-classifier AUC: {best_auc:.6f}')
        else:
            diversity_fully_active = (
                epoch + 1
                >= args.head_diversity_warmup_epochs
                + max(1, args.head_diversity_ramp_epochs)
            )
            if diversity_fully_active:
                stale_epochs += 1
                print(
                    f'No multi-classifier AUC improvement: {stale_epochs}/'
                    f'{args.early_stopping_patience}'
                )
                if (
                    args.early_stopping_patience > 0
                    and stale_epochs >= args.early_stopping_patience
                ):
                    print('Early stopping triggered.')
                    break
            else:
                stale_epochs = 0
                print('Early stopping is inactive during diversity warmup/ramp.')

    print('Best multi-classifier AUC:', best_auc)
    print('Best model:', args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    return_training_sample_id = bool(args.track_topk)
    normal_dataset = UCFDataset(
        args.visual_length,
        args.train_list,
        False,
        label_map,
        True,
        return_sample_id=return_training_sample_id,
    )
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(
        args.visual_length,
        args.train_list,
        False,
        label_map,
        False,
        return_sample_id=return_training_sample_id,
    )
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.training_mode == 'adapter_multi_classifier_epoch_eval':
        model = CLIPVADAdapterMultiClassifier(
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
            num_classifier_heads=args.num_classifier_heads,
            adapter_bottleneck=args.head_adapter_bottleneck,
            adapter_dropout=args.head_adapter_dropout,
            adapter_residual_scale=args.head_adapter_residual_scale,
        )
    elif args.training_mode == 'multi_classifier_epoch_eval':
        model = CLIPVADMultiClassifier(
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
            num_classifier_heads=args.num_classifier_heads,
        )
    else:
        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    if args.training_mode in {
        'multi_classifier_epoch_eval',
        'adapter_multi_classifier_epoch_eval',
    }:
        train_multi_classifier_epoch_eval(
            model, normal_loader, anomaly_loader, test_loader, args, label_map, device
        )
    elif args.training_mode == 'abnormal_topk_epoch_eval':
        train_abnormal_topk_epoch_eval(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
    elif args.training_mode in {'ranking_margin_epoch_eval', 'ranking_hard_pair_epoch_eval'}:
        train_ranking_margin_epoch_eval(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
    elif args.training_mode == 'transplant_erasing_epoch_eval':
        train_transplant_erasing_epoch_eval(
            model, normal_loader, anomaly_loader, test_loader, args, label_map, device
        )
    else:
        train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
