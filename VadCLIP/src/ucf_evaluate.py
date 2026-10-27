import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model import CLIPVAD
from model_description import CLIPVADDescription
from model_multi_prompt import CLIPVADMultiPrompt
from model_prompt_replacement import CLIPVADPromptReplacement
from model_prompt_residual import CLIPVADPromptResidual
from model_multi_prompt_residual import CLIPVADMultiPromptResidual
from model_temporal_proposal import CLIPVADTemporalProposal
from utils.prompt_replacement import build_compact_prompt_groups, build_multi_prompt_groups, build_multi_prompt_weights
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP


class UCFRelativeTestDataset(torch.utils.data.Dataset):
    def __init__(self, clip_dim, file_path, feature_root):
        self.clip_dim = clip_dim
        self.feature_root = Path(feature_root)
        with open(file_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.validate_feature_paths()

    def __len__(self):
        return len(self.rows)

    def validate_feature_paths(self):
        missing = []
        for row in self.rows:
            feature_path = self.feature_root / row["path"]
            if not feature_path.exists():
                missing.append((row["video_id"], row["label"], row["path"]))

        if missing:
            preview = "\n".join(
                f"  {video_id},{label},{path}" for video_id, label, path in missing[:50]
            )
            extra = "" if len(missing) <= 50 else f"\n  ... and {len(missing) - 50} more"
            raise FileNotFoundError(
                "Missing feature files referenced by the test list.\n"
                f"feature_root: {self.feature_root}\n"
                f"missing_files: {len(missing)}\n"
                "First missing entries:\n"
                f"{preview}{extra}"
            )

    def __getitem__(self, index):
        import utils.tools as tools

        row = self.rows[index]
        feature_path = self.feature_root / row["path"]
        clip_feature = np.load(feature_path)
        clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
        return torch.tensor(clip_feature), row["label"], clip_length, row["video_id"]


def build_model(args, model_type, device):
    if model_type == "description":
        model_cls = CLIPVADDescription
    elif model_type == "multi_prompt":
        model_cls = CLIPVADMultiPrompt
    elif model_type == "multi_prompt_residual":
        model_cls = CLIPVADMultiPromptResidual
    elif model_type == "prompt_replacement":
        model_cls = CLIPVADPromptReplacement
    elif model_type == "prompt_residual":
        model_cls = CLIPVADPromptResidual
    elif model_type == "temporal_proposal":
        return CLIPVADTemporalProposal(
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
    else:
        model_cls = CLIPVAD
    return model_cls(
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


def load_state_dict(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate_model(model, dataloader, args, label_map, device, prompt_text=None, prompt_weights=None):
    if prompt_text is None:
        prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model.to(device)
    model.eval()

    classifier_scores = []
    alignment_scores = []
    classifier_scores_by_video = []
    alignment_scores_by_video = []
    gt_by_video = []
    labels_by_video = []
    element_logits2_stack = []
    gt_cursor = 0

    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            length = int(item[2])
            len_cur = length
            if len_cur < args.visual_length:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)
            length_remaining = length
            lengths = torch.zeros(int(length / args.visual_length) + 1)
            for j in range(int(length / args.visual_length) + 1):
                if j == 0 and length_remaining < args.visual_length:
                    lengths[j] = length_remaining
                elif j == 0 and length_remaining > args.visual_length:
                    lengths[j] = args.visual_length
                    length_remaining -= args.visual_length
                elif length_remaining > args.visual_length:
                    lengths[j] = args.visual_length
                    length_remaining -= args.visual_length
                else:
                    lengths[j] = length_remaining
            lengths = lengths.to(int)

            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            if prompt_weights is not None:
                _, logits1, logits2 = model(
                    visual,
                    padding_mask,
                    prompt_text,
                    lengths,
                    prompt_weights=prompt_weights,
                )
            else:
                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            classifier_prob = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).detach().cpu().numpy()
            alignment_prob = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)).detach().cpu().numpy()

            classifier_repeated = np.repeat(classifier_prob, 16)
            alignment_repeated = np.repeat(alignment_prob, 16)
            gt_slice = gt[gt_cursor: gt_cursor + len(classifier_repeated)]
            gt_cursor += len(classifier_repeated)

            classifier_scores.append(classifier_repeated)
            alignment_scores.append(alignment_repeated)
            classifier_scores_by_video.append(classifier_repeated)
            alignment_scores_by_video.append(alignment_repeated)
            gt_by_video.append(gt_slice)
            labels_by_video.append(label)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    classifier_scores = np.concatenate(classifier_scores)
    alignment_scores = np.concatenate(alignment_scores)
    if len(classifier_scores) != len(gt):
        raise ValueError(f"Score length {len(classifier_scores)} does not match gt length {len(gt)}")

    classifier_auc = roc_auc_score(gt, classifier_scores) * 100
    classifier_ap = average_precision_score(gt, classifier_scores) * 100
    alignment_auc = roc_auc_score(gt, alignment_scores) * 100
    alignment_ap = average_precision_score(gt, alignment_scores) * 100

    if alignment_auc >= classifier_auc:
        best_scores_by_video = alignment_scores_by_video
        best_auc = alignment_auc
        best_branch = "alignment"
    else:
        best_scores_by_video = classifier_scores_by_video
        best_auc = classifier_auc
        best_branch = "classifier"

    abnormal_gt = []
    abnormal_scores = []
    for label, gt_video, score_video in zip(labels_by_video, gt_by_video, best_scores_by_video):
        if label != "Normal":
            abnormal_gt.append(gt_video)
            abnormal_scores.append(score_video)
    ano_auc = roc_auc_score(np.concatenate(abnormal_gt), np.concatenate(abnormal_scores)) * 100

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    avg_map = float(np.mean(dmap))

    metrics = {
        "classifier_auc": float(classifier_auc),
        "classifier_ap": float(classifier_ap),
        "alignment_auc": float(alignment_auc),
        "alignment_ap": float(alignment_ap),
        "best_auc": float(best_auc),
        "best_branch": best_branch,
        "ano_auc": float(ano_auc),
        "avg_mAP": avg_map,
    }
    for threshold, value in zip(iou, dmap):
        metrics[f"mAP@{threshold:.1f}"] = float(value)
    return metrics


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"  classifier_auc: {metrics['classifier_auc']:.2f}")
    print(f"  classifier_ap:  {metrics['classifier_ap']:.2f}")
    print(f"  alignment_auc:  {metrics['alignment_auc']:.2f}")
    print(f"  alignment_ap:   {metrics['alignment_ap']:.2f}")
    print(f"  best_auc:       {metrics['best_auc']:.2f} ({metrics['best_branch']})")
    print(f"  ano_auc:        {metrics['ano_auc']:.2f}")
    print("  mAP is computed from alignment branch logits2, matching existing VadCLIP code.")
    for key in ["mAP@0.1", "mAP@0.2", "mAP@0.3", "mAP@0.4", "mAP@0.5", "avg_mAP"]:
        print(f"  {key}: {metrics[key]:.2f}")


def print_comparison(baseline_metrics, description_metrics):
    columns = ["best_auc", "ano_auc", "mAP@0.1", "mAP@0.2", "mAP@0.3", "mAP@0.4", "mAP@0.5", "avg_mAP"]
    print("\nComparison")
    print("Model," + ",".join(columns))
    print("Baseline," + ",".join(f"{baseline_metrics[col]:.2f}" for col in columns))
    print("Description," + ",".join(f"{description_metrics[col]:.2f}" for col in columns))
    print("Delta," + ",".join(f"{description_metrics[col] - baseline_metrics[col]:+.2f}" for col in columns))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VadCLIP on UCF-Crime using paper-style metrics.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--baseline-model-path", default="../../model_ucf.pth")
    parser.add_argument("--description-model-path", default=None)
    parser.add_argument(
        "--description-model-type",
        default="description",
        choices=[
            "baseline",
            "description",
            "prompt_replacement",
            "prompt_residual",
            "multi_prompt",
            "multi_prompt_residual",
            "temporal_proposal",
        ],
    )
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")

    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--num-proposals", default=8, type=int)
    parser.add_argument("--proposal-heads", default=4, type=int)
    parser.add_argument("--proposal-min-width", default=0.02, type=float)
    parser.add_argument("--proposal-max-width", default=0.50, type=float)
    parser.add_argument("--proposal-initial-width", default=0.15, type=float)
    parser.add_argument("--mask-temperature", default=0.02, type=float)
    parser.add_argument("--context-scale", default=0.25, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--prompt-replacement-mode", default="manual_caption", choices=["prototype_only", "class_plus_prototype", "manual_caption"])
    parser.add_argument("--prototype-json", default="../../code/ucf_weighted_class_prompts.json")
    parser.add_argument("--prompt-weights-json", default=None)
    parser.add_argument("--prototype-source", default="manual", choices=["manual", "compact", "long_event_only"])
    parser.add_argument(
        "--prompt-mode",
        default="manual_caption",
        choices=["manual_caption", "class_plus_prototype", "prototype_only"],
    )
    parser.add_argument(
        "--compact-prototype-types",
        default=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
        nargs="+",
        choices=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
    )
    parser.add_argument(
        "--long-prototype-types",
        default=["core_action", "actor_object_interaction", "temporal_progression"],
        nargs="+",
        choices=[
            "core_action",
            "actor_object_interaction",
            "temporal_progression",
            "scene_context",
            "hard_negative_distinction",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("ucf_evaluate.py expects --batch-size 1 so frame scores stay aligned with UCF ground truth.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = {
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
    prompt_replacement_label_map = {
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

    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Test videos:", len(dataset))

    baseline_model = load_state_dict(build_model(args, "baseline", device), args.baseline_model_path, device)
    baseline_metrics = evaluate_model(baseline_model, dataloader, args, label_map, device)
    print_metrics("Baseline", baseline_metrics)

    if args.description_model_path:
        description_prompt_text = None
        description_prompt_weights = None
        if args.description_model_type in ("prompt_replacement", "prompt_residual"):
            prompt_mode = args.prompt_mode or args.prompt_replacement_mode
            if args.prototype_source == "compact":
                description_prompt_text = build_compact_prompt_groups(
                    args.prototype_json,
                    prompt_replacement_label_map,
                    prompt_mode if prompt_mode in ("prototype_only", "class_plus_prototype") else "class_plus_prototype",
                    args.compact_prototype_types,
                )
                description_prompt_weights = None
            else:
                description_prompt_text = build_multi_prompt_groups(
                    args.prototype_json,
                    prompt_replacement_label_map,
                    args.prototype_source,
                    args.compact_prototype_types,
                    args.long_prototype_types,
                    prompt_mode,
                )
                description_prompt_weights = build_multi_prompt_weights(
                    args.prompt_weights_json,
                    prompt_replacement_label_map,
                    device,
                )
        elif args.description_model_type in ("multi_prompt", "multi_prompt_residual"):
            description_prompt_text = build_multi_prompt_groups(
                args.prototype_json,
                prompt_replacement_label_map,
                args.prototype_source,
                args.compact_prototype_types,
                args.long_prototype_types,
                args.prompt_mode,
            )
            description_prompt_weights = build_multi_prompt_weights(
                args.prompt_weights_json,
                prompt_replacement_label_map,
                device,
            )
        description_model = build_model(args, args.description_model_type, device)
        if args.description_model_type == "prompt_residual":
            # Ensure p_c exists before loading Delta / saved buffers.
            description_model.to(device)
            description_model.init_class_prototypes(
                description_prompt_text,
                prompt_weights=description_prompt_weights,
            )
        if args.description_model_type == "multi_prompt_residual":
            description_model.to(device)
            description_model.init_prompt_embeddings(description_prompt_text)
        description_model = load_state_dict(
            description_model,
            args.description_model_path,
            device,
        )
        if args.description_model_type == "prompt_residual":
            description_model.prototypes_initialized = True
        if args.description_model_type == "multi_prompt_residual":
            description_model.prompts_initialized = True
        description_metrics = evaluate_model(
            description_model,
            dataloader,
            args,
            label_map,
            device,
            prompt_text=description_prompt_text,
            prompt_weights=description_prompt_weights,
        )
        print_metrics("Description-guided", description_metrics)
        print_comparison(baseline_metrics, description_metrics)


if __name__ == "__main__":
    main()
