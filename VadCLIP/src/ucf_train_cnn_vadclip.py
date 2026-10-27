import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model_cnn_vadclip import CLIPVADCNN
from ucf_cnn_common import (
    TRAIN_LABEL_MAP,
    UCFCNNFeatureDataset,
    aggregate_video_logits,
    alignment_mil_loss,
    setup_seed,
    text_regularization_loss,
)
from utils.tools import get_batch_label, get_prompt_text


def build_parser():
    parser = argparse.ArgumentParser(description="Train VadCLIP with a temporal CNN classifier")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--kernel-size", type=int, choices=(1, 3, 5, 7, 9), default=3)
    parser.add_argument("--score-mode", choices=("scoring_head", "kernel_response"), default="scoring_head")
    parser.add_argument("--residual-mode", choices=("on", "off"), default="on")
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--aggregation", choices=("topk", "max"), default="topk")
    parser.add_argument("--lambda-alignment", type=float, default=1.0)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Optional independent seed for the two training DataLoaders")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--epoch-checkpoint-dir", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--results-dir", required=True)
    return parser


def build_model(args, device):
    return CLIPVADCNN(
        14, 512, args.visual_length, 512, 1, 2, 8, 10, 10, device,
        kernel_size=args.kernel_size, score_mode=args.score_mode,
        projection_width=args.projection_width, hidden_width=args.cnn_channels,
        use_residual=args.residual_mode == "on",
    )


def train(args):
    if args.lambda_alignment < 0 or args.lambda_text < 0:
        raise ValueError("Loss weights must be nonnegative")
    if args.max_epoch < 1 or args.batch_size < 1:
        raise ValueError("max_epoch and batch_size must be positive")
    if args.projection_width < 1 or args.cnn_channels < 1:
        raise ValueError("projection_width and cnn_channels must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("VadCLIP's distance graph requires CUDA. Choose a Colab GPU runtime.")
    device = torch.device("cuda")
    setup_seed(args.seed)
    checkpoint_dir = Path(args.epoch_checkpoint_dir)
    result_dir = Path(args.results_dir)
    output_path = Path(args.output_model_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    normal = UCFCNNFeatureDataset(args.feature_root, args.train_list, args.visual_length, train=True, normal=True)
    abnormal = UCFCNNFeatureDataset(args.feature_root, args.train_list, args.visual_length, train=True, normal=False)
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, drop_last=True,
                         num_workers=args.num_workers, pin_memory=True)
    normal_generator = torch.Generator().manual_seed(args.data_seed) if args.data_seed is not None else None
    abnormal_generator = torch.Generator().manual_seed(args.data_seed + 1) if args.data_seed is not None else None
    normal_loader = DataLoader(normal, generator=normal_generator, **loader_kwargs)
    abnormal_loader = DataLoader(abnormal, generator=abnormal_generator, **loader_kwargs)
    steps = min(len(normal_loader), len(abnormal_loader))
    if steps == 0:
        raise ValueError("Each class group needs at least one full batch")

    model = build_model(args, device).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_rate)
    prompt_text = get_prompt_text(TRAIN_LABEL_MAP)
    history = []
    print(f"Training from scratch: {args.score_mode}, projection={args.projection_width}, channels={args.cnn_channels}, kernel={args.kernel_size}, residual={args.residual_mode}, aggregation={args.aggregation}")
    print(f"Feature views: {len(normal)} Normal + {len(abnormal)} abnormal; {steps} steps/epoch")
    print(f"Loss weights: video_BCE=1, alignment_MIL={args.lambda_alignment}, text_reg={args.lambda_text}")

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        sums = dict(loss_video_bce=0.0, loss_alignment=0.0, loss_text=0.0, loss_total=0.0)
        normal_iter = iter(normal_loader)
        abnormal_iter = iter(abnormal_loader)
        for _ in range(steps):
            n_features, n_labels, n_lengths = next(normal_iter)
            a_features, a_labels, a_lengths = next(abnormal_iter)
            visual = torch.cat((n_features, a_features), dim=0).to(device, non_blocking=True)
            lengths = torch.cat((n_lengths, a_lengths)).to(device)
            names = list(n_labels) + list(a_labels)
            labels = get_batch_label(names, prompt_text, TRAIN_LABEL_MAP).to(device)
            binary_labels = (1.0 - labels[:, 0]).float()

            use_alignment = bool(args.lambda_alignment or args.lambda_text)
            text_features, logits1, logits2 = model(
                visual, None, prompt_text, lengths, compute_alignment=use_alignment
            )
            video_logits = aggregate_video_logits(logits1, lengths, args.aggregation)
            loss_video = F.binary_cross_entropy_with_logits(video_logits, binary_labels)
            loss_alignment = alignment_mil_loss(logits2, labels, lengths) if args.lambda_alignment else loss_video.new_zeros(())
            loss_text = text_regularization_loss(text_features) if args.lambda_text else loss_video.new_zeros(())
            loss = loss_video + args.lambda_alignment * loss_alignment + args.lambda_text * loss_text
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CNN VadCLIP loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in zip(sums, (loss_video, loss_alignment, loss_text, loss)):
                sums[key] += float(value.detach())

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({key: value / steps for key, value in sums.items()})
        history.append(row)
        print("epoch {}/{} | BCE={:.5f} | align={:.5f} | text={:.5f} | total={:.5f}".format(
            epoch, args.max_epoch, row["loss_video_bce"], row["loss_alignment"],
            row["loss_text"], row["loss_total"]
        ), flush=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}.pth")
        with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        scheduler.step()

    torch.save(model.state_dict(), output_path)
    print(f"Final epoch model: {output_path}")


if __name__ == "__main__":
    train(build_parser().parse_args())
