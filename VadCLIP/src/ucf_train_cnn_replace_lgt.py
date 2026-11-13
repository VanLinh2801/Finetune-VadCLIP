import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model_cnn_replace_lgt import CLIPVADCNNReplaceLGT
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
    parser = argparse.ArgumentParser(
        description="Train VadCLIP with Temporal CNN replacing LGT-Adapter"
    )
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--kernel-size", type=int, choices=(1, 3, 5, 7, 9), default=3)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--score-mode", choices=("scoring_head", "kernel_response"), default="scoring_head")
    parser.add_argument("--aggregation", choices=("topk", "max"), default="topk")
    parser.add_argument("--lambda-alignment", type=float, default=1.0)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--epoch-checkpoint-dir", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--results-dir", required=True)
    return parser


def build_model(args, device):
    return CLIPVADCNNReplaceLGT(
        14, 512, args.visual_length, 512, 1, 2, 8, 10, 10, device,
        kernel_size=args.kernel_size,
        score_mode=args.score_mode,
        hidden_width=args.hidden_width,
        num_blocks=args.num_blocks,
    )


class TrainingLossStopper:
    def __init__(self, patience, min_delta, min_epochs):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_loss = float("inf")
        self.best_epoch = None
        self.reference_loss = float("inf")
        self.stale_epochs = 0

    def update(self, epoch, loss):
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.best_epoch = epoch
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved

    def should_stop(self, epoch):
        return self.patience > 0 and epoch >= self.min_epochs and self.stale_epochs >= self.patience


def train(args):
    if args.lambda_alignment < 0 or args.lambda_text < 0:
        raise ValueError("Loss weights must be nonnegative")
    if args.max_epoch < 1 or args.batch_size < 1:
        raise ValueError("max_epoch and batch_size must be positive")
    if args.num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")
    if args.hidden_width < 1:
        raise ValueError("hidden_width must be positive")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and min_delta must be nonnegative")
    if not 1 <= args.min_epochs <= args.max_epoch:
        raise ValueError("min_epochs must be between 1 and max_epoch")
    if not torch.cuda.is_available():
        raise RuntimeError("Choose a Colab GPU runtime before training.")
    device = torch.device("cuda")
    setup_seed(args.seed)
    checkpoint_dir = Path(args.epoch_checkpoint_dir)
    result_dir = Path(args.results_dir)
    output_path = Path(args.output_model_path)
    best_path = checkpoint_dir / "best.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    normal = UCFCNNFeatureDataset(args.feature_root, args.train_list, args.visual_length, train=True, normal=True)
    abnormal = UCFCNNFeatureDataset(args.feature_root, args.train_list, args.visual_length, train=True, normal=False)
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, drop_last=True,
                         num_workers=args.num_workers, pin_memory=True)
    normal_loader = DataLoader(normal, **loader_kwargs)
    abnormal_loader = DataLoader(abnormal, **loader_kwargs)
    steps = min(len(normal_loader), len(abnormal_loader))
    if steps == 0:
        raise ValueError("Each class group needs at least one full batch")

    model = build_model(args, device).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_rate)
    prompt_text = get_prompt_text(TRAIN_LABEL_MAP)
    history = []
    stopper = TrainingLossStopper(
        args.early_stopping_patience, args.early_stopping_min_delta, args.min_epochs
    )
    stopped_early = False
    print(
        "Training from scratch (CNN replaces LGT): "
        f"{args.score_mode}, kernel={args.kernel_size}, blocks={args.num_blocks}, "
        f"hidden={args.hidden_width}, aggregation={args.aggregation}"
    )
    print(f"Feature views: {len(normal)} Normal + {len(abnormal)} abnormal; {steps} steps/epoch")
    print(f"Loss weights: video_BCE=1, alignment_MIL={args.lambda_alignment}, text_reg={args.lambda_text}")
    print(
        f"Early stopping on train loss: patience={args.early_stopping_patience}, "
        f"min_delta={args.early_stopping_min_delta}, min_epochs={args.min_epochs}; "
        "test metrics are not used"
    )

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
            loss_alignment = (
                alignment_mil_loss(logits2, labels, lengths)
                if args.lambda_alignment else loss_video.new_zeros(())
            )
            loss_text = (
                text_regularization_loss(text_features)
                if args.lambda_text else loss_video.new_zeros(())
            )
            loss = loss_video + args.lambda_alignment * loss_alignment + args.lambda_text * loss_text
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CNN-replace-LGT loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in zip(sums, (loss_video, loss_alignment, loss_text, loss)):
                sums[key] += float(value.detach())

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({key: value / steps for key, value in sums.items()})
        if args.early_stopping_patience:
            if stopper.update(epoch, row["loss_total"]):
                torch.save(model.state_dict(), best_path)
            row["best_train_loss"] = stopper.best_loss
            row["epochs_without_improvement"] = stopper.stale_epochs
        print("epoch {}/{} | BCE={:.5f} | align={:.5f} | text={:.5f} | total={:.5f}".format(
            epoch, args.max_epoch, row["loss_video_bce"], row["loss_alignment"],
            row["loss_text"], row["loss_total"]
        ), flush=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}.pth")

        if args.early_stopping_patience:
            print(
                f"  best train loss={stopper.best_loss:.5f} at epoch {stopper.best_epoch}; "
                f"no significant improvement={stopper.stale_epochs}/{args.early_stopping_patience}",
                flush=True,
            )

        history.append(row)
        with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        scheduler.step()
        if args.early_stopping_patience:
            if optimizer.param_groups[0]["lr"] < row["lr"]:
                stopper.stale_epochs = 0
                print("LR reduced; reset early-stopping patience", flush=True)
            if stopper.should_stop(epoch):
                stopped_early = True
                print(f"Early stopping at epoch {epoch}; best train loss at epoch {stopper.best_epoch}", flush=True)
                break

    if args.early_stopping_patience:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    torch.save(model.state_dict(), output_path)
    print(f"Saved model: {output_path}")

    summary = {
        "monitor": "train_epoch_loss" if args.early_stopping_patience else "disabled",
        "best_epoch": stopper.best_epoch if args.early_stopping_patience else len(history),
        "best_train_loss": stopper.best_loss if args.early_stopping_patience else None,
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "best_checkpoint": str(best_path) if args.early_stopping_patience else str(checkpoint_dir / f"epoch_{len(history):03d}.pth"),
        "output_model_path": str(output_path),
    }
    (result_dir / "early_stopping.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Early stopping summary:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    train(build_parser().parse_args())
