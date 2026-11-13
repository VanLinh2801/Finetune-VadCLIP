import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model_cnn_vadclip import CLIPVADCNN, validate_cnn_kernel
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
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Positive temporal size; single_conv supports odd and even sizes")
    parser.add_argument("--cnn-architecture", choices=("two_block", "single_conv"), default="two_block")
    parser.add_argument("--score-mode", choices=("scoring_head", "kernel_response", "competitive_response", "competitive_response_gelu"), default="scoring_head")
    parser.add_argument("--residual-mode", choices=("on", "off"), default="on")
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--aggregation", choices=("topk", "max"), default="topk")
    parser.add_argument("--lambda-alignment", type=float, default=1.0)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Optional independent seed for the two training DataLoaders")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--evaluate-every-epoch", action="store_true")
    parser.add_argument("--evaluate-epochs", type=int, nargs="+", default=[],
                        help="Evaluate only these epochs unless evaluate-every-epoch is enabled")
    for name in ("test-list", "gt-path", "gt-segment-path", "gt-label-path",
                 "evaluate-dir", "analyze-dir"):
        parser.add_argument("--" + name)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many epochs without train-loss improvement (0 disables)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--epoch-checkpoint-dir", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--results-dir", required=True)
    return parser


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


def build_model(args, device):
    return CLIPVADCNN(
        14, 512, args.visual_length, 512, 1, 2, 8, 10, 10, device,
        kernel_size=args.kernel_size, score_mode=args.score_mode,
        projection_width=args.projection_width, hidden_width=args.cnn_channels,
        use_residual=args.residual_mode == "on",
        cnn_architecture=args.cnn_architecture,
    )


def train(args):
    validate_cnn_kernel(args.kernel_size, args.cnn_architecture)
    if args.grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be nonnegative")
    if any(epoch < 1 or epoch > args.max_epoch for epoch in args.evaluate_epochs):
        raise ValueError("evaluate-epochs must be between 1 and max-epoch")
    if args.evaluate_every_epoch or args.evaluate_epochs:
        for name in ("test_list", "gt_path", "gt_segment_path", "gt_label_path"):
            if not getattr(args, name) or not Path(getattr(args, name)).is_file():
                raise ValueError(f"Epoch evaluation requires an existing {name}")
        if not args.evaluate_dir or not args.analyze_dir:
            raise ValueError("Epoch evaluation requires evaluate-dir and analyze-dir")
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.data_seed is None:
            args.data_seed = args.seed
    if args.lambda_alignment < 0 or args.lambda_text < 0:
        raise ValueError("Loss weights must be nonnegative")
    if args.max_epoch < 1 or args.batch_size < 1:
        raise ValueError("max_epoch and batch_size must be positive")
    if args.projection_width < 1 or args.cnn_channels < 1:
        raise ValueError("projection_width and cnn_channels must be positive")
    if args.score_mode in ("competitive_response", "competitive_response_gelu") and (args.cnn_architecture != "single_conv" or args.cnn_channels % 2):
        raise ValueError("competitive_response requires single_conv and an even channel count")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and min_delta must be nonnegative")
    if not 1 <= args.min_epochs <= args.max_epoch:
        raise ValueError("min_epochs must be between 1 and max_epoch")
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
    environment = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                       deterministic=torch.are_deterministic_algorithms_enabled(),
                       python=sys.version)
    (result_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print("Runtime:", environment, flush=True)

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
    evaluation_history = []
    stopper = TrainingLossStopper(args.early_stopping_patience, args.early_stopping_min_delta, args.min_epochs)
    best_path = result_dir / "best_train_loss.pth"
    stopped_early = False
    print(f"Training from scratch: architecture={args.cnn_architecture}, {args.score_mode}, projection={args.projection_width if args.cnn_architecture == 'two_block' else 'none'}, channels={args.cnn_channels}, kernel={args.kernel_size}, residual={args.residual_mode if args.cnn_architecture == 'two_block' else 'none'}, aggregation={args.aggregation}")
    print(f"Feature views: {len(normal)} Normal + {len(abnormal)} abnormal; {steps} steps/epoch")
    if args.score_mode in ("competitive_response", "competitive_response_gelu"):
        activation = "GELU before channel max" if args.score_mode == "competitive_response_gelu" else "no GELU"
        print(f"Kernel competition: {args.cnn_channels // 2} abnormal + {args.cnn_channels // 2} normal; logit = max(abnormal) - max(normal) + bias; {activation}")
    print(f"Loss weights: video_BCE=1, alignment_MIL={args.lambda_alignment}, text_reg={args.lambda_text}")
    print(f"Stability: seed={args.seed}, data_seed={args.data_seed}, deterministic={args.deterministic}, "
          f"grad_clip_norm={args.grad_clip_norm}, lr={args.lr}, milestones={args.scheduler_milestones}; "
          f"epoch evaluation={'all' if args.evaluate_every_epoch else args.evaluate_epochs}", flush=True)
    if args.early_stopping_patience:
        print(f"Early stopping: train loss, patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}, min_epochs={args.min_epochs}; test metrics are not used")

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
            if args.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
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
        history.append(row)
        print("epoch {}/{} | BCE={:.5f} | align={:.5f} | text={:.5f} | total={:.5f}".format(
            epoch, args.max_epoch, row["loss_video_bce"], row["loss_alignment"],
            row["loss_text"], row["loss_total"]
        ), flush=True)
        if args.early_stopping_patience:
            print(f"  best train loss={stopper.best_loss:.5f} at epoch {stopper.best_epoch}; no significant improvement={stopper.stale_epochs}/{args.early_stopping_patience}", flush=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}.pth")
        with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        scheduler.step()
        if args.evaluate_every_epoch or epoch in args.evaluate_epochs:
            metrics = evaluate_epoch(args, epoch, checkpoint_dir, result_dir)
            evaluation_history.append({"epoch": epoch, **{key: metrics[key] for key in (
                "classifier_auc", "classifier_ap", "alignment_auc", "alignment_ap", "ano_auc", "avg_mAP")}})
            with (result_dir / "evaluation_history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(evaluation_history[0]))
                writer.writeheader()
                writer.writerows(evaluation_history)
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
    summary = {
        "monitor": "train_epoch_loss" if args.early_stopping_patience else "disabled",
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "best_epoch": stopper.best_epoch if args.early_stopping_patience else len(history),
        "best_train_loss": stopper.best_loss if args.early_stopping_patience else None,
        "best_checkpoint": str(best_path) if args.early_stopping_patience else str(checkpoint_dir / f"epoch_{len(history):03d}.pth"),
        "output_model_path": str(output_path),
    }
    (result_dir / "early_stopping.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Training summary:", json.dumps(summary), flush=True)


def evaluate_epoch(args, epoch, checkpoint_dir, result_dir):
    # A separate process leaves training RNG, model mode and gradients untouched.
    tag = f"epoch_{epoch:03d}"
    evaluate_dir = Path(args.evaluate_dir) / tag
    options = dict(model_type="cnn", model_path=str(checkpoint_dir / f"{tag}.pth"),
                   train_config_json=str(result_dir / "config.json"),
                   evaluate_dir=str(evaluate_dir), analyze_dir=str(Path(args.analyze_dir) / tag))
    for key in ("feature_root", "test_list", "gt_path", "gt_segment_path", "gt_label_path",
                "cnn_architecture", "score_mode", "kernel_size", "cnn_channels",
                "projection_width", "residual_mode", "visual_length"):
        options[key] = getattr(args, key)
    command = [sys.executable, "-u", str(Path(__file__).with_name("ucf_evaluate_cnn_vadclip.py"))]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    torch.cuda.empty_cache()
    print(f"Evaluating {tag} (monitoring only, no test-based model selection)", flush=True)
    subprocess.run(command, check=True)
    return json.loads((evaluate_dir / "metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    train(build_parser().parse_args())
