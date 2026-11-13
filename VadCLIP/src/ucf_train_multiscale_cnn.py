"""From-scratch LGT + multiscale video CNN with optional FC L1."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model_multiscale_cnn import MultiscaleCNN
from ucf_cnn_common import UCFCNNFeatureDataset, setup_seed
from ucf_evaluate_multiscale_cnn import write_csv


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-root", "train-list", "test-list", "gt-path", "results-dir",
                 "evaluate-dir", "analyze-dir", "epoch-checkpoint-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--kernel-sizes", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--channels-per-branch", type=int, default=32)
    p.add_argument("--cnn-activation", choices=("gelu", "none"), default="gelu")
    p.add_argument("--visual-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=234)
    p.add_argument("--batch-size", type=int, default=64, help="Per label group: 64+64 by default")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lambda-fc-l1", type=float, default=0.0,
                   help="L1 coefficient for the sum of absolute FC weights; bias excluded")
    p.add_argument("--fc-l1-target", choices=("weight", "delta"), default="weight",
                   help="Regularize FC weights separately, or their Abnormal-Normal difference")
    p.add_argument("--max-epoch", type=int, default=10)
    p.add_argument("--scheduler-milestones", type=int, nargs="+", default=[4, 8])
    p.add_argument("--scheduler-rate", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.)
    p.add_argument("--evaluate-epochs", type=int, nargs="*", default=[5, 7, 10])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--expected-train-views", type=int, default=16100)
    p.add_argument("--expected-test-videos", type=int, default=290)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU for smoke tests only")
    return p


def fc_l1_penalty(fc_weight, target):
    if target == "weight":
        return fc_weight.abs().sum()
    if target == "delta":
        return (fc_weight[1] - fc_weight[0]).abs().sum()
    raise ValueError(f"Unknown FC L1 target: {target}")


def fc_gradient_diagnostics(ce, fc_weight, coefficient, target="weight"):
    """Inspect the current batch without changing .grad or consuming the graph."""
    ce_grad = torch.autograd.grad(ce, fc_weight, retain_graph=True)[0].detach()
    if target == "weight":
        l1_grad = coefficient * fc_weight.detach().sign()
    elif target == "delta":
        sign = (fc_weight[1] - fc_weight[0]).detach().sign()
        l1_grad = coefficient * torch.stack((-sign, sign))
    else:
        raise ValueError(f"Unknown FC L1 target: {target}")
    ce_norm = float(torch.linalg.vector_norm(ce_grad))
    l1_norm = float(torch.linalg.vector_norm(l1_grad))
    dot = float((ce_grad * l1_grad).sum())
    return {
        "fc_grad_ce_norm_first_batch": ce_norm,
        "fc_grad_weighted_l1_norm_first_batch": l1_norm,
        "fc_grad_l1_over_ce_first_batch": l1_norm / ce_norm if ce_norm > 0 else None,
        "fc_grad_cosine_first_batch": dot / (ce_norm * l1_norm) if ce_norm > 0 and l1_norm > 0 else None,
    }


def train(args):
    if (args.max_epoch < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0
            or not math.isfinite(args.lambda_fc_l1) or args.lambda_fc_l1 < 0):
        raise ValueError("Invalid training hyperparameters")
    if args.grad_clip_norm <= 0 or any(e < 1 or e > args.max_epoch for e in args.evaluate_epochs):
        raise ValueError("Invalid gradient clip or evaluation epochs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime with CUDA PyTorch")
    for path in (args.train_list, args.test_list, args.gt_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    result_dir, checkpoints = Path(args.results_dir), Path(args.epoch_checkpoint_dir)
    if (result_dir / "config.json").exists() or (checkpoints.exists() and any(checkpoints.glob("*.pth"))):
        raise FileExistsError("Run already exists. Use a new run tag; no automatic resume or overwrite.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    setup_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    datasets = [UCFCNNFeatureDataset(args.feature_root, args.train_list, args.visual_length,
                                    train=True, normal=value) for value in (True, False)]
    test_dataset = UCFCNNFeatureDataset(args.feature_root, args.test_list, args.visual_length)
    if args.expected_test_videos and len(test_dataset) != args.expected_test_videos:
        raise ValueError(f"Expected {args.expected_test_videos} test videos, found {len(test_dataset)}")
    views = sum(map(len, datasets))
    if args.expected_train_views and views != args.expected_train_views:
        raise ValueError(f"Expected {args.expected_train_views} train views, found {views}")
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          generator=torch.Generator().manual_seed(args.seed + i))
               for i, dataset in enumerate(datasets)]
    steps = min(map(len, loaders))
    if not steps:
        raise ValueError("Each label group needs at least one full batch")
    model = MultiscaleCNN(args.kernel_sizes, args.channels_per_branch, args.visual_length,
                          activation=args.cnn_activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    (result_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "python": sys.version, "deterministic": True,
                   "lgt_padding": "original VadCLIP; invalid positions excluded before CNN and max pooling"}
    (result_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    l1_description = "sum(abs(FC.weight))" if args.fc_l1_target == "weight" else "sum(abs(FC.weight[1] - FC.weight[0]))"
    print(f"Scratch training; loss=video CE + {args.lambda_fc_l1:g} * {l1_description}, bias excluded; "
          f"{len(datasets[0])} normal + {len(datasets[1])} abnormal views; "
          f"{steps} steps/epoch. No CLIP/text/classifier checkpoint loaded.", flush=True)
    history, evaluation_history = [], []
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        ce_sum, l1_sum, weighted_l1_sum, total_sum, correct, count = 0., 0., 0., 0., 0, 0
        grad_diagnostics = None
        for normal, abnormal in zip(*loaders):
            features = torch.cat((normal[0], abnormal[0])).to(device)
            lengths = torch.cat((normal[2], abnormal[2])).to(device)
            labels = torch.cat((torch.zeros(len(normal[0])), torch.ones(len(abnormal[0])))).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, lengths)
            ce = F.cross_entropy(output["video_logits"], labels)
            l1 = fc_l1_penalty(model.head.fc.weight, args.fc_l1_target)
            weighted_l1 = args.lambda_fc_l1 * l1
            loss = ce + weighted_l1 if args.lambda_fc_l1 else ce
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if grad_diagnostics is None:
                grad_diagnostics = fc_gradient_diagnostics(ce, model.head.fc.weight, args.lambda_fc_l1, args.fc_l1_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            ce_sum += float(ce.detach()) * len(labels)
            l1_sum += float(l1.detach()) * len(labels)
            weighted_l1_sum += float(weighted_l1.detach()) * len(labels)
            total_sum += float(loss.detach()) * len(labels)
            correct += int((output["video_logits"].argmax(-1) == labels).sum())
            count += len(labels)
        delta = (model.head.fc.weight[1] - model.head.fc.weight[0]).detach().abs().float().cpu()
        quantiles = torch.quantile(delta, torch.tensor([0.1, 0.5, 0.9]))
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "video_ce": ce_sum / count, "fc_l1_target": args.fc_l1_target,
               "fc_l1": l1_sum / count,
               "fc_weight_l1_end": float(model.head.fc.weight.detach().abs().sum()),
               "delta_l1_end": float(delta.sum()),
               "weighted_fc_l1": weighted_l1_sum / count, "total_loss": total_sum / count,
               "weighted_l1_over_ce": weighted_l1_sum / ce_sum if ce_sum > 0 else None,
               "abs_delta_w_mean": float(delta.mean()),
               "abs_delta_w_p10": float(quantiles[0]), "abs_delta_w_p50": float(quantiles[1]),
               "abs_delta_w_p90": float(quantiles[2]),
               "delta_w_abs_below_0_001_count": int((delta < 0.001).sum()),
               "delta_w_abs_below_0_001_fraction": float((delta < 0.001).float().mean()),
               "video_accuracy": 100 * correct / count, "views_seen": count}
        row.update(grad_diagnostics)
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        checkpoint = checkpoints / f"epoch_{epoch:03d}.pth"
        torch.save({"model_type": "multiscale_video_cnn", "epoch": epoch, "config": config,
                    "state_dict": model.state_dict()}, checkpoint)
        write_csv(result_dir / "training_history.csv", history)
        if epoch in args.evaluate_epochs:
            # Separate process keeps evaluation from consuming training RNG state.
            ev = Path(args.evaluate_dir) / f"epoch_{epoch:03d}"
            command = [sys.executable, "-u", str(Path(__file__).with_name("ucf_evaluate_multiscale_cnn.py")),
                       "--model-path", str(checkpoint), "--feature-root", args.feature_root,
                       "--test-list", args.test_list, "--gt-path", args.gt_path,
                       "--evaluate-dir", str(ev), "--analyze-dir", str(Path(args.analyze_dir) / ev.name),
                       "--expected-videos", str(args.expected_test_videos), "--micro-batch", str(args.micro_batch)]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            subprocess.run(command, check=True)
            metrics = json.loads((ev / "metrics.json").read_text(encoding="utf-8"))
            evaluation_history.append({key: metrics[key] for key in (
                "epoch", "classifier_auc", "classifier_ap", "ano_auc", "video_auc", "video_ap", "video_accuracy")})
            write_csv(result_dir / "evaluation_history.csv", evaluation_history)
    (result_dir / "completed.json").write_text(json.dumps({"epoch": args.max_epoch,
        "final_checkpoint": str(checkpoint), "selection": "fixed final epoch, not test AUC"}, indent=2), encoding="utf-8")
    print("Final checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    train(build_parser().parse_args())
