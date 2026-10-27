import argparse


parser = argparse.ArgumentParser(description="Train VadCLIP with a temporal score correction")
parser.add_argument("--seed", default=234, type=int)
parser.add_argument("--embed-dim", default=512, type=int)
parser.add_argument("--visual-length", default=256, type=int)
parser.add_argument("--visual-width", default=512, type=int)
parser.add_argument("--visual-head", default=1, type=int)
parser.add_argument("--visual-layers", default=2, type=int)
parser.add_argument("--attn-window", default=8, type=int)
parser.add_argument("--prompt-prefix", default=10, type=int)
parser.add_argument("--prompt-postfix", default=10, type=int)
parser.add_argument("--classes-num", default=14, type=int)
parser.add_argument(
    "--temporal-mode",
    required=True,
    choices=["feature_sequence", "feature_difference"],
)
parser.add_argument("--temporal-hidden-width", default=128, type=int)
parser.add_argument("--max-temporal-delta", default=0.5, type=float)
parser.add_argument("--max-epoch", default=10, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", default=[4, 8], nargs="+", type=int)
parser.add_argument("--num-workers", default=2, type=int)
parser.add_argument("--pin-memory", default=True, type=lambda x: x.lower() in {"1", "true", "yes"})
parser.add_argument("--train-list", required=True)
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
parser.add_argument("--output-model-path", required=True)
parser.add_argument("--epoch-checkpoint-dir", required=True)
parser.add_argument("--results-dir", required=True)
