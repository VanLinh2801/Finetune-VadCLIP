import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


parser = argparse.ArgumentParser(description="Weakly supervised temporal proposal VadCLIP")
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

parser.add_argument("--num-proposals", default=8, type=int)
parser.add_argument("--proposal-heads", default=4, type=int)
parser.add_argument("--proposal-min-width", default=0.02, type=float)
parser.add_argument("--proposal-max-width", default=0.50, type=float)
parser.add_argument("--proposal-initial-width", default=0.15, type=float)
parser.add_argument("--mask-temperature", default=0.02, type=float)
parser.add_argument("--context-scale", default=0.25, type=float)
parser.add_argument("--bag-temperature", default=1.0, type=float)
parser.add_argument("--class-selection-temperature", default=0.25, type=float)
parser.add_argument("--boundary-margin", default=0.10, type=float)

parser.add_argument("--lambda-normal", default=0.50, type=float)
parser.add_argument("--lambda-class", default=1.00, type=float)
parser.add_argument("--lambda-width", default=0.0, type=float)
parser.add_argument("--lambda-coverage", default=0.05, type=float)
parser.add_argument("--proposal-min-coverage", default=0.05, type=float)
parser.add_argument("--proposal-max-coverage", default=0.50, type=float)
parser.add_argument("--lambda-overlap", default=0.05, type=float)
parser.add_argument("--lambda-boundary", default=0.01, type=float)
parser.add_argument("--boundary-warmup-epochs", default=1, type=int)

parser.add_argument("--max-epoch", default=10, type=int)
parser.add_argument("--early-stopping-patience", default=3, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", default=[4, 8], nargs="+", type=int)
parser.add_argument("--num-workers", default=2, type=int)
parser.add_argument("--pin-memory", default=True, type=str2bool)

parser.add_argument("--feature-root", default="../../UCFClipFeatures")
parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_description.csv")
parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
parser.add_argument("--output-model-path", default="model/model_ucf_proposal_scratch.pth")
parser.add_argument("--checkpoint-path", default="model/checkpoint_proposal_scratch.pth")
parser.add_argument("--save-cur-path", default="model/model_cur_proposal_scratch.pth")
parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints_proposal_scratch")
parser.add_argument("--results-dir", default="../../code/results/evaluate/temporal_proposal_scratch")
