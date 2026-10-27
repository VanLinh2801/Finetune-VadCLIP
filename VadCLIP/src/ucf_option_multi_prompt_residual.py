import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    description="Multi-prompt + per-prompt residual (Hướng 2a+3) fine-tuning on UCF-Crime"
)
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

parser.add_argument("--max-epoch", default=10, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", default=[4, 8], nargs="+", type=int)

parser.add_argument("--prototype-source", default="manual", choices=["manual", "compact", "long_event_only"])
parser.add_argument("--prototype-json", default="../../code/ucf_weighted_class_prompts.json")
parser.add_argument(
    "--prompt-weights-json",
    default=None,
    help="Optional JSON containing prompt_weights[class] aligned with --prototype-json prompts.",
)
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
parser.add_argument("--lambda-consistency", default=0.01, type=float)
parser.add_argument("--consistency-top-k-ratio", default=0.15, type=float)
parser.add_argument(
    "--regularize-score-branch",
    default="both",
    choices=["classifier", "alignment", "both"],
    help="Score branch used by normal/background/ranking regularizers.",
)
parser.add_argument("--regularize-top-k-ratio", default=0.15, type=float)
parser.add_argument(
    "--lambda-normal-topk",
    default=0.0,
    type=float,
    help="Suppress top-k anomaly scores in Normal videos. Disabled when 0.",
)
parser.add_argument(
    "--lambda-abnormal-bg",
    default=0.0,
    type=float,
    help=(
        "Intra-video high/low top-k separation margin on abnormal videos: "
        "enforce min(top-k highest scores) >= max(top-k lowest scores) + abnormal-bg-margin. "
        "Disabled when 0."
    ),
)
parser.add_argument(
    "--abnormal-bg-margin",
    default=0.20,
    type=float,
    help="Margin for intra-video high/low top-k separation on abnormal videos.",
)
parser.add_argument(
    "--abnormal-bg-warmup-epochs",
    default=0,
    type=int,
    help=(
        "Skip abnormal-bg margin loss for this many initial epochs "
        "(0-indexed epoch < warmup). Disabled when 0."
    ),
)
parser.add_argument(
    "--lambda-ranking-margin",
    default=0.0,
    type=float,
    help="Enforce abnormal top-k scores to be higher than normal top-k scores. Disabled when 0.",
)
parser.add_argument("--ranking-margin", default=0.20, type=float)
parser.add_argument(
    "--lambda-residual",
    default=1e-3,
    type=float,
    help="Weight for mean ||Delta_{c,k}||^2 regularizer on per-prompt residuals.",
)
parser.add_argument(
    "--train-mode",
    default="from_scratch",
    choices=["from_scratch", "warmstart_continue", "warmstart_delta_only"],
    help=(
        "from_scratch: random VAD + soft prompt + Delta. "
        "warmstart_continue: load Hướng 2a, keep training VAD + soft prompt + Delta. "
        "warmstart_delta_only: load Hướng 2a, freeze VAD; train soft prompt + Delta_{c,k}."
    ),
)
parser.add_argument(
    "--base-model-path",
    default=None,
    help="Required for warmstart_*: path to trained Hướng 2a (multi_prompt) checkpoint.",
)

parser.add_argument("--use-pretrained-model", default=False, type=str2bool)
parser.add_argument("--pretrained-model-path", default="../../model_ucf.pth")
parser.add_argument("--output-model-path", default="model/model_ucf_multi_prompt_residual_scratch.pth")
parser.add_argument("--checkpoint-path", default="model/checkpoint_multi_prompt_residual_scratch.pth")

parser.add_argument("--feature-root", default="../../UCFClipFeatures")
parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_description.csv")
parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")

parser.add_argument("--eval-steps", default=0, type=int)
parser.add_argument(
    "--early-stopping-patience",
    default=3,
    type=int,
    help="Evaluate after each epoch and stop after this many epochs without AUC improvement. Disabled when 0.",
)
parser.add_argument("--save-cur-path", default="model/model_cur_multi_prompt_residual_scratch.pth")
parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints_multi_prompt_residual_scratch")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--pin-memory", default=True, type=str2bool)
parser.add_argument("--use-amp", default=False, type=str2bool)
parser.add_argument("--track-topk", default=False, type=str2bool)
parser.add_argument("--topk-tracking-output-dir", default="model/topk_tracking_multi_prompt_residual")
parser.add_argument(
    "--topk-tracking-branches",
    default="both",
    choices=["classifier", "alignment", "both"],
)
parser.add_argument(
    "--topk-tracking-scope",
    default="abnormal",
    choices=["abnormal", "all"],
)
