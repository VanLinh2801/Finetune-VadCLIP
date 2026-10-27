import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[4, 8])
parser.add_argument(
    '--training-mode',
    default='original',
    choices=[
        'original',
        'abnormal_topk_epoch_eval',
        'ranking_margin_epoch_eval',
        'ranking_hard_pair_epoch_eval',
        'transplant_erasing_epoch_eval',
        'multi_classifier_epoch_eval',
        'adapter_multi_classifier_epoch_eval',
    ],
)
parser.add_argument('--eval-steps', default=1280, type=int)
parser.add_argument('--eval-each-epoch', default=False, type=bool)
parser.add_argument('--lambda-normal-topk', default=0.0, type=float)
parser.add_argument('--normal-topk-warmup-epochs', default=0, type=int)
parser.add_argument('--lambda-abnormal-bg', default=0.0, type=float)
parser.add_argument('--abnormal-bg-margin', default=0.20, type=float)
parser.add_argument('--abnormal-bg-warmup-epochs', default=0, type=int)
parser.add_argument('--lambda-ranking-margin', default=0.05, type=float)
parser.add_argument('--ranking-margin', default=0.20, type=float)
parser.add_argument('--ranking-warmup-epochs', default=1, type=int)
parser.add_argument('--ranking-soft-temperature', default=0.10, type=float)
parser.add_argument('--ranking-hard-normal-fraction', default=0.25, type=float)
parser.add_argument('--regularize-score-branch', default='both', choices=['classifier', 'alignment', 'both'])
parser.add_argument('--regularize-top-k-ratio', default=0.15, type=float)
parser.add_argument('--lambda-transplant', default=0.05, type=float)
parser.add_argument('--lambda-evidence-erasing', default=0.10, type=float)
parser.add_argument('--augmentation-warmup-epochs', default=2, type=int)
parser.add_argument('--augmentation-min-segment-length', default=4, type=int)
parser.add_argument('--augmentation-max-segment-length', default=16, type=int)
parser.add_argument(
    '--transplant-score-branch',
    default='classifier',
    choices=['classifier', 'alignment', 'both'],
)
parser.add_argument('--transplant-context-width', default=8, type=int)
parser.add_argument('--transplant-high-exclusion-ratio', default=0.15, type=float)
parser.add_argument('--transplant-boundary-blend-width', default=1, type=int)
parser.add_argument('--track-topk', action='store_true')
parser.add_argument('--topk-tracking-output-dir', default='model/topk_tracking')
parser.add_argument(
    '--topk-tracking-branches',
    default='both',
    choices=['classifier', 'alignment', 'both'],
)
parser.add_argument(
    '--topk-tracking-scope',
    default='abnormal',
    choices=['abnormal', 'all'],
)
parser.add_argument('--early-stopping-patience', default=3, type=int)
parser.add_argument('--epoch-checkpoint-dir', default='model/epoch_checkpoints')
parser.add_argument('--save-cur-path', default='model/model_cur.pth')
parser.add_argument('--save-epoch-checkpoints', action='store_true')
parser.add_argument('--num-classifier-heads', default=2, type=int)
parser.add_argument('--head-adapter-bottleneck', default=128, type=int)
parser.add_argument('--head-adapter-dropout', default=0.1, type=float)
parser.add_argument('--head-adapter-residual-scale', default=0.1, type=float)
parser.add_argument('--lambda-head-diversity', default=0.01, type=float)
parser.add_argument('--lambda-head-balance', default=0.001, type=float)
parser.add_argument('--head-diversity-temperature', default=1.0, type=float)
parser.add_argument('--head-diversity-warmup-epochs', default=2, type=int)
parser.add_argument('--head-diversity-ramp-epochs', default=2, type=int)
parser.add_argument(
    '--multi-head-cls-aggregate-weight', default=0.5, type=float
)
parser.add_argument(
    '--model-type',
    default='baseline',
    choices=['baseline', 'multi_classifier', 'adapter_multi_classifier'],
)
