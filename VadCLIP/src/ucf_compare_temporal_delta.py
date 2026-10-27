import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ucf_evaluate import UCFRelativeTestDataset, evaluate_model, print_metrics
from ucf_evaluate_temporal_score import LABEL_MAP, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Compare one temporal checkpoint with its score correction enabled and disabled."
    )
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--temporal-hidden-width", default=128, type=int)
    parser.add_argument("--max-temporal-delta", default=0.5, type=float)
    parser.add_argument("--temporal-mode", required=True,
                        choices=["feature_sequence", "feature_difference"])
    parser.add_argument("--temporal-model-path", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UCFRelativeTestDataset(
        args.visual_length, args.test_list, args.feature_root
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = load_model(args, device, True, args.temporal_model_path)

    model.temporal_correction_enabled = True
    metrics_on = evaluate_model(model, loader, args, LABEL_MAP, device)
    model.temporal_correction_enabled = False
    metrics_off = evaluate_model(model, loader, args, LABEL_MAP, device)

    print_metrics("Temporal checkpoint: delta ON", metrics_on)
    print_metrics("Same checkpoint: delta OFF", metrics_off)
    comparison = {
        "temporal_checkpoint": args.temporal_model_path,
        "temporal_mode": args.temporal_mode,
        "test_videos": len(dataset),
        "delta_on": metrics_on,
        "delta_off": metrics_off,
        "delta_on_minus_off": {
            key: metrics_on[key] - metrics_off[key]
            for key in metrics_on
            if isinstance(metrics_on[key], (int, float))
        },
        "note": (
            "Both evaluations use the same learned VadCLIP weights. Only the direct "
            "temporal correction added to logits1 is toggled. Alignment logits2 are "
            "computed from base_logits1 and should therefore remain unchanged."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics_delta_on_off.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved delta on/off comparison: {output_path}")


if __name__ == "__main__":
    main()
