import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ucf_train_cnn_vadclip import build_parser, evaluate_epoch


class EpochEvaluationTests(unittest.TestCase):
    def args(self, extra=()):
        return build_parser().parse_args([
            "--feature-root", "features", "--train-list", "train.csv",
            "--epoch-checkpoint-dir", "epochs", "--output-model-path", "final.pth",
            "--results-dir", "results", *extra,
        ])

    def test_legacy_defaults_remain_opt_in(self):
        args = self.args()
        self.assertFalse(args.deterministic)
        self.assertFalse(args.evaluate_every_epoch)
        self.assertEqual(args.grad_clip_norm, 0)
        self.assertEqual(args.lr, 1e-5)
        self.assertEqual(args.max_epoch, 10)

    def test_stable_options_and_schedule(self):
        args = self.args([
            "--deterministic", "--data-seed", "234", "--grad-clip-norm", "1",
            "--evaluate-every-epoch", "--lr", "5e-6", "--max-epoch", "20",
            "--scheduler-milestones", "8", "16",
        ])
        self.assertTrue(args.deterministic)
        self.assertTrue(args.evaluate_every_epoch)
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.scheduler_milestones, gamma=args.scheduler_rate)
        rates = []
        for _ in range(20):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        for start, end, expected in ((0, 8, 5e-6), (8, 16, 5e-7), (16, 20, 5e-8)):
            for rate in rates[start:end]:
                self.assertAlmostEqual(rate, expected)

    def test_epoch_eval_passes_matching_model_and_isolated_outputs(self):
        args = self.args([
            "--cnn-architecture", "single_conv", "--score-mode", "competitive_response",
            "--kernel-size", "5", "--cnn-channels", "128",
            "--evaluate-dir", "eval", "--analyze-dir", "analyze",
            "--test-list", "test.csv", "--gt-path", "gt.npy",
            "--gt-segment-path", "segments.npy", "--gt-label-path", "labels.npy",
        ])
        rng = torch.get_rng_state().clone()
        with patch("ucf_train_cnn_vadclip.subprocess.run") as run, \
                patch("torch.cuda.empty_cache"), \
                patch.object(Path, "read_text", return_value=json.dumps({"classifier_auc": 87.0})):
            metrics = evaluate_epoch(args, 3, Path("epochs"), Path("results"))
        command = run.call_args.args[0]
        options = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(options["--model-path"], str(Path("epochs/epoch_003.pth")))
        self.assertEqual(options["--evaluate-dir"], str(Path("eval/epoch_003")))
        self.assertEqual(options["--analyze-dir"], str(Path("analyze/epoch_003")))
        self.assertEqual(options["--score-mode"], "competitive_response")
        self.assertEqual(options["--kernel-size"], "5")
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(metrics["classifier_auc"], 87.0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_gradient_clip_bounds_norm(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([30., 40.])
        norm = torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
        self.assertAlmostEqual(float(norm), 50.)
        self.assertLessEqual(float(parameter.grad.norm()), 1.000001)


if __name__ == "__main__":
    unittest.main()
