import argparse
import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from model_cnn_vadclip import CLIPVADCNN, SingleConvReadout, validate_cnn_kernel
from ucf_train_cnn_vadclip import build_parser


class KernelSweepTests(unittest.TestCase):
    def test_all_sizes_keep_length_mask_padding_and_backward(self):
        for k in range(1, 10):
            with self.subTest(k=k):
                model = SingleConvReadout(input_width=4, channels=8, kernel_size=k,
                                         score_mode="competitive_response_gelu")
                x = torch.randn(2, 17, 4)
                lengths = torch.tensor([3, 17])
                changed = x.clone()
                changed[0, 3:] = 1e5
                y = model(x, lengths)
                self.assertEqual(tuple(y.shape), (2, 17, 1))
                torch.testing.assert_close(y, model(changed, lengths))
                self.assertTrue(torch.equal(y[0, 3:], torch.zeros_like(y[0, 3:])))
                self.assertTrue(torch.isfinite(y).all())
                y.sum().backward()
                self.assertTrue(torch.isfinite(model.conv.weight.grad).all())
                self.assertGreater(float(model.conv.weight.grad[:4].abs().sum()), 0.)
                self.assertGreater(float(model.conv.weight.grad[4:].abs().sum()), 0.)

    def test_same_padding_alignment_matches_explicit_convolution(self):
        for k in range(1, 10):
            with self.subTest(k=k):
                model = SingleConvReadout(input_width=2, channels=4, kernel_size=k)
                x = torch.randn(1, 2, 13)
                expected = F.conv1d(F.pad(x, ((k-1)//2, k//2)), model.conv.weight, model.conv.bias)
                torch.testing.assert_close(model.conv(x), expected)
                if k % 2:
                    self.assertEqual(model.conv.padding, (k//2,))
                self.assertEqual(model.conv(x[:, :, :1]).shape[-1], 1)

    def test_kernel_dimensions_parameter_count_and_checkpoint(self):
        for k in (1, 2, 5, 8, 9, 11):
            model = SingleConvReadout(kernel_size=k, score_mode="competitive_response_gelu")
            self.assertEqual(tuple(model.conv.weight.shape), (128, 512, k))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 128*512*k + 129)
            clone = SingleConvReadout(kernel_size=k, score_mode="competitive_response_gelu")
            clone.load_state_dict(model.state_dict(), strict=True)

    def test_invalid_kernel_rejected_before_clip_load(self):
        for k in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                SingleConvReadout(kernel_size=k)
        with self.assertRaises(ValueError):
            CLIPVADCNN(kernel_size=2, cnn_architecture="two_block")
        validate_cnn_kernel(12, "single_conv")

    def test_evaluation_parser_accepts_even_kernel(self):
        # Exercise the actual parser without importing metric dependencies.
        path = Path(__file__).with_name("ucf_evaluate_cnn_vadclip.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        args = namespace["build_parser"]().parse_args([
            "--model-path", "m", "--feature-root", "f", "--test-list", "t", "--gt-path", "g",
            "--gt-segment-path", "s", "--gt-label-path", "l", "--evaluate-dir", "e", "--analyze-dir", "a",
            "--cnn-architecture", "single_conv", "--score-mode", "competitive_response_gelu", "--kernel-size", "4",
        ])
        self.assertEqual(args.kernel_size, 4)

    def test_notebook_options_paths_and_completion_guards(self):
        notebook = Path(__file__).resolve().parents[2] / "code/train_cnn_kernel_sweep_colab.ipynb"
        if not notebook.is_file():
            self.skipTest("Notebook not uploaded; model/parser tests still run")
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        definitions = []
        for cell in cells:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                definitions.extend(n for n in tree.body if isinstance(n, ast.FunctionDef))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ns = dict(Path=Path, json=json, csv=csv, RESULT_ROOT=root / "results", SRC_DIR=root / "src",
                      LIST_DIR=root / "list", FEATURE_ROOT=root / "features", SWEEP_TAG="test_sweep",
                      SCORE_MODE="competitive_response_gelu", SEED=234, MAX_EPOCH=10,
                      EVALUATE_EPOCHS=[5, 7, 10])
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook_helpers", "exec"), ns)
            outputs = set()
            for k in range(1, 10):
                paths = ns["variant_paths"](k)
                options = ns["train_options"](k, paths)
                args = build_parser().parse_args(ns["command_args"](options))
                self.assertEqual((args.kernel_size, args.seed, args.data_seed), (k, 234, 234))
                self.assertFalse(args.evaluate_every_epoch)
                self.assertTrue(args.deterministic)
                self.assertEqual(args.evaluate_epochs, [5, 7, 10])
                self.assertEqual(args.max_epoch, 10)
                self.assertEqual(args.lr, 5e-6)
                self.assertEqual(args.scheduler_milestones, [8, 16])
                outputs.add(str(paths["epochs"]))
                ns["check_config"](options, paths)
            self.assertEqual(len(outputs), 9)
            paths["train"].mkdir(parents=True)
            (paths["train"] / "config.json").write_text(json.dumps(options), encoding="utf-8")
            ns["check_config"](options, paths)
            with self.assertRaises(RuntimeError):
                ns["check_config"]({**options, "lr": 1e-5}, paths)
            self.assertFalse(ns["completed"](paths))
            paths["epochs"].mkdir(parents=True)
            (paths["epochs"] / "epoch_010.pth").touch()
            paths["final"].parent.mkdir(parents=True)
            paths["final"].touch()
            (paths["train"] / "early_stopping.json").write_text(
                json.dumps(dict(epochs_ran=10, stopped_early=False)), encoding="utf-8")
            self.assertTrue(ns["completed"](paths))

    @unittest.skipUnless(torch.cuda.is_available(), "Full model test requires CUDA")
    def test_full_model_even_kernel_cuda(self):
        from ucf_cnn_common import TRAIN_LABEL_MAP, aggregate_video_logits
        from utils.tools import get_prompt_text
        device = torch.device("cuda")
        model = CLIPVADCNN(14, 512, 256, 512, 1, 2, 8, 10, 10, device,
                          kernel_size=2, hidden_width=128, cnn_architecture="single_conv",
                          score_mode="competitive_response_gelu").to(device)
        lengths = torch.tensor([170, 256], device=device)
        _, logits1, logits2 = model(torch.randn(2, 256, 512, device=device), None,
                                   get_prompt_text(TRAIN_LABEL_MAP), lengths)
        self.assertEqual(tuple(logits1.shape), (2, 256, 1))
        self.assertEqual(tuple(logits2.shape), (2, 256, 14))
        loss = F.binary_cross_entropy_with_logits(aggregate_video_logits(logits1, lengths, "topk"),
                                                 torch.tensor([0., 1.], device=device))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.cnn_classifier.conv.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.clipmodel.parameters()))


if __name__ == "__main__":
    unittest.main()
