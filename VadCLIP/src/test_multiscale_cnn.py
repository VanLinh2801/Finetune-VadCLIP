import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import CLIPVAD
from model_multiscale_cnn import MultiscaleCNN, MultiscaleVideoHead, VadCLIPVisualLGT, DeviceDistanceAdj


class MultiscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_head_shapes_padding_and_readout(self):
        torch.manual_seed(234)
        head = MultiscaleVideoHead((1, 2, 3, 7), 4, input_width=8)
        features = torch.randn(2, 11, 8)
        lengths = torch.tensor([3, 11])
        output = head(features, lengths)
        self.assertEqual(output["responses"].shape, (2, 16, 11))
        self.assertEqual(output["video_logits"].shape, (2, 2))
        self.assertEqual(output["frame_logits"].shape, (2, 11, 2))
        self.assertTrue((output["argmax"][0] < 3).all())
        polluted = features.clone()
        polluted[0, 3:] = 1e6
        torch.testing.assert_close(head(polluted, lengths)["video_logits"], output["video_logits"])
        short = head(features[:1, :3], [3])
        torch.testing.assert_close(short["video_logits"], output["video_logits"][:1])
        torch.testing.assert_close(short["frame_logits"], output["frame_logits"][:1, :3])
        contrast = head.fc.weight[1] - head.fc.weight[0]
        frame_difference = torch.einsum("c,bct->bt", contrast, output["responses"]) + head.fc.bias[1] - head.fc.bias[0]
        torch.testing.assert_close(frame_difference[1].sigmoid(), output["frame_logits"][1].softmax(-1)[:, 1])
        difference = output["contributions"].sum(-1) + head.fc.bias[1] - head.fc.bias[0]
        torch.testing.assert_close(difference, output["video_logits"][:, 1] - output["video_logits"][:, 0])
        F.cross_entropy(output["video_logits"], torch.tensor([0, 1])).backward()
        for conv in head.branches:
            self.assertGreater(float(conv.weight.grad.abs().sum()), 0)

    def test_eight_scales_256_channels(self):
        head = MultiscaleVideoHead(tuple(range(2, 10)), 32)
        self.assertEqual(head.fc.in_features, 256)
        output = head(torch.randn(2, 13, 512), [5, 13])
        self.assertEqual(output["responses"].shape, (2, 256, 13))
        self.assertEqual(output["pooled"].shape, (2, 256))
        self.assertEqual(output["video_logits"].shape, (2, 2))
        self.assertTrue((output["argmax"][0] < 5).all())
        F.cross_entropy(output["video_logits"], torch.tensor([0, 1])).backward()
        for conv in head.branches:
            self.assertTrue(torch.isfinite(conv.weight.grad).all())

    def test_raw_response_without_gelu(self):
        head = MultiscaleVideoHead((1, 2), 2, input_width=1, activation="none")
        with torch.no_grad():
            for conv in head.branches:
                conv.weight.fill_(1)
                conv.bias.fill_(-2)
        x = torch.ones(1, 3, 1)
        output = head(x, [2])
        torch.testing.assert_close(output["responses"][0, :2, :2], -torch.ones(2, 2))
        self.assertTrue((output["argmax"] < 2).all())
        gelu_head = MultiscaleVideoHead((1, 2), 2, input_width=1)
        gelu_head.load_state_dict(head.state_dict(), strict=True)
        torch.testing.assert_close(gelu_head(x, [2])["responses"], F.gelu(output["responses"]))
        model = MultiscaleCNN((1, 2), 2, 8, activation="none")
        self.assertIsInstance(model.head.activation, nn.Identity)
        self.assertEqual(type(model.lgt.gelu).__name__, "QuickGELU")

    def test_negative_gelu_not_replaced_by_padding_max(self):
        head = MultiscaleVideoHead((1,), 1, input_width=1)
        with torch.no_grad():
            head.branches[0].weight.fill_(1)
            head.branches[0].bias.zero_()
        output = head(-torch.ones(1, 5, 1), [1])
        self.assertLess(float(output["pooled"][0, 0].detach()), 0)
        self.assertEqual(int(output["argmax"][0, 0]), 0)

    def test_fc_l1_penalty_excludes_bias_and_other_modules(self):
        model = MultiscaleCNN((1, 3), 2, 8, activation="none")
        with torch.no_grad():
            model.head.fc.weight.fill_(0.25)
        penalty = model.head.fc.weight.abs().sum()
        self.assertAlmostEqual(float(penalty.detach()), 2 * 4 * 0.25)
        (1e-5 * penalty).backward()
        torch.testing.assert_close(model.head.fc.weight.grad, torch.full_like(model.head.fc.weight, 1e-5))
        self.assertIsNone(model.head.fc.bias.grad)
        self.assertIsNone(model.head.branches[0].weight.grad)
        self.assertIsNone(model.lgt.linear.weight.grad)

    def test_fc_gradient_diagnostics_preserves_backward(self):
        from ucf_train_multiscale_cnn import fc_gradient_diagnostics
        fc = nn.Linear(3, 2)
        inputs = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])
        targets = torch.tensor([0, 1])
        ce = F.cross_entropy(fc(inputs), targets)
        grad = torch.autograd.grad(ce, fc.weight, retain_graph=True)[0]
        measured = fc_gradient_diagnostics(ce, fc.weight, 3e-5)
        self.assertIsNone(fc.weight.grad)
        self.assertAlmostEqual(measured["fc_grad_ce_norm_first_batch"], float(grad.norm()), places=7)
        expected_l1 = 3e-5 * fc.weight.detach().sign()
        self.assertAlmostEqual(measured["fc_grad_weighted_l1_norm_first_batch"], float(expected_l1.norm()), places=10)
        (ce + 3e-5 * fc.weight.abs().sum()).backward()
        torch.testing.assert_close(fc.weight.grad, grad + expected_l1)

    def test_delta_l1_and_gradient(self):
        from ucf_train_multiscale_cnn import fc_l1_penalty, fc_gradient_diagnostics
        fc = nn.Linear(3, 2)
        with torch.no_grad():
            fc.weight.copy_(torch.tensor([[1., -1., 0.5], [3., -2., 0.5]]))
        penalty = fc_l1_penalty(fc.weight, "delta")
        self.assertEqual(float(penalty.detach()), 3.)
        self.assertEqual(float(fc_l1_penalty(fc.weight, "weight").detach()), 8.)
        ce = F.cross_entropy(fc(torch.ones(2, 3)), torch.tensor([0, 1]))
        ce_grad = torch.autograd.grad(ce, fc.weight, retain_graph=True)[0]
        measured = fc_gradient_diagnostics(ce, fc.weight, 1e-4, "delta")
        expected = 1e-4 * torch.tensor([[-1., 1., 0.], [1., -1., 0.]])
        self.assertAlmostEqual(measured["fc_grad_weighted_l1_norm_first_batch"], float(expected.norm()), places=9)
        self.assertIsNone(fc.weight.grad)
        (ce + 1e-4 * penalty).backward()
        torch.testing.assert_close(fc.weight.grad, ce_grad + expected)

        shifted = fc.weight.detach() + torch.tensor([4., -2., 1.])
        torch.testing.assert_close(fc_l1_penalty(shifted, "delta"), penalty.detach())

    def test_lgt_matches_original_with_identical_weights(self):
        # Avoid a CLIP download: original encode_video does not call CLIP.
        with patch("model.clip.load", return_value=(nn.Identity(), None)):
            original = CLIPVAD(14, 512, 16, 512, 1, 2, 8, 10, 10, torch.device("cpu"))
        original.disAdj = DeviceDistanceAdj()
        lgt = VadCLIPVisualLGT(16)
        original.load_state_dict(lgt.state_dict(), strict=False)
        original.eval()
        lgt.eval()
        x = torch.randn(2, 16, 512)
        x[0, 9:] = 0
        lengths = torch.tensor([9, 16])
        with torch.no_grad():
            torch.testing.assert_close(lgt(x, lengths), original.encode_video(x, None, lengths), rtol=0, atol=0)
        from scipy.spatial.distance import pdist, squareform
        distance = torch.from_numpy(squareform(pdist(np.arange(16).reshape(-1, 1), metric="cityblock").astype(np.float32)))
        expected = torch.exp(-distance / torch.exp(torch.tensor(1.)))
        torch.testing.assert_close(lgt.disAdj(2, 16), expected[None].repeat(2, 1, 1))

    def test_full_model_backward_and_whole_video_pooling(self):
        model = MultiscaleCNN((1, 3), 2, visual_length=8)
        x = torch.randn(2, 8, 512)
        lengths = torch.tensor([8, 3])
        output = model(x, lengths)
        loss = F.cross_entropy(output["video_logits"], torch.tensor([0, 1]))
        loss.backward()
        for parameter in (model.head.fc.weight, model.lgt.temporal.resblocks[0].attn.in_proj_weight,
                          model.lgt.linear.weight, model.lgt.gc1.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertFalse(any("clipmodel" in name or "text" in name for name, _ in model.named_parameters()))
        model.eval()
        with torch.no_grad():
            full = model.forward_video(x, lengths, micro_batch=1)
            h = model.lgt(x, lengths)
            expected = model.head(torch.cat((h[0, :8], h[1, :3]))[None], [11])
            torch.testing.assert_close(full["video_logits"], expected["video_logits"])
            torch.testing.assert_close(full["frame_logits"], expected["frame_logits"])
            self.assertEqual(full["frame_logits"].shape, (1, 11, 2))
            short = model.forward_video(x[1:], lengths[1:])
            self.assertEqual(short["frame_logits"].shape, (1, 3, 2))

    def test_tiny_epoch_checkpoint_and_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i, label in enumerate(("Normal", "Normal", "Arson", "Arson")):
                path = root / f"v{i}.npy"
                np.save(path, np.random.default_rng(i).normal(size=(19 if i % 2 else 5, 512)).astype("float32"))
                rows.append({"path": str(path), "label": label, "video_id": f"v{i}"})
            for name, items in (("train.csv", rows), ("test.csv", [rows[0], rows[3]])):
                with (root / name).open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(items)
            np.save(root / "gt.npy", np.r_[np.zeros(5*16), np.zeros(8*16), np.ones(11*16)])
            command = [sys.executable, str(Path(__file__).with_name("ucf_train_multiscale_cnn.py")),
                       "--feature-root", str(root), "--train-list", str(root / "train.csv"),
                       "--test-list", str(root / "test.csv"), "--gt-path", str(root / "gt.npy"),
                       "--results-dir", str(root / "train"), "--evaluate-dir", str(root / "eval"),
                       "--analyze-dir", str(root / "analyze"), "--epoch-checkpoint-dir", str(root / "model"),
                       "--device", "cpu", "--visual-length", "8", "--kernel-sizes", "1", "3",
                       "--channels-per-branch", "2", "--cnn-activation", "none", "--lambda-fc-l1", "1e-5", "--fc-l1-target", "delta", "--batch-size", "1", "--max-epoch", "1",
                       "--evaluate-epochs", "1", "--num-workers", "0", "--expected-train-views", "4",
                       "--expected-test-videos", "2"]
            import os
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads((root / "eval/epoch_001/metrics.json").read_text())
            self.assertEqual(metrics["frames"], 24*16)
            self.assertEqual(metrics["videos"], 2)
            self.assertEqual(metrics["cnn_activation"], "none")
            self.assertEqual(metrics["fc_l1_target"], "delta")
            self.assertIsNotNone(metrics["classifier_auc"])
            self.assertTrue((root / "analyze/epoch_001/kernel_evidence.csv").exists())
            saved = torch.load(root / "model/epoch_001.pth", weights_only=True)
            self.assertEqual(saved["config"]["cnn_activation"], "none")
            self.assertEqual(saved["config"]["lambda_fc_l1"], 1e-5)
            self.assertEqual(saved["config"]["fc_l1_target"], "delta")
            with (root / "train/training_history.csv").open(newline="") as f:
                row = next(csv.DictReader(f))
            ce, l1, weighted, total = (float(row[key]) for key in ("video_ce", "fc_l1", "weighted_fc_l1", "total_loss"))
            self.assertAlmostEqual(weighted, 1e-5 * l1, places=9)
            self.assertAlmostEqual(total, ce + weighted, places=6)
            self.assertAlmostEqual(float(row["weighted_l1_over_ce"]), weighted / ce, places=7)
            self.assertIn("abs_delta_w_p50", row)
            self.assertEqual(row["fc_l1_target"], "delta")
            self.assertGreater(float(row["fc_weight_l1_end"]), 0)
            self.assertGreater(float(row["delta_l1_end"]), 0)
            self.assertGreater(float(row["fc_grad_ce_norm_first_batch"]), 0)
            self.assertGreater(float(row["fc_grad_weighted_l1_norm_first_batch"]), 0)
            self.assertTrue(0 <= float(row["delta_w_abs_below_0_001_fraction"]) <= 1)
            MultiscaleCNN((1, 3), 2, 8, activation="none").load_state_dict(saved["state_dict"], strict=True)
            self.assertEqual(saved["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
