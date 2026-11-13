import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model_multiscale_cnn import MultiscaleCNN
from ucf_analyze_multiscale_selection import build_selections, selection_logits, validate_pair


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fc_channel_selection_bias_and_sign(self):
        fc = nn.Linear(4, 2)
        with torch.no_grad():
            fc.weight.copy_(torch.tensor([[1., 0., 1., 3.], [3., -3., 1., 2.]]))
            fc.bias.copy_(torch.tensor([-1., 0.5]))
        selections = build_selections(fc.weight, [2, 1], True)
        self.assertEqual(selections["top_2_abs_delta"].tolist(), [0, 1])
        self.assertEqual(selections["positive_delta"].tolist(), [0])
        response = torch.tensor([[2., -1., 1., 4.], [-2., 3., -1., 0.]])
        torch.testing.assert_close(selection_logits(response, fc, selections["all"]), fc(response))
        expected = F.linear(response[:, :2], fc.weight[:, :2], fc.bias)
        torch.testing.assert_close(selection_logits(response, fc, selections["top_2_abs_delta"]), expected)
        torch.testing.assert_close(selection_logits(response, fc, np.array([], dtype=int)), fc.bias.expand(2, -1))
        self.assertLess(float((response[1, 0] * (fc.weight[1, 0] - fc.weight[0, 0])).detach()), 0)
        with self.assertRaises(ValueError):
            build_selections(fc.weight, [5], True)

    def test_checkpoint_pair_validation(self):
        original = dict(kernel_sizes=[1, 3], channels_per_branch=2, visual_length=8, seed=234,
                        cnn_activation="none", batch_size=1, lr=1e-5)
        l1 = {**original, "lambda_fc_l1": 1e-5}
        validate_pair(original, l1, 10, 10, 10)  # Legacy checkpoint: missing lambda means zero.
        with self.assertRaises(ValueError):
            validate_pair(original, {**l1, "cnn_activation": "gelu"}, 10, 10, 10)
        with self.assertRaises(ValueError):
            validate_pair(original, l1, 10, 7, 10)
        with self.assertRaises(ValueError):
            validate_pair(original, {**l1, "lambda_fc_l1": 0}, 10, 10, 10)

    def test_two_checkpoints_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, (label, length) in enumerate((("Normal", 3), ("Arson", 11))):
                path = root / f"v{i}.npy"
                np.save(path, np.random.default_rng(i).normal(size=(length, 512)).astype("float32"))
                rows.append({"path": str(path), "label": label, "video_id": f"v{i}"})
            with (root / "test.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.save(root / "gt.npy", np.r_[np.zeros(3*16), np.zeros(3*16), np.ones(8*16)])
            config = dict(kernel_sizes=[1, 3], channels_per_branch=2, visual_length=8,
                          seed=234, cnn_activation="none", batch_size=1, lr=1e-5)
            for name, l1 in (("no_l1", 0), ("l1", 1e-5)):
                model = MultiscaleCNN((1, 3), 2, 8, activation="none")
                cfg = {**config, "lambda_fc_l1": l1, "fc_l1_target": "delta" if l1 else "weight"}
                torch.save(dict(model_type="multiscale_video_cnn", epoch=1, config=cfg,
                                state_dict=model.state_dict()), root / f"{name}.pth")
            script = Path(__file__).with_name("ucf_analyze_multiscale_selection.py")
            command = [sys.executable, str(script), "--no-l1-checkpoint", str(root / "no_l1.pth"),
                       "--l1-checkpoint", str(root / "l1.pth"), "--feature-root", str(root),
                       "--test-list", str(root / "test.csv"), "--gt-path", str(root / "gt.npy"),
                       "--output-dir", str(root / "analysis"), "--expected-videos", "2",
                       "--expected-epoch", "1", "--micro-batch", "1", "--top-k", "3", "1"]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata = json.loads((root / "analysis/metadata.json").read_text())
            self.assertEqual(metadata["videos"], 2)
            self.assertEqual(metadata["gt_frames"], 14*16)
            self.assertEqual(metadata["weight_summaries"][1]["lambda_fc_l1"], 1e-5)
            self.assertEqual(metadata["weight_summaries"][1]["fc_l1_target"], "delta")
            with (root / "analysis/selection_metrics.csv").open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 8)
            self.assertEqual({row["selection"] for row in metrics},
                             {"all", "top_3_abs_delta", "top_1_abs_delta", "positive_delta"})
            self.assertTrue(all(row["frame_auc"] and row["frame_ap"] for row in metrics))
            self.assertEqual({row["model"] for row in metrics if row["selection"] == "top_3_abs_delta"}, {"no_l1", "l1"})
            with (root / "analysis/weights_l1.csv").open(newline="") as handle:
                kernels = list(csv.DictReader(handle))
            self.assertEqual(len(kernels), 4)
            self.assertEqual(sum(int(k["selected_top_3_abs_delta"]) for k in kernels), 3)
            self.assertTrue((root / "analysis/kernel_size_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
