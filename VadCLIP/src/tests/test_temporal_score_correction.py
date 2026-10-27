import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_temporal_score import TemporalScoreCorrection


class TemporalScoreCorrectionTests(unittest.TestCase):
    def test_modes_preserve_zero_initial_residual_and_mask_padding(self):
        for mode in ("feature_sequence", "feature_difference"):
            with self.subTest(mode=mode):
                branch = TemporalScoreCorrection(16, mode)
                features = torch.randn(2, 8, 16)
                delta, valid = branch(features, torch.tensor([8, 5]))

                self.assertEqual(tuple(delta.shape), (2, 8, 1))
                self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
                self.assertTrue(valid[0].all())
                self.assertTrue(valid[1, :5].all())
                self.assertFalse(valid[1, 5:].any())
                self.assertTrue(torch.equal(delta[1, 5:], torch.zeros_like(delta[1, 5:])))

    def test_temporal_convolution_receives_gradient_after_output_projection_moves(self):
        branch = TemporalScoreCorrection(16, "feature_difference")
        features = torch.randn(2, 8, 16, requires_grad=True)
        lengths = torch.tensor([8, 5])

        first_delta, _ = branch(features, lengths)
        first_delta.sum().backward()
        self.assertIsNotNone(branch.score_projection.weight.grad)

        with torch.no_grad():
            branch.score_projection.weight.normal_(mean=0.0, std=0.01)
        branch.zero_grad(set_to_none=True)
        features.grad = None
        second_delta, _ = branch(features, lengths)
        second_delta.sum().backward()

        self.assertIsNotNone(branch.temporal_conv.weight.grad)
        self.assertGreater(branch.temporal_conv.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
