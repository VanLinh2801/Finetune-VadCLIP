import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_video_context import TemporalContextHead


class TemporalContextHeadTests(unittest.TestCase):
    def test_shapes_and_zero_initialized_residual(self):
        head = TemporalContextHead(16, bin_size=4, hidden_width=8, max_delta=0.1)
        sequences = [torch.randn(9, 16), torch.randn(5, 16)]

        deltas, video_logits, tokens, token_lengths = head(sequences)

        self.assertEqual([tuple(delta.shape) for delta in deltas], [(9, 1), (5, 1)])
        self.assertEqual(tuple(video_logits.shape), (2,))
        self.assertEqual(tuple(tokens.shape), (2, 3, 16))
        self.assertEqual(token_lengths.tolist(), [3, 2])
        self.assertTrue(torch.equal(deltas[0], torch.zeros_like(deltas[0])))
        self.assertTrue(torch.equal(deltas[1], torch.zeros_like(deltas[1])))
        self.assertTrue(torch.isfinite(video_logits).all())

    def test_video_bce_backpropagates_through_gru(self):
        head = TemporalContextHead(16, bin_size=4, hidden_width=8, max_delta=0.1)
        sequences = [torch.randn(9, 16), torch.randn(5, 16)]
        _, video_logits, _, _ = head(sequences)
        targets = torch.tensor([0.0, 1.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(video_logits, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(head.gru.weight_ih_l0.grad)
        self.assertGreater(head.gru.weight_ih_l0.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(head.video_classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
