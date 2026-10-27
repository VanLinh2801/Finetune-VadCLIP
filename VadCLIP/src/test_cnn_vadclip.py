import unittest

import torch

from model_cnn_vadclip import TemporalCNNClassifier, TemporalConvBlock
from ucf_cnn_common import aggregate_video_logits, alignment_mil_loss, text_regularization_loss


class TemporalCNNTests(unittest.TestCase):
    def test_pooling_ignores_padding_and_uses_ceil(self):
        logits = torch.full((2, 32, 1), -2.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 16:] = 100.0
            logits[1, 0] = 4.0
            logits[1, 1] = 2.0
        lengths = torch.tensor([16, 32])
        topk = aggregate_video_logits(logits, lengths, "topk")
        maximum = aggregate_video_logits(logits, lengths, "max")
        self.assertTrue(torch.allclose(topk, torch.tensor([-2.0, 3.0])))
        self.assertTrue(torch.allclose(maximum, torch.tensor([-2.0, 4.0])))
        topk.sum().backward()
        self.assertEqual(float(logits.grad[0, 16:].abs().sum()), 0.0)

    def test_scoring_modes_mask_padding_and_backward(self):
        lengths = torch.tensor([11, 32])
        features = torch.randn(2, 32, 512)
        altered = features.clone()
        altered[0, 11:] = 1000.0
        for mode in ("scoring_head", "kernel_response"):
            model = TemporalCNNClassifier(kernel_size=3, score_mode=mode)
            model.eval()
            scores = model(features, lengths)
            changed = model(altered, lengths)
            self.assertEqual(scores.shape, (2, 32, 1))
            self.assertTrue(torch.allclose(scores[0, :11], changed[0, :11], atol=1e-5))
            self.assertEqual(float(scores[0, 11:].detach().abs().sum()), 0.0)
            scores[:, :11].sum().backward()
            self.assertIsNotNone(model.projection[0].weight.grad)
            self.assertIsNotNone(model.blocks[0].conv1.weight.grad)

    def test_auxiliary_losses_are_finite(self):
        logits = torch.randn(2, 32, 14, requires_grad=True)
        labels = torch.zeros(2, 14)
        labels[0, 0] = 1
        labels[1, 6] = 1
        text = torch.randn(14, 512, requires_grad=True)
        loss = alignment_mil_loss(logits, labels, torch.tensor([17, 32]))
        loss = loss + text_regularization_loss(text)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(text.grad)

    def test_large_kernels_keep_sequence_length(self):
        features = torch.randn(1, 12, 512)
        lengths = torch.tensor([9])
        for kernel_size in (7, 9):
            model = TemporalCNNClassifier(kernel_size=kernel_size)
            scores = model(features, lengths)
            self.assertEqual(scores.shape, (1, 12, 1))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertEqual(float(scores[0, 9:].detach().abs().sum()), 0.0)
            scores[:, :9].sum().backward()
            self.assertIsNotNone(model.blocks[0].conv1.weight.grad)

    def test_wider_projection_keeps_128_cnn_filters_and_backward(self):
        model = TemporalCNNClassifier(projection_width=256, hidden_width=128, kernel_size=5)
        self.assertEqual(tuple(model.projection[0].weight.shape), (256, 512))
        self.assertEqual(tuple(model.blocks[0].conv1.weight.shape), (128, 256, 5))
        self.assertEqual(tuple(model.blocks[0].conv2.weight.shape), (128, 128, 5))
        self.assertEqual(tuple(model.blocks[1].conv1.weight.shape), (128, 128, 5))
        self.assertEqual(tuple(model.blocks[0].skip_projection.weight.shape), (128, 256))
        scores = model(torch.randn(2, 12, 512), torch.tensor([8, 12]))
        self.assertEqual(tuple(scores.shape), (2, 12, 1))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(float(scores[0, 8:].detach().abs().sum()), 0.0)
        scores[:, :8].sum().backward()
        self.assertIsNotNone(model.projection[0].weight.grad)
        self.assertIsNotNone(model.blocks[0].conv1.weight.grad)
        self.assertIsNotNone(model.blocks[0].skip_projection.weight.grad)

    def test_64_filters_with_128_projection(self):
        model = TemporalCNNClassifier(projection_width=128, hidden_width=64, kernel_size=5)
        self.assertEqual(tuple(model.projection[0].weight.shape), (128, 512))
        self.assertEqual(tuple(model.blocks[0].conv1.weight.shape), (64, 128, 5))
        self.assertEqual(tuple(model.blocks[0].skip_projection.weight.shape), (64, 128))
        self.assertEqual(tuple(model.blocks[1].conv1.weight.shape), (64, 64, 5))
        scores = model(torch.randn(2, 12, 512), torch.tensor([8, 12]))
        self.assertEqual(tuple(scores.shape), (2, 12, 1))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(float(scores[0, 8:].detach().abs().sum()), 0.0)
        scores[:, :8].sum().backward()
        self.assertIsNotNone(model.blocks[0].skip_projection.weight.grad)
        self.assertIsNotNone(model.blocks[1].conv2.weight.grad)

    def test_default_projection_preserves_old_checkpoint_keys(self):
        original = TemporalCNNClassifier(hidden_width=128, kernel_size=5)
        explicit = TemporalCNNClassifier(hidden_width=128, projection_width=128, kernel_size=5)
        explicit.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(any("skip_projection" in key for key in original.state_dict()))

    def test_residual_toggle_changes_only_skip_connection(self):
        features = torch.randn(2, 11, 128, requires_grad=True)
        mask = torch.ones(2, 11, dtype=torch.bool)
        with_skip = TemporalConvBlock(128, 5, use_residual=True)
        without_skip = TemporalConvBlock(128, 5, use_residual=False)
        without_skip.load_state_dict(with_skip.state_dict(), strict=True)
        on, on_response = with_skip(features, mask)
        off, off_response = without_skip(features, mask)
        self.assertTrue(torch.allclose(on_response, off_response))
        self.assertTrue(torch.allclose(on, with_skip.norm(features + on_response)))
        self.assertTrue(torch.allclose(off, without_skip.norm(off_response)))
        off.sum().backward()
        self.assertIsNotNone(without_skip.conv1.weight.grad)


if __name__ == "__main__":
    unittest.main()
