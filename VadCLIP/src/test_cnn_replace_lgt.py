import unittest

import torch

from model_cnn_replace_lgt import TemporalCNNBackbone
from ucf_cnn_common import aggregate_video_logits, alignment_mil_loss, text_regularization_loss


class TemporalCNNReplaceLGTTests(unittest.TestCase):
    def test_backbone_masks_padding_and_backward(self):
        lengths = torch.tensor([11, 32])
        features = torch.randn(2, 32, 512)
        altered = features.clone()
        altered[0, 11:] = 1000.0
        for mode_kernel in (3, 7):
            model = TemporalCNNBackbone(kernel_size=mode_kernel, num_blocks=2, hidden_width=128)
            model.eval()
            hidden, response, mask = model(features, lengths)
            changed, _, _ = model(altered, lengths)
            self.assertEqual(hidden.shape, (2, 32, 128))
            self.assertEqual(response.shape, (2, 32, 128))
            self.assertTrue(torch.equal(mask[0, :11], torch.ones(11, dtype=torch.bool)))
            self.assertTrue(torch.allclose(hidden[0, :11], changed[0, :11], atol=1e-5))
            self.assertEqual(float(hidden[0, 11:].detach().abs().sum()), 0.0)
            hidden[:, :11].sum().backward()
            self.assertIsNotNone(model.projection[0].weight.grad)
            self.assertIsNotNone(model.blocks[0].conv1.weight.grad)

    def test_num_blocks_is_configurable(self):
        features = torch.randn(1, 16, 512)
        lengths = torch.tensor([12])
        for num_blocks in (1, 3, 4):
            model = TemporalCNNBackbone(num_blocks=num_blocks, kernel_size=3, hidden_width=64)
            self.assertEqual(len(model.blocks), num_blocks)
            hidden, response, mask = model(features, lengths)
            self.assertEqual(hidden.shape, (1, 16, 64))
            self.assertEqual(float(hidden[0, 12:].detach().abs().sum()), 0.0)
            self.assertTrue(mask[0, :12].all())

    def test_large_kernels_keep_sequence_length(self):
        features = torch.randn(1, 12, 512)
        lengths = torch.tensor([9])
        for kernel_size in (7, 9):
            model = TemporalCNNBackbone(kernel_size=kernel_size, num_blocks=2)
            hidden, _, _ = model(features, lengths)
            self.assertEqual(hidden.shape, (1, 12, 128))
            self.assertTrue(torch.isfinite(hidden).all())
            self.assertEqual(float(hidden[0, 9:].detach().abs().sum()), 0.0)

    def test_pooling_and_auxiliary_losses(self):
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

        class_logits = torch.randn(2, 32, 14, requires_grad=True)
        labels = torch.zeros(2, 14)
        labels[0, 0] = 1
        labels[1, 6] = 1
        text = torch.randn(14, 512, requires_grad=True)
        loss = alignment_mil_loss(class_logits, labels, torch.tensor([17, 32]))
        loss = loss + text_regularization_loss(text)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(class_logits.grad)
        self.assertIsNotNone(text.grad)


@unittest.skipUnless(torch.cuda.is_available(), "Full model smoke needs CUDA for CLIP runtime")
class CLIPVADCNNReplaceLGTSmokeTests(unittest.TestCase):
    def test_replaces_lgt_and_trains_cnn_only(self):
        from model_cnn_replace_lgt import CLIPVADCNNReplaceLGT
        from ucf_cnn_common import TRAIN_LABEL_MAP

        device = torch.device("cuda")
        visual = torch.randn(2, 64, 512, device=device)
        lengths = torch.tensor([40, 64], device=device)
        labels = torch.zeros(2, 14, device=device)
        labels[0, 0] = 1
        labels[1, 6] = 1
        prompts = list(TRAIN_LABEL_MAP.values())

        for mode in ("scoring_head", "kernel_response"):
            model = CLIPVADCNNReplaceLGT(
                14, 512, 64, 512, 1, 2, 8, 10, 10, device,
                kernel_size=3, score_mode=mode, hidden_width=128, num_blocks=3,
            ).to(device)
            for name in (
                "temporal", "gc1", "gc2", "gc3", "gc4", "disAdj", "linear",
                "gelu", "classifier", "mlp2", "frame_position_embeddings",
            ):
                self.assertFalse(hasattr(model, name), name)
            self.assertEqual(len(model.cnn_backbone.blocks), 3)

            text, logits1, logits2 = model(visual, None, prompts, lengths)
            self.assertEqual(logits1.shape, (2, 64, 1))
            self.assertEqual(logits2.shape, (2, 64, 14))
            self.assertEqual(float(logits1[0, 40:].detach().abs().sum()), 0.0)

            video_logits = aggregate_video_logits(logits1, lengths, "topk")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                video_logits, torch.tensor([0.0, 1.0], device=device)
            )
            loss = loss + alignment_mil_loss(logits2, labels, lengths) + text_regularization_loss(text)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.cnn_backbone.projection[0].weight.grad)
            if mode == "scoring_head":
                self.assertIsNotNone(model.scorer.weight.grad)
            else:
                self.assertIsNotNone(model.raw_scale.grad)
            if not isinstance(model.feature_proj, torch.nn.Identity):
                self.assertIsNotNone(model.feature_proj.weight.grad)
            self.assertTrue(all(p.grad is None for p in model.clipmodel.parameters()))
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
