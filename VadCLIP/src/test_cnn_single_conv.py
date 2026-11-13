import unittest

import torch

from model_cnn_vadclip import CLIPVADCNN, SingleConvKernelResponse, SingleConvReadout
from ucf_cnn_common import aggregate_video_logits
from ucf_train_cnn_vadclip import build_parser


class SingleConvTests(unittest.TestCase):
    def test_competitive_response_gelu_precedes_group_max(self):
        model = SingleConvReadout(input_width=1, channels=4, kernel_size=1,
                                 score_mode="competitive_response_gelu")
        raw = torch.tensor([-3., -1., -2., -0.5])
        with torch.no_grad():
            model.conv.weight.zero_()
            model.conv.bias.copy_(raw)
        activated = torch.nn.functional.gelu(raw)
        expected = activated[:2].max() - activated[2:].max()
        scores = model(torch.ones(1, 3, 1), torch.tensor([2]))
        self.assertTrue(torch.allclose(scores[0, :2, 0], expected.expand(2)))
        self.assertEqual(float(scores[0, 2, 0].detach()), 0.)
        # GELU is not monotonic over all negative values: activation must precede max.
        after_max = torch.nn.functional.gelu(raw[:2].max()) - torch.nn.functional.gelu(raw[2:].max())
        self.assertFalse(torch.allclose(expected, after_max))
        scores.sum().backward()
        self.assertGreater(float(model.conv.weight.grad[:2].abs().sum()), 0.)
        self.assertGreater(float(model.conv.weight.grad[2:].abs().sum()), 0.)

    def test_gelu_competition_padding_and_parameter_compatibility(self):
        torch.manual_seed(234)
        raw = SingleConvReadout(score_mode="competitive_response")
        torch.manual_seed(234)
        gelu = SingleConvReadout(score_mode="competitive_response_gelu")
        self.assertEqual(set(raw.state_dict()), set(gelu.state_dict()))
        self.assertTrue(torch.equal(raw.conv.weight, gelu.conv.weight))
        features = torch.randn(2, 20, 512)
        lengths = torch.tensor([9, 20])
        altered = features.clone()
        altered[0, 9:] = 1e4
        self.assertTrue(torch.allclose(gelu(features, lengths), gelu(altered, lengths)))
        with self.assertRaises(ValueError):
            SingleConvReadout(channels=3, score_mode="competitive_response_gelu")
        with self.assertRaises(ValueError):
            CLIPVADCNN(cnn_architecture="two_block", score_mode="competitive_response_gelu")

    def test_single_conv_channel_max_and_video_max(self):
        model = SingleConvKernelResponse(input_width=512, channels=128, kernel_size=5)
        self.assertEqual(tuple(model.conv.weight.shape), (128, 512, 5))
        self.assertEqual(len([module for module in model.modules() if isinstance(module, torch.nn.Conv1d)]), 1)
        features = torch.randn(2, 12, 512)
        lengths = torch.tensor([7, 12])
        altered = features.clone()
        altered[0, 7:] = 1000.0
        scores = model(features, lengths)
        changed = model(altered, lengths)
        response = torch.nn.functional.gelu(model.conv(
            (features * (torch.arange(12)[None, :] < lengths[:, None]).unsqueeze(-1)).transpose(1, 2)
        )).amax(dim=1).unsqueeze(-1)
        expected = torch.nn.functional.softplus(model.raw_scale) * response + model.bias
        self.assertEqual(tuple(scores.shape), (2, 12, 1))
        self.assertTrue(torch.allclose(scores[0, :7], expected[0, :7]))
        self.assertTrue(torch.allclose(scores[0, :7], changed[0, :7]))
        self.assertTrue(torch.equal(scores[0, 7:], torch.zeros_like(scores[0, 7:])))
        video_logits = aggregate_video_logits(scores, lengths, "max")
        self.assertTrue(torch.allclose(video_logits[0], scores[0, :7, 0].max()))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(video_logits, torch.tensor([0., 1.]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.conv.weight.grad)
        self.assertGreater(float(model.conv.weight.grad.abs().sum()), 0.0)

    def test_training_options_remain_separate(self):
        args = build_parser().parse_args([
            "--feature-root", "features", "--train-list", "train.csv",
            "--epoch-checkpoint-dir", "epochs", "--output-model-path", "model.pth",
            "--results-dir", "results",
        ])
        self.assertEqual(args.cnn_architecture, "two_block")
        self.assertEqual(args.aggregation, "topk")

    def test_readout_ablation_shares_identical_initial_conv(self):
        torch.manual_seed(234)
        maximum = SingleConvKernelResponse(score_mode="kernel_response")
        torch.manual_seed(234)
        weighted = SingleConvKernelResponse(score_mode="scoring_head")
        self.assertTrue(torch.equal(maximum.conv.weight, weighted.conv.weight))
        self.assertEqual(set(maximum.state_dict()), {"conv.weight", "conv.bias", "raw_scale", "bias"})
        self.assertEqual(set(weighted.state_dict()), {"conv.weight", "conv.bias", "scorer.weight", "scorer.bias"})
        features = torch.randn(2, 32, 512)
        lengths = torch.tensor([17, 32])
        for model in (maximum, weighted):
            scores = model(features, lengths)
            self.assertEqual(tuple(scores.shape), (2, 32, 1))
            self.assertTrue(torch.equal(scores[0, 17:], torch.zeros_like(scores[0, 17:])))
            video_logits = aggregate_video_logits(scores, lengths, "topk")
            video_logits.sum().backward()
            self.assertGreater(float(model.conv.weight.grad.abs().sum()), 0.0)

    def test_topk_pooling_uses_multiple_valid_snippets(self):
        logits = torch.zeros(1, 32, 1, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0] = 3.0
            logits[0, 1, 0] = 2.0
            logits[0, 16:, 0] = 100.0
        pooled = aggregate_video_logits(logits, torch.tensor([16]), "topk")
        self.assertTrue(torch.allclose(pooled, torch.tensor([3.0])))
        pooled = aggregate_video_logits(logits, torch.tensor([32]), "topk")
        self.assertTrue(torch.allclose(pooled, torch.tensor([100.0])))

        with torch.no_grad():
            logits[0, 16:, 0] = -100.0
        pooled = aggregate_video_logits(logits, torch.tensor([32]), "topk")
        self.assertTrue(torch.allclose(pooled, torch.tensor([2.5])))
        pooled.backward()
        self.assertGreater(float(logits.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 16:].abs().sum()), 0.0)

    def test_competition_is_signed_raw_response_and_normal_evidence_suppresses(self):
        model = SingleConvReadout(input_width=1, channels=4, kernel_size=1,
                                 score_mode="competitive_response")
        with torch.no_grad():
            model.conv.weight.zero_()
            model.conv.bias.copy_(torch.tensor([-8., -9., -1., -2.]))
        features = torch.ones(1, 3, 1)
        scores = model(features, torch.tensor([2]))
        self.assertTrue(torch.equal(scores[0, :2, 0], torch.tensor([-7., -7.])))
        self.assertEqual(float(scores[0, 2, 0]), 0.0)
        self.assertTrue(torch.allclose(aggregate_video_logits(scores, torch.tensor([2]), "topk"), torch.tensor([-7.])))
        with torch.no_grad():
            model.conv.bias[2:] += 2.0
        suppressed = model(features, torch.tensor([2]))
        self.assertTrue(torch.equal(suppressed[0, :2, 0], torch.tensor([-9., -9.])))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            aggregate_video_logits(suppressed, torch.tensor([2]), "topk"), torch.ones(1)
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(float(model.conv.weight.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(model.conv.weight.grad[2, 0, 0]), 0.0)
        self.assertEqual(set(model.state_dict()), {"conv.weight", "conv.bias", "bias"})

    def test_competition_padding_and_checkpoint_roundtrip(self):
        model = SingleConvReadout(score_mode="competitive_response")
        clone = SingleConvReadout(score_mode="competitive_response")
        clone.load_state_dict(model.state_dict(), strict=True)
        features = torch.randn(2, 32, 512)
        lengths = torch.tensor([17, 32])
        altered = features.clone()
        altered[0, 17:] = 1e4
        scores = model(features, lengths)
        self.assertEqual(tuple(scores.shape), (2, 32, 1))
        self.assertTrue(torch.allclose(scores, clone(altered, lengths)))
        self.assertTrue(torch.isfinite(scores).all())

    def test_competition_rejects_invalid_architecture_and_channels(self):
        for channels in (1, 3):
            with self.assertRaises(ValueError):
                SingleConvReadout(channels=channels, score_mode="competitive_response")
        with self.assertRaises(ValueError):
            CLIPVADCNN(cnn_architecture="two_block", score_mode="competitive_response")

    def test_competition_is_accepted_by_train_and_eval_parsers(self):
        from ucf_evaluate_cnn_vadclip import build_parser as eval_parser
        train_args = build_parser().parse_args([
            "--feature-root", "features", "--train-list", "train.csv",
            "--epoch-checkpoint-dir", "epochs", "--output-model-path", "model.pth",
            "--results-dir", "results", "--cnn-architecture", "single_conv",
            "--score-mode", "competitive_response",
        ])
        eval_args = eval_parser().parse_args([
            "--model-path", "model.pth", "--feature-root", "features", "--test-list", "test.csv",
            "--gt-path", "gt.npy", "--gt-segment-path", "segments.npy", "--gt-label-path", "labels.npy",
            "--evaluate-dir", "eval", "--analyze-dir", "analyze", "--cnn-architecture", "single_conv",
            "--score-mode", "competitive_response",
        ])
        self.assertEqual(train_args.score_mode, eval_args.score_mode)
        self.assertEqual(train_args.aggregation, "topk")

    @unittest.skipUnless(torch.cuda.is_available(), "Full VadCLIP smoke test requires CUDA")
    def test_competition_full_model_cuda(self):
        from ucf_cnn_common import TRAIN_LABEL_MAP, alignment_mil_loss, text_regularization_loss
        from utils.tools import get_prompt_text

        device = torch.device("cuda")
        model = CLIPVADCNN(14, 512, 256, 512, 1, 2, 8, 10, 10, device,
                          kernel_size=5, hidden_width=128, cnn_architecture="single_conv",
                          score_mode="competitive_response").to(device)
        lengths = torch.tensor([170, 256], device=device)
        text, logits1, logits2 = model(torch.randn(2, 256, 512, device=device), None,
                                      get_prompt_text(TRAIN_LABEL_MAP), lengths)
        self.assertEqual(tuple(logits1.shape), (2, 256, 1))
        self.assertEqual(tuple(logits2.shape), (2, 256, 14))
        labels = torch.zeros(2, 14, device=device)
        labels[0, 0], labels[1, 6] = 1, 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            aggregate_video_logits(logits1, lengths, "topk"), torch.tensor([0., 1.], device=device)
        ) + alignment_mil_loss(logits2, labels, lengths) + text_regularization_loss(text)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad = model.cnn_classifier.conv.weight.grad
        self.assertGreater(float(grad[:64].abs().sum()), 0.0)
        self.assertGreater(float(grad[64:].abs().sum()), 0.0)
        self.assertTrue(any(p.grad is not None for p in model.temporal.parameters()))
        self.assertTrue(all(p.grad is None for p in model.clipmodel.parameters()))


if __name__ == "__main__":
    unittest.main()
