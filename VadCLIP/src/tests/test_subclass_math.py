import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_subclass import (
    CLIPVADSubclass, aggregate_class_scores, aggregate_subclass_logits,
    aggregate_video_frames,
    load_subclass_bank, subclass_group_mask, subclass_video_scores,
    text_regularization,
)
from ucf_train_subclass import class_mil_loss


class StubSubclassModel(CLIPVADSubclass):
    """Exercises VadCLIP subclass wiring without downloading CLIP weights."""

    def __init__(self, n_prompts=70):
        nn.Module.__init__(self)
        self.num_class = 14
        self.classifier = nn.Linear(512, 1)
        self.mlp1 = nn.Linear(512, 512)
        self.mlp2 = nn.Linear(512, 512)
        self.prompt_embeddings = nn.Embedding(n_prompts, 512)

    def encode_video(self, visual, padding_mask, lengths):
        return visual

    def encode_textprompt(self, text):
        return self.prompt_embeddings(torch.arange(len(text)))


class SubclassMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = load_subclass_bank(Path(__file__).resolve().parents[3] / "code" / "ucf_subclass_bank_5x.json")
        cls.groups_10x = load_subclass_bank(Path(__file__).resolve().parents[3] / "code" / "ucf_subclass_bank_10x.json")
        cls.groups_semantic_10x = load_subclass_bank(Path(__file__).resolve().parents[3] / "code" / "ucf_subclass_bank_semantic_10x.json")
        cls.groups_variable = load_subclass_bank(Path(__file__).resolve().parents[3] / "code" / "ucf_subclass_bank_description_other.json")

    def test_bank_shape(self):
        self.assertEqual((len(self.groups), len(self.groups[0])), (14, 5))
        self.assertEqual((len(self.groups_10x), len(self.groups_10x[0])), (14, 10))
        self.assertEqual((len(self.groups_semantic_10x), len(self.groups_semantic_10x[0])), (14, 10))
        self.assertEqual(self.groups_semantic_10x[6][0], "Vehicle Explosion")
        self.assertEqual([len(group) for group in self.groups_variable], [1, 6, 5, 5, 5, 6, 4, 5, 9, 7, 6, 6, 7, 7])

    def test_variable_bank_forward_mask_and_gradients(self):
        groups = self.groups_variable
        mask = subclass_group_mask(groups)
        model = StubSubclassModel(n_prompts=sum(map(len, groups)))
        visual = torch.randn(2, 6, 512, requires_grad=True)
        class_text, logits1, subclass_logits = model(visual, None, groups, [5, 6])
        self.assertEqual(subclass_logits.shape, (2, 6, 14, 9))
        self.assertTrue(torch.isneginf(subclass_logits[:, :, 0, 1:]).all())
        scores = subclass_video_scores(subclass_logits, [5, 6])
        class_scores = aggregate_class_scores(scores, "soft", subclass_mask=mask)
        frame_map, weights, _ = aggregate_video_frames(
            subclass_logits[0, :5], "soft", subclass_mask=mask
        )
        self.assertEqual(class_scores.shape, (2, 14))
        self.assertEqual(frame_map.shape, (5, 14))
        self.assertTrue(torch.isfinite(frame_map).all())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(14)))
        self.assertEqual(weights[0, 1:].abs().sum().item(), 0)
        loss = class_scores.sum() + text_regularization(class_text) + logits1.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.prompt_embeddings.weight.grad)
        self.assertIsNotNone(visual.grad)

    def test_10x_forward_and_gradients(self):
        for groups in (self.groups_10x, self.groups_semantic_10x):
            with self.subTest(first_subclass=groups[0][0]):
                model = StubSubclassModel(n_prompts=140)
                visual = torch.randn(2, 6, 512, requires_grad=True)
                class_text, logits1, subclass_logits = model(visual, None, groups, [5, 6])
                self.assertEqual(class_text.shape, (14, 512))
                self.assertEqual(logits1.shape, (2, 6, 1))
                self.assertEqual(subclass_logits.shape, (2, 6, 14, 10))
                scores = subclass_video_scores(subclass_logits, [5, 6])
                self.assertEqual(scores.shape, (2, 14, 10))
                frame_map, weights, _ = aggregate_video_frames(subclass_logits[0, :5], "soft")
                self.assertEqual(frame_map.shape, (5, 14))
                self.assertEqual(weights.shape, (14, 10))
                loss = aggregate_class_scores(scores, "soft").sum() + text_regularization(class_text)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(model.prompt_embeddings.weight.grad)
                self.assertIsNotNone(visual.grad)

    def test_topk_excludes_padding(self):
        logits = torch.zeros(2, 6, 14, 5, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 6, 2] = 10
            logits[0, 5] = 100
        scores = subclass_video_scores(logits, [5, 6])
        self.assertEqual(scores.shape, (2, 14, 5))
        self.assertEqual(scores[0, 6, 2].item(), 10)
        self.assertEqual(scores[0, 6, 0].item(), 0)

    def test_subclasses_are_aggregated_before_temporal_topk(self):
        logits = torch.zeros(1, 16, 14, 2)
        logits[0, 0, 1, 0] = 10
        logits[0, 1, 1, 1] = 10
        mask = torch.ones(14, 2, dtype=torch.bool)

        class_logits = aggregate_subclass_logits(logits, "max", subclass_mask=mask)
        self.assertEqual(class_logits[0, 0, 1].item(), 10)
        self.assertEqual(class_logits[0, 1, 1].item(), 10)
        loss, usage = class_mil_loss(logits, torch.tensor([1]), [16], "max", 1.0, mask)

        # k=2: class-level Top-K can select evidence supplied by two different
        # subclasses. The old Top-K-first order would average each one with 0.
        self.assertLess(loss.item(), 0.001)
        self.assertTrue(torch.allclose(usage[0, 1], torch.tensor([0.5, 0.5])))

    def test_aggregation_and_gradients(self):
        model = StubSubclassModel()
        visual = torch.randn(2, 6, 512, requires_grad=True)
        class_text, logits1, subclass_logits = model(visual, None, self.groups, [5, 6])
        self.assertEqual(class_text.shape, (14, 512))
        self.assertEqual(logits1.shape, (2, 6, 1))
        self.assertEqual(subclass_logits.shape, (2, 6, 14, 5))
        scores = subclass_video_scores(subclass_logits, [5, 6])
        for mode in ("max", "soft"):
            self.assertEqual(aggregate_class_scores(scores, mode).shape, (2, 14))
        frame_map, weights, _ = aggregate_video_frames(subclass_logits[0, :5], "soft")
        self.assertEqual(frame_map.shape, (5, 14))
        self.assertEqual(weights.shape, (14, 5))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(14)))
        loss = aggregate_class_scores(scores, "soft").sum() + text_regularization(class_text)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.prompt_embeddings.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNotNone(visual.grad)


if __name__ == "__main__":
    unittest.main()
