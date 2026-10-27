import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucf_train import hard_pair_soft_ranking_loss


class HardPairRankingTests(unittest.TestCase):
    def test_loss_is_finite_and_both_branches_receive_gradients(self):
        torch.manual_seed(7)
        logits1 = torch.randn(4, 8, 1, requires_grad=True)
        logits2 = torch.randn(4, 8, 14, requires_grad=True)
        labels = torch.zeros(4, 14)
        labels[:2, 0] = 1
        labels[2:, 1] = 1
        lengths = torch.tensor([8, 5, 8, 5])

        loss, active = hard_pair_soft_ranking_loss(
            logits1, logits2, labels, lengths, 0.30, 0.10, 0.25, 'both'
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
        self.assertGreaterEqual(active.item(), 0)
        self.assertLessEqual(active.item(), 1)
        loss.backward()
        self.assertGreater(logits1.grad.abs().sum().item(), 0)
        self.assertGreater(logits2.grad.abs().sum().item(), 0)

    def test_padding_does_not_change_loss(self):
        logits1 = torch.zeros(2, 8, 1)
        logits2 = torch.zeros(2, 8, 14)
        labels = torch.zeros(2, 14)
        labels[0, 0] = 1
        labels[1, 1] = 1
        lengths = torch.tensor([3, 8])

        original, _ = hard_pair_soft_ranking_loss(
            logits1, logits2, labels, lengths, 0.30, 0.10, 1.0, 'both'
        )
        logits1[0, 3:] = 100
        logits2[0, 3:, 0] = -100
        changed, _ = hard_pair_soft_ranking_loss(
            logits1, logits2, labels, lengths, 0.30, 0.10, 1.0, 'both'
        )
        self.assertTrue(torch.allclose(original, changed))

    def test_hard_normal_is_not_hidden_by_easy_normals(self):
        logits1 = torch.tensor([[[3.0]], [[-3.0]], [[-3.0]], [[0.0]]])
        logits2 = torch.zeros(4, 1, 14)
        labels = torch.zeros(4, 14)
        labels[:3, 0] = 1
        labels[3, 1] = 1

        loss, active = hard_pair_soft_ranking_loss(
            logits1, logits2, labels, torch.ones(4, dtype=torch.long),
            0.30, 0.10, 0.25, 'classifier'
        )
        self.assertGreater(active.item(), 0)
        self.assertGreater(loss.item(), 0.3)


if __name__ == '__main__':
    unittest.main()
