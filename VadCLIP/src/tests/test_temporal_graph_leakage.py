import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import CLIPVAD, QuickGELU, Transformer
from ucf_analyze_temporal_graph_leakage import (
    capture_encode_video,
    exposure_affinity_association,
    graph_forward,
    paired_group_differences,
    per_video_means,
    remove_sources,
    similarity_rows,
)
from utils.layers import GraphConvolution


class CpuDistance(nn.Module):
    def forward(self, batch_size, max_seqlen):
        index = torch.arange(max_seqlen)
        distance = (index[:, None] - index[None, :]).abs()
        return torch.exp(-distance.float()).unsqueeze(0).expand(batch_size, -1, -1)


class TinyVad(nn.Module):
    adj4 = CLIPVAD.adj4
    encode_video = CLIPVAD.encode_video

    def __init__(self):
        super().__init__()
        self.visual_length = 8
        self.device = torch.device("cpu")
        self.frame_position_embeddings = nn.Embedding(8, 4)
        mask = torch.full((8, 8), float("-inf"))
        mask[:4, :4] = 0
        mask[4:, 4:] = 0
        self.temporal = Transformer(4, 2, 1, mask)
        self.gc1 = GraphConvolution(4, 2)
        self.gc2 = GraphConvolution(2, 2)
        self.gc3 = GraphConvolution(4, 2)
        self.gc4 = GraphConvolution(2, 2)
        self.disAdj = CpuDistance()
        self.linear = nn.Linear(4, 4)
        self.gelu = QuickGELU()
        self.mlp2 = nn.Identity()
        self.classifier = nn.Linear(4, 1)


class LeakageTests(unittest.TestCase):
    def test_capture_preserves_forward_and_graph_replay(self):
        torch.manual_seed(2)
        model = TinyVad().eval()
        visual = torch.randn(2, 8, 4)
        with torch.no_grad():
            expected = model.encode_video(visual, None, [8, 5])
            captured = capture_encode_video(model, visual, [8, 5])
            replayed = graph_forward(model, captured["temporal"],
                                     captured["similarity_adj"], captured["distance_adj"])
        self.assertEqual(len(captured["attention"]), 2)
        self.assertEqual(tuple(captured["attention"][0].shape), (2, 1, 8, 8))
        torch.testing.assert_close(captured["final_visual"], expected)
        torch.testing.assert_close(replayed, expected)

    def test_graph_intervention_only_changes_normal_targets(self):
        adjacency = torch.full((1, 4, 4), 0.25)
        gt = torch.tensor([[False, False, False, True]])
        modified, removed = remove_sources(adjacency, gt, [4], "anomaly", 42)
        self.assertEqual(removed, 3)
        self.assertEqual(float(modified[0, 0, 3]), 0)
        self.assertEqual(float(modified[0, 2, 3]), 0)
        torch.testing.assert_close(modified[0, [0, 1, 2]].sum(dim=-1), torch.ones(3))
        torch.testing.assert_close(modified[0, 3], adjacency[0, 3])
        random_control, random_count = remove_sources(adjacency, gt, [4], "normal", 42)
        self.assertEqual(random_count, removed)
        self.assertTrue(torch.all(torch.diag(random_control[0]) > 0))

    def test_video_paired_summary_uses_matched_groups(self):
        rows = [
            {"video_id": "a", "class_name": "X", "group": "false_high", "value": 0.8},
            {"video_id": "a", "class_name": "X", "group": "false_high", "value": 0.6},
            {"video_id": "a", "class_name": "X", "group": "true_low", "value": 0.2},
            {"video_id": "b", "class_name": "X", "group": "true_low", "value": 0.1},
        ]
        video_rows = per_video_means(rows, ("value",))
        paired = paired_group_differences(video_rows, ("value",))
        self.assertEqual(paired["value"]["n"], 1)
        self.assertAlmostEqual(paired["value"]["mean"], 0.5)

    def test_stage_similarity_uses_three_groups(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        groups = np.array(["false_high", "true_high", "true_low"])
        rows = similarity_rows({stage: features for stage in ("raw_clip", "temporal", "final_visual")},
                               groups, "video", "X")
        self.assertEqual(len(rows), 3)
        self.assertGreater(rows[0]["delta"], 0)

    def test_exposure_association_matches_video_ids(self):
        stage_rows = [
            {"video_id": video, "stage": stage, "delta": value}
            for video, values in (("a", (0.0, 0.1, 0.3)), ("b", (0.0, 0.1, 0.2)),
                                  ("c", (0.0, 0.1, 0.1)))
            for stage, value in zip(("raw_clip", "temporal", "final_visual"), values)
        ]
        exposures = [
            {"video_id": video, "group": "false_high", "temporal_excess": value,
             "similarity_graph_excess": value, "distance_graph_excess": value}
            for video, value in (("a", 0.3), ("b", 0.2), ("c", 0.1))
        ]
        rows, correlations = exposure_affinity_association(stage_rows, exposures)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(
            correlations["similarity_graph_excess_vs_delta_temporal_to_final"], 1.0)


if __name__ == "__main__":
    unittest.main()
