import json
import contextlib
import io
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from model_cnn_vadclip import SingleConvReadout
from ucf_analyze_competitive_kernels import (
    build_parser, decompose, describe, frame_metrics, load_config, main, make_groups, winner_summary,
)


class CompetitiveDiagnosticsTests(unittest.TestCase):
    def test_decomposition_matches_both_models_and_ignores_padding(self):
        for mode in ("competitive_response", "competitive_response_gelu"):
            model = SingleConvReadout(input_width=4, channels=8, kernel_size=5, score_mode=mode)
            features = torch.randn(3, 12, 4)
            lengths = torch.tensor([3, 11, 12])
            valid = torch.arange(12)[None, :] < lengths[:, None]
            features[~valid] = 1e5
            raw = model.conv((features * valid.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
            parts = decompose(raw, mode, model.bias)
            actual = model(features, lengths).squeeze(-1)
            torch.testing.assert_close(actual[valid], parts["logit"][valid])
            for side in ("abnormal", "normal"):
                torch.testing.assert_close(parts[side + "_credit"].sum(-1), torch.ones(3, 12))

    def test_gelu_before_max_and_tie_credit(self):
        raw = torch.tensor([[-3., -1., -2., -.5], [1., 1., 2., 2.]])
        parts = decompose(raw, "competitive_response_gelu", torch.tensor(.3))
        self.assertEqual(int(parts["abnormal_winner"][0]), 0)
        self.assertEqual(int(parts["abnormal_tie_count"][1]), 2)
        torch.testing.assert_close(parts["normal_credit"][1], torch.tensor([.5, .5]))

    def test_topk_is_whole_video_and_groups_cover_all_positions(self):
        score = np.array([.8, .1, .9, .7, .6, .2])
        pos = np.array([16, 16, 0, 0, 0, 0])
        groups = make_groups(score, ["Explosion", "Normal"], pos, [0, 4, 6], .25)
        self.assertEqual(groups.tolist(), ["missed_anomaly", "missed_anomaly", "false_high_abnormal",
                                          "true_low", "false_high_normal", "true_low"])
        # A fixed reference grouping must not be recomputed from a second model's score.
        changed = make_groups(1-score, ["Explosion", "Normal"], pos, [0, 4, 6], .25)
        self.assertEqual(changed[1], "true_high_abnormal")
        self.assertEqual(groups[1], "missed_anomaly")

    def test_fractional_kernel_usage(self):
        credit = np.array([[.5, .5, 0], [1, 0, 0]])
        stats, counts = winner_summary(credit)
        np.testing.assert_allclose(counts, [1.5, .5, 0])
        self.assertEqual(stats["top1_share"], .75)
        self.assertEqual(stats["used_kernels"], 2)
        self.assertIsNone(winner_summary(np.empty((0, 3)))[0]["top1_share"])
        self.assertIsNone(describe([])["mean"])

    def test_frame_metrics_match_explicit_frames_including_ties(self):
        scores = np.array([.3, .8, .8, .2, .1])
        counts = np.array([7, 12, 3, 0, 16])
        expanded_scores = np.repeat(scores, 16)
        truth = np.concatenate([np.r_[np.ones(n), np.zeros(16-n)] for n in counts])
        positives = expanded_scores[truth == 1]
        negatives = expanded_scores[truth == 0]
        auc = np.mean((positives[:, None] > negatives).astype(float) + .5*(positives[:, None] == negatives))
        ap, recall = 0., 0.
        for threshold in sorted(set(scores), reverse=True):
            chosen = expanded_scores >= threshold
            new_recall = truth[chosen].sum() / truth.sum()
            ap += (new_recall-recall)*truth[chosen].mean()
            recall = new_recall
        metrics, points = frame_metrics(scores, counts, [.01, .05, .5])
        self.assertAlmostEqual(metrics["auc"], auc*100)
        self.assertAlmostEqual(metrics["ap"], ap*100)
        for target, point in points.items():
            selected = expanded_scores >= point["threshold"]
            self.assertAlmostEqual(selected[truth == 0].mean(), point["actual_fpr"])
            self.assertAlmostEqual(selected[truth == 1].mean(), point["recall"])
            self.assertLessEqual(point["actual_fpr"], target)

    def test_single_label_auc_undefined_and_invalid_gt_rejected(self):
        self.assertIsNone(frame_metrics(np.array([.1]), np.array([0]), [.05])[0]["auc"])
        with self.assertRaises(ValueError):
            frame_metrics(np.array([.1]), np.array([17]), [.05])

    def test_config_is_required_to_disambiguate_gelu_identical_state_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pth"
            checkpoint.touch()
            config_path = Path(temp) / "config.json"
            config = dict(cnn_architecture="single_conv", score_mode="competitive_response_gelu")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_config(dict(model_type="cnn", model_path=str(checkpoint),
                                              train_config_json=str(config_path)))["score_mode"], config["score_mode"])
            with self.assertRaises(ValueError):
                load_config(dict(model_type="baseline", model_path=str(checkpoint)))

    def test_end_to_end_outputs_with_mock_inference(self):
        def fake_infer(entry, config, args, gt):
            score = np.array([.1, .2, .6, .9]) if entry["name"] == "final" else np.array([.3, .2, .9, .4])
            logit = np.log(score/(1-score))
            result = dict(score=score, logit=logit, offsets=np.array([0, 2, 4]),
                          ids=["normal", "event"], labels=["Normal", "Explosion"],
                          positive=np.array([0, 0, 16, 0]), bias=None)
            if entry["model_type"] == "cnn":
                result.update(abnormal_max=logit+3, normal_max=np.full(4, 3.), difference=logit,
                              abnormal_credit=np.tile([1., 0.], (4, 1)),
                              normal_credit=np.tile([0., 1.], (4, 1)), bias=0.)
            return result
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "models.json"
            manifest.write_text(json.dumps(dict(reference_model="final", comparison_reference="baseline",
                models=[dict(name="early", model_type="cnn", epoch=1, model_path="early.pth"),
                        dict(name="final", model_type="cnn", epoch=20, model_path="final.pth"),
                        dict(name="baseline", model_type="baseline", model_path="baseline.pth")])), encoding="utf-8")
            np.save(root / "gt.npy", np.r_[np.zeros(32), np.ones(16), np.zeros(16)])
            (root / "test.csv").write_text("test", encoding="utf-8")
            args = build_parser().parse_args([
                "--models-json", str(manifest), "--feature-root", temp, "--test-list", str(root / "test.csv"),
                "--gt-path", str(root / "gt.npy"), "--output-dir", str(root / "out"), "--expected-videos", "2",
            ])
            with patch("torch.cuda.is_available", return_value=True), \
                    patch("ucf_analyze_competitive_kernels.load_config", return_value={}), \
                    patch("ucf_analyze_competitive_kernels.infer", side_effect=fake_infer), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(args)
            for filename in ("metrics_by_model.csv", "kernel_evolution.csv", "response_evolution.csv",
                             "matched_fpr_comparison.csv", "per_class_comparison.csv", "per_video_comparison.csv"):
                self.assertTrue((root / "out" / filename).is_file())
            with (root / "out/early/frame_responses.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[2]["reference_group"], "missed_anomaly")
            self.assertEqual(rows[2]["own_group"], "true_high_abnormal")
            self.assertEqual(rows[2]["frame_start"], "0")


if __name__ == "__main__":
    unittest.main()
