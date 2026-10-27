import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class TopKSelectionTracker:
    """Track the exact snippet selections used by VadCLIP's MIL losses."""

    DETAIL_FIELDS = [
        "epoch",
        "sample_id",
        "label",
        "branch",
        "valid_length",
        "k",
        "selected_indices",
        "topk_mean",
        "outside_mean",
        "topk_gap",
        "previous_jaccard",
        "cumulative_coverage_ratio",
        "max_selection_frequency",
    ]

    SUMMARY_FIELDS = [
        "epoch",
        "branch",
        "label",
        "num_samples",
        "mean_previous_jaccard",
        "mean_cumulative_coverage_ratio",
        "mean_max_selection_frequency",
        "mean_topk_gap",
    ]

    def __init__(self, output_dir, branches="both", scope="abnormal"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.branches = branches
        self.scope = scope
        self.previous = {}
        self.frequency = {}
        self.seen = defaultdict(int)
        self.labels = {}
        self.epoch_rows = []

        self.detail_path = self.output_dir / "topk_selection_details.csv"
        self.summary_path = self.output_dir / "topk_epoch_summary.csv"
        self.frequency_path = self.output_dir / "topk_selection_frequency.json"
        self._initialize_csv(self.detail_path, self.DETAIL_FIELDS)
        self._initialize_csv(self.summary_path, self.SUMMARY_FIELDS)

    @staticmethod
    def _initialize_csv(path, fields):
        with path.open("w", encoding="utf-8", newline="") as output_file:
            csv.DictWriter(output_file, fieldnames=fields).writeheader()

    def _should_track(self, label):
        return self.scope == "all" or str(label) != "Normal"

    def _record(self, epoch, sample_id, label, branch, scores, valid_length):
        valid_length = max(1, min(int(valid_length), int(scores.shape[0])))
        k = min(valid_length, max(1, int(valid_length / 16 + 1)))
        valid_scores = scores[:valid_length].float()
        selected = torch.topk(valid_scores, k=k, largest=True).indices.detach().cpu().numpy()
        selected_set = set(int(index) for index in selected.tolist())
        selected_mask = torch.zeros(valid_length, dtype=torch.bool, device=valid_scores.device)
        selected_mask[selected.tolist()] = True

        topk_mean = float(valid_scores[selected_mask].mean().item())
        outside_mean = (
            float(valid_scores[~selected_mask].mean().item())
            if (~selected_mask).any()
            else float("nan")
        )

        key = (str(sample_id), branch)
        previous_set = self.previous.get(key)
        if previous_set is None:
            previous_jaccard = float("nan")
        else:
            union = selected_set | previous_set
            previous_jaccard = len(selected_set & previous_set) / max(1, len(union))
        self.previous[key] = selected_set

        if key not in self.frequency or self.frequency[key].shape[0] != valid_length:
            old_frequency = self.frequency.get(key)
            frequency = np.zeros(valid_length, dtype=np.int32)
            if old_frequency is not None:
                copy_length = min(valid_length, old_frequency.shape[0])
                frequency[:copy_length] = old_frequency[:copy_length]
            self.frequency[key] = frequency
        self.frequency[key][selected] += 1
        self.seen[key] += 1
        self.labels[key] = str(label)

        frequency = self.frequency[key]
        cumulative_coverage = float(np.count_nonzero(frequency) / valid_length)
        max_selection_frequency = float(frequency.max() / max(1, self.seen[key]))
        self.epoch_rows.append(
            {
                "epoch": int(epoch),
                "sample_id": str(sample_id),
                "label": str(label),
                "branch": branch,
                "valid_length": valid_length,
                "k": k,
                "selected_indices": json.dumps(sorted(selected_set)),
                "topk_mean": topk_mean,
                "outside_mean": outside_mean,
                "topk_gap": topk_mean - outside_mean,
                "previous_jaccard": previous_jaccard,
                "cumulative_coverage_ratio": cumulative_coverage,
                "max_selection_frequency": max_selection_frequency,
            }
        )

    def update_from_batch(
        self, epoch, sample_ids, label_names, one_hot_labels, lengths, logits1, logits2
    ):
        classifier_scores = torch.sigmoid(logits1.detach()[..., 0])
        for i, sample_id in enumerate(sample_ids):
            label = str(label_names[i])
            if not self._should_track(label):
                continue
            valid_length = int(lengths[i])
            class_index = int(one_hot_labels[i].argmax().item())
            if self.branches in ("classifier", "both"):
                self._record(
                    epoch, sample_id, label, "classifier", classifier_scores[i], valid_length
                )
            if self.branches in ("alignment", "both"):
                self._record(
                    epoch,
                    sample_id,
                    label,
                    "alignment_true_class_logit",
                    logits2.detach()[i, :, class_index],
                    valid_length,
                )

    @staticmethod
    def _safe_mean(values):
        finite = [float(value) for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    def end_epoch(self, epoch):
        if not self.epoch_rows:
            print("Top-k tracker: no samples were recorded for this epoch.")
            return

        with self.detail_path.open("a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.DETAIL_FIELDS)
            writer.writerows(self.epoch_rows)

        grouped = defaultdict(list)
        for row in self.epoch_rows:
            grouped[(row["branch"], "ALL")].append(row)
            grouped[(row["branch"], row["label"])].append(row)

        summaries = []
        for (branch, label), rows in sorted(grouped.items()):
            summaries.append(
                {
                    "epoch": int(epoch),
                    "branch": branch,
                    "label": label,
                    "num_samples": len(rows),
                    "mean_previous_jaccard": self._safe_mean(
                        row["previous_jaccard"] for row in rows
                    ),
                    "mean_cumulative_coverage_ratio": self._safe_mean(
                        row["cumulative_coverage_ratio"] for row in rows
                    ),
                    "mean_max_selection_frequency": self._safe_mean(
                        row["max_selection_frequency"] for row in rows
                    ),
                    "mean_topk_gap": self._safe_mean(row["topk_gap"] for row in rows),
                }
            )

        with self.summary_path.open("a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.SUMMARY_FIELDS)
            writer.writerows(summaries)

        frequency_records = []
        for key in sorted(self.frequency):
            sample_id, branch = key
            frequency_records.append(
                {
                    "sample_id": sample_id,
                    "label": self.labels[key],
                    "branch": branch,
                    "seen_epochs": self.seen[key],
                    "selection_counts": self.frequency[key].tolist(),
                }
            )
        self.frequency_path.write_text(
            json.dumps(frequency_records, ensure_ascii=False), encoding="utf-8"
        )

        print("Top-k selection tracking:")
        for row in summaries:
            if row["label"] != "ALL":
                continue
            print(
                f"  {row['branch']}: samples={row['num_samples']} | "
                f"previous_jaccard={row['mean_previous_jaccard']:.4f} | "
                f"cumulative_coverage={row['mean_cumulative_coverage_ratio']:.4f} | "
                f"max_frequency={row['mean_max_selection_frequency']:.4f} | "
                f"topk_gap={row['mean_topk_gap']:.4f}"
            )
        print("  details:", self.detail_path)
        print("  summary:", self.summary_path)
        print("  frequency:", self.frequency_path)
        self.epoch_rows = []
