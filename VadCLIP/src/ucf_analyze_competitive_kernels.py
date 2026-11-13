"""Read-only diagnostics of competitive kernels and matched-frame score changes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


GROUPS = ("true_high_abnormal", "false_high_abnormal", "false_high_normal",
          "missed_anomaly", "true_low")
COMPETITIVE = ("competitive_response", "competitive_response_gelu")


def write_csv(path, rows):
    rows = list(rows)
    if rows:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def describe(values):
    values = np.asarray(values)
    if not values.size:
        return dict(count=0, mean=None, std=None, p10=None, p50=None, p90=None)
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                **{f"p{p}": float(np.percentile(values, p)) for p in (10, 50, 90)})


def decompose(raw, mode, bias):
    """Apply the exact readout; share winner credit across exact ties, like amax."""
    if mode not in COMPETITIVE or raw.shape[-1] % 2:
        raise ValueError("Expected competitive mode with an even number of channels")
    active = F.gelu(raw) if mode.endswith("_gelu") else raw
    result = {}
    for side, response, original in zip(("abnormal", "normal"), active.chunk(2, -1), raw.chunk(2, -1)):
        peak = response.amax(-1)
        ties = response == peak.unsqueeze(-1)
        result[side + "_max"] = peak
        result[side + "_raw_max"] = original.amax(-1)
        result[side + "_winner"] = response.argmax(-1)
        result[side + "_tie_count"] = ties.sum(-1)
        result[side + "_credit"] = ties.float() / ties.sum(-1, keepdim=True)
    result["difference"] = result["abnormal_max"] - result["normal_max"]
    result["logit"] = result["difference"] + bias
    return result


def make_groups(scores, labels, positive_counts, offsets, ratio):
    groups = np.full(len(scores), "true_low", dtype="<U24")
    groups[positive_counts > 0] = "missed_anomaly"
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        k = max(1, int(np.ceil((end - start) * ratio)))
        top = start + np.argsort(-scores[start:end], kind="stable")[:k]
        if labels[i] == "Normal":
            groups[top] = "false_high_normal"
        else:
            groups[top] = np.where(positive_counts[top] > 0, "true_high_abnormal", "false_high_abnormal")
    return groups


def winner_summary(credit):
    counts = credit.sum(0)
    n = len(credit)
    probability = counts / max(n, 1)
    nonzero = probability[probability > 0]
    return dict(frames=n, used_kernels=int((counts > 0).sum()),
                top1_share=float(probability.max()) if n else None,
                top5_share=float(np.sort(probability)[-5:].sum()) if n else None,
                effective_kernels=float(np.exp(-(nonzero * np.log(nonzero)).sum())) if n else None), counts


def frame_metrics(scores, positive_counts, fprs):
    # Each feature score occurs 16 times; weights preserve mixed-GT frame counts exactly.
    scores, positive_counts = np.asarray(scores), np.asarray(positive_counts)
    if scores.ndim != 1 or scores.shape != positive_counts.shape or not np.isfinite(scores).all():
        raise ValueError("Expected matching finite score and positive-frame arrays")
    if np.any((positive_counts < 0) | (positive_counts > 16)):
        raise ValueError("Positive frame counts must be between 0 and 16")
    if positive_counts.sum() == 0 or (16 - positive_counts).sum() == 0:
        return dict(auc=None, ap=None), {}
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp_counts = np.cumsum(positive_counts[order], dtype=np.float64)[ends]
    fp_counts = np.cumsum(16 - positive_counts[order], dtype=np.float64)[ends]
    tp = np.r_[0., tp_counts / tp_counts[-1]]
    fp = np.r_[0., fp_counts / fp_counts[-1]]
    thresholds = np.r_[np.inf, sorted_scores[ends]]
    metrics = dict(auc=float(np.sum(np.diff(fp) * (tp[1:] + tp[:-1]) / 2) * 100),
                   ap=float(np.sum(np.diff(tp) * tp_counts / (tp_counts + fp_counts)) * 100))
    points = {}
    for target in fprs:
        index = np.flatnonzero(fp <= target)[-1]
        points[target] = dict(threshold=float(thresholds[index]), actual_fpr=float(fp[index]),
                              recall=float(tp[index]))
    return metrics, points


def load_config(entry):
    path = Path(entry["model_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if entry["model_type"] == "baseline":
        if entry.get("baseline_source") != "self_trained":
            raise ValueError("Baseline must be explicitly identified as self_trained")
        return dict(visual_length=entry.get("visual_length", 256))
    if entry["model_type"] != "cnn":
        raise ValueError("model_type must be baseline or cnn")
    config = json.loads(Path(entry["train_config_json"]).read_text(encoding="utf-8"))
    if config.get("cnn_architecture") != "single_conv":
        raise ValueError("This analyzer expects single_conv CNN checkpoints")
    if config.get("score_mode") not in (*COMPETITIVE, "scoring_head", "kernel_response"):
        raise ValueError("Unsupported score mode in train config")
    return config


@torch.inference_mode()
def infer(entry, config, args, gt):
    from model import CLIPVAD
    from model_cnn_vadclip import CLIPVADCNN
    from ucf_cnn_common import TEST_LABEL_MAP, UCFCNNFeatureDataset
    from utils.tools import get_batch_mask, get_prompt_text
    device = torch.device("cuda")
    length = config.get("visual_length", 256)
    common = (14, 512, length, 512, 1, 2, 8, 10, 10, device)
    mode = config.get("score_mode")
    if entry["model_type"] == "baseline":
        model = CLIPVAD(*common)
    else:
        model = CLIPVADCNN(*common, cnn_architecture="single_conv", score_mode=mode,
                           kernel_size=config["kernel_size"], hidden_width=config["cnn_channels"])
    model.load_state_dict(torch.load(entry["model_path"], map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    dataset = UCFCNNFeatureDataset(args.feature_root, args.test_list, length)
    prompt = get_prompt_text(TEST_LABEL_MAP)
    data = {"score": [], "logit": []}
    offsets, labels, ids, positive = [0], [], [], []
    cursor = 0
    buffer = {}
    hook = None
    if mode in COMPETITIVE:
        hook = model.cnn_classifier.conv.register_forward_hook(
            lambda module, inputs, output: buffer.update(raw=output.transpose(1, 2)))
    try:
        for video_index in range(len(dataset)):
            chunks, label, n, video_id, lengths = dataset[video_index]
            if video_id in ids:
                raise ValueError(f"Duplicate test video_id: {video_id}")
            video_gt = gt[cursor:cursor + n * 16]
            if len(video_gt) != n * 16:
                raise ValueError(f"GT length mismatch for {video_id}")
            counts = video_gt.reshape(n, 16).sum(1).astype(np.int16)
            if label == "Normal" and counts.any():
                raise ValueError(f"Normal video has positive GT; check test ordering: {video_id}")
            positive.append(counts)
            cursor += n * 16
            for start in range(0, len(chunks), args.chunk_batch_size):
                x = chunks[start:start + args.chunk_batch_size].to(device)
                lens = lengths[start:start + args.chunk_batch_size].to(device)
                mask = get_batch_mask(lens, length).to(device)
                if entry["model_type"] == "cnn":
                    _, logits, _ = model(x, mask, prompt, lens, compute_alignment=False)
                else:
                    _, logits, _ = model(x, mask, prompt, lens)
                values = {"logit": logits.squeeze(-1), "score": logits.squeeze(-1).sigmoid()}
                if mode in COMPETITIVE:
                    parts = decompose(buffer.pop("raw"), mode, model.cnn_classifier.bias)
                    torch.testing.assert_close(parts["logit"].masked_fill(mask, 0), logits.squeeze(-1))
                    values.update({key: value for key, value in parts.items() if key != "logit"})
                valid = torch.arange(length, device=device)[None, :] < lens[:, None]
                for key, value in values.items():
                    array = value[valid].cpu().numpy()
                    if not np.isfinite(array).all():
                        raise ValueError(f"Nonfinite {key} in {video_id}")
                    data.setdefault(key, []).append(array)
            offsets.append(offsets[-1] + n)
            ids.append(video_id)
            labels.append(label)
            if (video_index + 1) % 25 == 0:
                print(f"  {entry['name']}: {video_index + 1}/{len(dataset)} videos", flush=True)
    finally:
        if hook is not None:
            hook.remove()
    if cursor != len(gt) or len(dataset) != args.expected_videos:
        raise ValueError(f"Expected {args.expected_videos} videos/{len(gt)} frames, read {len(dataset)}/{cursor}")
    data = {key: np.concatenate(value) for key, value in data.items()}
    data.update(offsets=np.asarray(offsets), ids=ids, labels=labels, positive=np.concatenate(positive))
    data["bias"] = float(model.cnn_classifier.bias.cpu()) if mode in COMPETITIVE else None
    del model
    torch.cuda.empty_cache()
    return data


def summarize_kernels(data, own_groups, fixed_groups, output):
    summaries, usage, concentration, overlap = [], [], [], []
    columns = ["score", "abnormal_max", "normal_max", "difference", "logit",
               "abnormal_raw_max", "normal_raw_max", "abnormal_tie_count", "normal_tie_count"]
    for scope, groups in (("own", own_groups), ("fixed_reference", fixed_groups)):
        for group in ("all", *GROUPS):
            select = np.ones(len(groups), dtype=bool) if group == "all" else groups == group
            for key in columns:
                if key in data:
                    summaries.append(dict(scope=scope, group=group, quantity=key, **describe(data[key][select])))
            for side in ("abnormal", "normal"):
                if side + "_credit" not in data:
                    continue
                stats, counts = winner_summary(data[side + "_credit"][select])
                concentration.append(dict(scope=scope, group=group, side=side, **stats))
                for kernel, count in enumerate(counts):
                    usage.append(dict(scope=scope, group=group, side=side, kernel=kernel,
                                      winner_credit=float(count), share=float(count / max(select.sum(), 1))))
        for side in ("abnormal", "normal"):
            if side + "_credit" not in data:
                continue
            a = data[side + "_credit"][groups == "true_high_abnormal"].sum(0)
            b = data[side + "_credit"][groups == "false_high_abnormal"].sum(0)
            value = float(np.minimum(a / a.sum(), b / b.sum()).sum()) if a.sum() and b.sum() else None
            overlap.append(dict(scope=scope, side=side, true_false_winner_histogram_overlap=value))
    write_csv(output / "response_summary.csv", summaries)
    write_csv(output / "kernel_usage.csv", usage)
    write_csv(output / "kernel_concentration.csv", concentration)
    write_csv(output / "true_false_kernel_overlap.csv", overlap)
    print("  Fixed-reference groups: quantity | group | mean | p10 | p50 | p90", flush=True)
    for row in summaries:
        if row["scope"] == "fixed_reference" and row["quantity"] in ("score", "abnormal_max", "normal_max", "difference"):
            print("  ", row["quantity"], row["group"], *[row[k] for k in ("mean", "p10", "p50", "p90")], flush=True)
    print("  Kernel concentration (fixed-reference groups):", flush=True)
    for row in concentration:
        if row["scope"] == "fixed_reference":
            print("  ", row, flush=True)
    print("  True/false winner overlap:", overlap, flush=True)
    return summaries, concentration


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("models-json", "feature-root", "test-list", "gt-path", "output-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--top-k-ratio", type=float, default=.15)
    parser.add_argument("--chunk-batch-size", type=int, default=16)
    parser.add_argument("--expected-videos", type=int, default=290)
    parser.add_argument("--fpr-targets", type=float, nargs="+", default=[.01, .05])
    return parser


def main(args):
    if not 0 < args.top_k_ratio < 1 or args.chunk_batch_size < 1 or any(not 0 < x < 1 for x in args.fpr_targets):
        raise ValueError("Invalid ratio, chunk batch size or FPR target")
    if not torch.cuda.is_available():
        raise RuntimeError("Choose a GPU runtime; VadCLIP distance graph requires CUDA")
    manifest = json.loads(Path(args.models_json).read_text(encoding="utf-8"))
    entries = manifest["models"]
    names = [e["name"] for e in entries]
    if not names or len(set(names)) != len(names) or any(not n or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in n) for n in names):
        raise ValueError("Model names must be unique alphanumeric/underscore/hyphen identifiers")
    reference = manifest["reference_model"]
    comparison = manifest.get("comparison_reference", reference)
    if reference not in names or comparison not in names:
        raise ValueError("Reference models must be listed")
    configs = {e["name"]: load_config(e) for e in entries}
    entries = sorted(entries, key=lambda e: (e["name"] != reference, e["name"] != comparison))
    gt = np.load(args.gt_path)
    if gt.ndim != 1 or not np.isin(gt, [0, 1]).all():
        raise ValueError("Expected binary, flat frame ground truth")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory to avoid mixing diagnostic runs")
    metadata = dict(manifest=manifest, train_configs=configs, arguments=vars(args), torch=str(torch.__version__),
                    test_list_sha256=hashlib.sha256(Path(args.test_list).read_bytes()).hexdigest(),
                    gt_sha256=hashlib.sha256(Path(args.gt_path).read_bytes()).hexdigest(),
                    notes=["Ground truth is used only for diagnostics, never training.",
                           "FPR thresholds are exploratory test-derived thresholds, not deployment calibration.",
                           "Feature position t corresponds to original frames [16*t,16*(t+1)).",
                           "Groups use any-positive-of-16 GT; frame AUC/AP use all 16 labels exactly.",
                           "Winner credit is shared across exact ties; kernel IDs are local to each group.",
                           "Top-k groups are rank-based, not a fixed prediction threshold.",
                           "Kernel identities across independently trained runs are not aligned."])
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reference_data, compare_data = None, None
    score_rows, evolution, paired, video_rows, response_evolution, class_rows = [], [], [], [], [], []
    for entry in entries:
        name = entry["name"]
        print(f"\nAnalyzing {name}: {entry['model_path']}", flush=True)
        data = infer(entry, configs[name], args, gt)
        if reference_data is None:
            reference_data = {k: data[k] for k in ("score", "positive", "ids", "labels", "offsets")}
            fixed = make_groups(data["score"], data["labels"], data["positive"], data["offsets"], args.top_k_ratio)
        for key in ("positive", "ids", "labels", "offsets"):
            if not np.array_equal(data[key], reference_data[key]):
                raise ValueError(f"Test ordering/GT mismatch for {name}: {key}")
        own = make_groups(data["score"], data["labels"], data["positive"], data["offsets"], args.top_k_ratio)
        folder = output / name
        folder.mkdir()
        summaries, concentrations = summarize_kernels(data, own, fixed, folder)
        response_evolution.extend(dict(model=name, epoch=entry.get("epoch"), **row) for row in summaries)
        frame_labels = np.repeat(data["labels"], np.diff(data["offsets"]))
        class_response_rows = []
        for label in sorted(set(data["labels"])):
            for group in GROUPS:
                select = (frame_labels == label) & (fixed == group)
                for quantity in ("score", "abnormal_max", "normal_max", "difference"):
                    if quantity in data:
                        class_response_rows.append(dict(label=label, reference_group=group,
                                                        quantity=quantity, **describe(data[quantity][select])))
        write_csv(folder / "per_class_responses.csv", class_response_rows)
        metrics, operating = frame_metrics(data["score"], data["positive"], args.fpr_targets)
        score_rows.append(dict(model=name, epoch=entry.get("epoch"), **metrics))
        for row in concentrations:
            evolution.append(dict(model=name, epoch=entry.get("epoch"), **row))
        (folder / "metrics.json").write_text(json.dumps(dict(**metrics, bias=data["bias"]), indent=2), encoding="utf-8")
        np.savez_compressed(folder / "scores.npz", score=data["score"], positive_frames=data["positive"],
                            video_ids=np.asarray(data["ids"]), offsets=data["offsets"])
        def frame_rows():
            for i, (start, end) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
                for j in range(start, end):
                    row = dict(video_id=data["ids"][i], label=data["labels"][i], feature_index=int(j-start),
                               frame_start=int(16*(j-start)), frame_end_exclusive=int(16*(j-start+1)),
                               positive_frame_fraction=float(data["positive"][j]/16),
                               own_group=own[j], reference_group=fixed[j],
                               reference_score=float(reference_data["score"][j]))
                    for key, value in data.items():
                        if key in ("score", "logit", "difference") or key.endswith(("_max", "_winner", "_tie_count")):
                            row[key] = float(value[j])
                    yield row
        write_csv(folder / "frame_responses.csv", frame_rows())
        if name == comparison:
            compare_data = {"score": data["score"], "operating": operating}
        # Inference reference may precede comparison reference; compare saved scores after the loop.
        print("  Frame metrics:", score_rows[-1], flush=True)
        del data
    for entry in entries:
        name = entry["name"]
        with np.load(output / name / "scores.npz") as saved:
            score = saved["score"]
        pos = reference_data["positive"]
        _, points = frame_metrics(score, pos, args.fpr_targets)
        for target, point in points.items():
            base_point = compare_data["operating"][target]
            predicted = score >= point["threshold"]
            base_predicted = compare_data["score"] >= base_point["threshold"]
            paired.append(dict(model=name, comparison_reference=comparison, target_fpr=target,
                               actual_fpr=point["actual_fpr"], recall=point["recall"],
                               threshold=point["threshold"] if np.isfinite(point["threshold"]) else None,
                               baseline_actual_fpr=base_point["actual_fpr"], baseline_recall=base_point["recall"],
                               recovered_anomaly_frames=int(pos[predicted & ~base_predicted].sum()),
                               lost_anomaly_frames=int(pos[~predicted & base_predicted].sum()),
                               removed_false_positive_frames=int((16-pos)[~predicted & base_predicted].sum()),
                               added_false_positive_frames=int((16-pos)[predicted & ~base_predicted].sum())))
        def comparison_frame_rows():
            for i, (start, end) in enumerate(zip(reference_data["offsets"][:-1], reference_data["offsets"][1:])):
                for j in range(start, end):
                    row = dict(video_id=reference_data["ids"][i], label=reference_data["labels"][i],
                               feature_index=int(j-start), frame_start=int(16*(j-start)),
                               frame_end_exclusive=int(16*(j-start+1)), positive_frame_fraction=float(pos[j]/16),
                               score=float(score[j]), comparison_reference=comparison,
                               reference_score=float(compare_data["score"][j]),
                               score_delta=float(score[j]-compare_data["score"][j]))
                    for target, point in points.items():
                        suffix = f"fpr_{target:g}"
                        row["predicted_" + suffix] = int(score[j] >= point["threshold"])
                        row["reference_predicted_" + suffix] = int(
                            compare_data["score"][j] >= compare_data["operating"][target]["threshold"])
                    yield row
        write_csv(output / name / "matched_frame_scores.csv", comparison_frame_rows())
        for i, (start, end) in enumerate(zip(reference_data["offsets"][:-1], reference_data["offsets"][1:])):
            p = pos[start:end]
            current, _ = frame_metrics(score[start:end], p, [])
            base, _ = frame_metrics(compare_data["score"][start:end], p, [])
            delta = score[start:end] - compare_data["score"][start:end]
            video_rows.append(dict(model=name, video_id=reference_data["ids"][i], label=reference_data["labels"][i],
                                   auc=current["auc"], reference_auc=base["auc"],
                                   auc_delta=current["auc"]-base["auc"] if current["auc"] is not None else None,
                                   positive_score_delta=float(np.average(delta, weights=p)) if p.sum() else None,
                                   negative_score_delta=float(np.average(delta, weights=16-p)) if (16-p).sum() else None))
        for label in sorted(set(reference_data["labels"])):
            select = np.repeat(np.asarray(reference_data["labels"]) == label, np.diff(reference_data["offsets"]))
            current, _ = frame_metrics(score[select], pos[select], [])
            base, _ = frame_metrics(compare_data["score"][select], pos[select], [])
            class_rows.append(dict(model=name, label=label, auc=current["auc"], ap=current["ap"],
                                   reference_auc=base["auc"], reference_ap=base["ap"],
                                   auc_delta=current["auc"]-base["auc"] if current["auc"] is not None else None))
    write_csv(output / "metrics_by_model.csv", score_rows)
    write_csv(output / "kernel_evolution.csv", evolution)
    write_csv(output / "response_evolution.csv", response_evolution)
    write_csv(output / "matched_fpr_comparison.csv", paired)
    write_csv(output / "per_video_comparison.csv", video_rows)
    write_csv(output / "per_class_comparison.csv", class_rows)
    print("\nAUC/AP by checkpoint:", *score_rows, sep="\n", flush=True)
    print("\nMatched-FPR comparisons (test diagnostics only):", *paired, sep="\n", flush=True)
    print("\nPer-class comparisons (Normal-only AUC is undefined):", *class_rows, sep="\n", flush=True)
    for name in names:
        rows = [row for row in video_rows if row["model"] == name and row["auc_delta"] is not None]
        print(f"\n{name}: largest per-video AUC drops against {comparison}", flush=True)
        for row in sorted(rows, key=lambda r: r["auc_delta"])[:5]:
            print(row, flush=True)
    print("Saved diagnostics:", output, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
