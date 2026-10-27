import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.tools import get_batch_mask
from utils.ucf_detectionMAP import getDetectionMAP as dmAP


def topk_gap(scores, top_k_ratio=0.15):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size < 2:
        return np.nan
    top_k = max(1, int(scores.size * top_k_ratio))
    top_k = min(top_k, scores.size)
    if 2 * top_k > scores.size:
        return np.nan
    sorted_scores = np.sort(scores)
    low_k = sorted_scores[:top_k]
    high_k = sorted_scores[-top_k:]
    return float(np.mean(high_k) - np.mean(low_k))


def summarize_values(values):
    values = np.asarray([value for value in values if not np.isnan(value)], dtype=np.float32)
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def print_score_separation_diagnostics(video_score_records, frame_scores1, frame_scores2, gt, top_k_ratio=0.15):
    gt = np.asarray(gt)
    frame_scores1 = np.asarray(frame_scores1, dtype=np.float32)
    frame_scores2 = np.asarray(frame_scores2, dtype=np.float32)
    valid_length = min(len(gt), len(frame_scores1), len(frame_scores2))
    gt = gt[:valid_length]
    frame_scores1 = frame_scores1[:valid_length]
    frame_scores2 = frame_scores2[:valid_length]

    pos_mask = gt == 1
    neg_mask = gt == 0
    abnormal_records = [record for record in video_score_records if record["label"] != "Normal"]

    gap1 = summarize_values([topk_gap(record["score1"], top_k_ratio) for record in abnormal_records])
    gap2 = summarize_values([topk_gap(record["score2"], top_k_ratio) for record in abnormal_records])

    pos_mean1 = float(np.mean(frame_scores1[pos_mask])) if np.any(pos_mask) else np.nan
    neg_mean1 = float(np.mean(frame_scores1[neg_mask])) if np.any(neg_mask) else np.nan
    pos_mean2 = float(np.mean(frame_scores2[pos_mask])) if np.any(pos_mask) else np.nan
    neg_mean2 = float(np.mean(frame_scores2[neg_mask])) if np.any(neg_mask) else np.nan

    print("Score separation diagnostics:")
    print(
        "  Intra-video abnormal top-k gap "
        f"(top_k_ratio={top_k_ratio}, abnormal_videos={len(abnormal_records)})"
    )
    print(
        "    logits1/classifier: "
        f"mean={gap1['mean']:.4f}, median={gap1['median']:.4f}, "
        f"min={gap1['min']:.4f}, max={gap1['max']:.4f}"
    )
    print(
        "    logits2/alignment:  "
        f"mean={gap2['mean']:.4f}, median={gap2['median']:.4f}, "
        f"min={gap2['min']:.4f}, max={gap2['max']:.4f}"
    )
    print("  Global frame-level separation")
    print(
        "    logits1/classifier: "
        f"positive_mean={pos_mean1:.4f}, negative_mean={neg_mean1:.4f}, "
        f"gap={pos_mean1 - neg_mean1:.4f}"
    )
    print(
        "    logits2/alignment:  "
        f"positive_mean={pos_mean2:.4f}, negative_mean={neg_mean2:.4f}, "
        f"gap={pos_mean2 - neg_mean2:.4f}"
    )


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, prompt_weights=None):
    model.to(device)
    model.eval()

    element_logits2_stack = []
    video_score_records = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            if prompt_weights is not None:
                _, logits1, logits2 = model(
                    visual,
                    padding_mask,
                    prompt_text,
                    lengths,
                    prompt_weights=prompt_weights,
                )
            else:
                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = 1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            video_score_records.append(
                {
                    "label": str(label),
                    "score1": prob1.detach().cpu().numpy(),
                    "score2": prob2.detach().cpu().numpy(),
                }
            )

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy().tolist()
    ap2 = ap2.cpu().numpy().tolist()
    score_len = len(np.repeat(ap1, 16))
    if score_len != len(gt):
        raise ValueError(
            "Test score length does not match ground truth length. "
            f"score_len={score_len}, gt_len={len(gt)}. "
            "Use the full 290-video UCF test list for gt_ucf.npy, not the UCA-filtered description test list."
        )

    roc1 = roc_auc_score(gt, np.repeat(ap1, 16))
    ap_score1 = average_precision_score(gt, np.repeat(ap1, 16))
    roc2 = roc_auc_score(gt, np.repeat(ap2, 16))
    ap_score2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", roc1, " AP1: ", ap_score1)
    print("AUC2: ", roc2, " AP2:", ap_score2)
    print_score_separation_diagnostics(
        video_score_records,
        np.repeat(ap1, 16),
        np.repeat(ap2, 16),
        gt,
    )

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    average_map = 0
    for i in range(5):
        print("mAP@{0:.1f} ={1:.2f}%".format(iou[i], dmap[i]))
        average_map += dmap[i]
    average_map = average_map / (i + 1)
    print("average MAP: {:.2f}".format(average_map))

    return roc1, ap_score1
