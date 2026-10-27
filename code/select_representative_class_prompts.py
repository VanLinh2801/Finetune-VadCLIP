import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "VadCLIP" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip import clip  # noqa: E402
from model import CLIPVAD  # noqa: E402
from utils.prompt_replacement import _format_prompt  # noqa: E402


PROMPT_LABEL_MAP = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "roadAccidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}

HARD_NEGATIVE_CLASS_MAP = {
    "Abuse": ["Assault", "Fighting"],
    "Arrest": ["Fighting", "Assault"],
    "Arson": ["Explosion", "Vandalism"],
    "Assault": ["Fighting", "Abuse"],
    "Burglary": ["Robbery", "Stealing", "Shoplifting", "Vandalism"],
    "Explosion": ["Arson", "RoadAccidents"],
    "Fighting": ["Assault", "Abuse", "Arrest"],
    "Normal": [
        "Abuse",
        "Arrest",
        "Arson",
        "Assault",
        "Burglary",
        "Explosion",
        "Fighting",
        "RoadAccidents",
        "Robbery",
        "Shooting",
        "Shoplifting",
        "Stealing",
        "Vandalism",
    ],
    "RoadAccidents": ["Explosion"],
    "Robbery": ["Stealing", "Burglary", "Shoplifting"],
    "Shooting": ["Robbery", "Assault", "Fighting"],
    "Shoplifting": ["Stealing", "Robbery", "Burglary"],
    "Stealing": ["Shoplifting", "Robbery", "Burglary"],
    "Vandalism": ["Arson", "Burglary"],
}

# Quality score uses z-scored components within each class:
#   quality = w_a*z(anchor) + w_m*z(margin) + w_c*z(centroid) - w_h*z(hard_neg) - w_e*z(inter)
# Pairwise intra is diagnostic-only unless legacy preset enables it.
SCORE_PRESETS = {
    "balanced": {
        "anchor": 0.30,
        "anchor_margin": 0.35,
        "centroid": 0.05,
        "intra": 0.0,
        "inter": 0.0,
        "hard_negative": 0.30,
    },
    "conservative": {
        "anchor": 0.40,
        "anchor_margin": 0.30,
        "centroid": 0.10,
        "intra": 0.0,
        "inter": 0.0,
        "hard_negative": 0.20,
    },
    "separation_focused": {
        "anchor": 0.25,
        "anchor_margin": 0.35,
        "centroid": 0.05,
        "intra": 0.0,
        "inter": 0.0,
        "hard_negative": 0.40,
    },
    "legacy_auc8715": {
        "anchor": 0.45,
        "anchor_margin": 0.35,
        "centroid": 0.0,
        "intra": 0.15,
        "inter": 0.05,
        "hard_negative": 0.0,
    },
}

SELECTION_METHOD = "zscore_anchor_margin_centroid_hardneg_quality_v2"
WEIGHTING_METHOD = "softmax_over_diversity_topk_quality_score"


MODEL_CONFIG = dict(
    num_class=14,
    embed_dim=512,
    visual_length=256,
    visual_width=512,
    visual_head=1,
    visual_layers=2,
    attn_window=8,
    prompt_prefix=10,
    prompt_postfix=10,
)


def discover_prompt_jsons():
    prompt_files = []
    for pattern in (
        "ucf_class_prompts_short_nounverb_v1.json",
        "ucf_class_prompts_short_merged.json",
        "ucf_manual_class_prompts_5x*.json",
        "ucf_manual_class_prompts_10x*.json",
        "ucf_keyword_class_prompts_5x.json",
    ):
        prompt_files.extend(PROJECT_ROOT.joinpath("code").glob(pattern))
    prompt_files = [
        path for path in prompt_files
        if "selected" not in path.stem
        and "representative" not in path.stem
        and "weighted" not in path.stem
    ]
    return sorted({path.resolve() for path in prompt_files})


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score UCF-Crime prompt banks in CLIP/VadCLIP text space with a "
            "z-scored quality formula, then build Top-1 and diversity Top-K "
            "weighted prompt banks for training."
        )
    )
    parser.add_argument(
        "--prompt-jsons",
        nargs="+",
        default=None,
        help=(
            "Prompt-bank JSON files. If omitted, short noun-verb / short-merged / "
            "manual 5x-10x / keyword 5x banks are used."
        ),
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "model_ucf.pth"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "code" / "vadclip_representative_prompt_selection"))
    parser.add_argument(
        "--selected-output-json",
        default=str(PROJECT_ROOT / "code" / "ucf_selected_representative_class_prompts.json"),
        help="Top-1 representative prompt bank for diagnostics/ablation, produced from --main-space.",
    )
    parser.add_argument(
        "--weighted-output-json",
        default=str(PROJECT_ROOT / "code" / "ucf_weighted_class_prompts.json"),
        help=(
            "Main diversity Top-K weighted prompt bank for training "
            "(from --main-space)."
        ),
    )
    parser.add_argument(
        "--prompt-weight-temperature",
        default=0.20,
        type=float,
        help="Softmax temperature used to convert Top-K quality scores into prompt weights.",
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--w-anchor", default=0.30, type=float)
    parser.add_argument("--w-anchor-margin", default=0.35, type=float)
    parser.add_argument(
        "--w-centroid",
        default=0.05,
        type=float,
        help="Weight for similarity to the same-class embedding centroid.",
    )
    parser.add_argument(
        "--w-intra",
        default=0.0,
        type=float,
        help="Legacy pairwise intra-class weight. Prefer --w-centroid for large pools.",
    )
    parser.add_argument("--w-inter", default=0.0, type=float)
    parser.add_argument(
        "--w-hard-negative",
        default=0.30,
        type=float,
        help="Penalty weight for similarity to prompts from confusing hard-negative classes.",
    )
    parser.add_argument(
        "--score-preset",
        default="balanced",
        choices=["custom", "balanced", "conservative", "separation_focused", "legacy_auc8715"],
        help=(
            "Preset quality formula. Non-custom presets override --w-* weights. "
            "Default balanced uses z-scored anchor/margin/centroid/hard-neg."
        ),
    )
    parser.add_argument(
        "--inter-mode",
        default="max",
        choices=["max", "top3_mean", "mean"],
        help="How to aggregate similarity to prompts from other / hard-negative classes.",
    )
    parser.add_argument(
        "--normalize-components",
        default="zscore",
        choices=["zscore", "none"],
        help="Normalize quality components within each class before applying weights.",
    )
    parser.add_argument(
        "--topk",
        default=5,
        type=int,
        help="Number of diversity-filtered prompts kept per class for the weighted bank.",
    )
    parser.add_argument(
        "--diversity-mode",
        default="mmr",
        choices=["none", "mmr", "threshold"],
        help=(
            "How to pick Top-K after quality ranking: none=top score only, "
            "mmr=MMR with optional hard cosine filter, threshold=greedy cosine filter."
        ),
    )
    parser.add_argument(
        "--diversity-threshold",
        default=0.97,
        type=float,
        help="Cosine similarity threshold used to reject near-duplicate prompts.",
    )
    parser.add_argument(
        "--mmr-lambda",
        default=0.70,
        type=float,
        help="MMR trade-off: lambda * quality - (1-lambda) * max_sim_to_selected.",
    )
    parser.add_argument(
        "--main-space",
        default="raw_clip_encode_text",
        choices=["vadclip_encode_textprompt", "raw_clip_encode_text"],
        help="Embedding space used to write selected/weighted output JSONs.",
    )
    parser.add_argument(
        "--scoring-prompt-mode",
        default="prototype_only",
        choices=["manual_caption", "class_plus_prototype", "prototype_only"],
        help=(
            "How prompt candidates are formatted before scoring. prototype_only measures "
            "the raw prompt against class-name anchors without injecting class names."
        ),
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=["vadclip_encode_textprompt", "raw_clip_encode_text"],
        choices=["vadclip_encode_textprompt", "raw_clip_encode_text"],
    )
    return parser.parse_args()


def resolve_score_weights(args):
    if args.score_preset != "custom":
        return dict(SCORE_PRESETS[args.score_preset])
    return {
        "anchor": args.w_anchor,
        "anchor_margin": args.w_anchor_margin,
        "centroid": args.w_centroid,
        "intra": args.w_intra,
        "inter": args.w_inter,
        "hard_negative": args.w_hard_negative,
    }


def camel_to_words(text):
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.replace("_", " ").strip().lower()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("Saved CSV:", path)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved JSON:", path)


def normalize_np(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def instantiate_model(model_path, device):
    config = dict(MODEL_CONFIG)
    config["device"] = device
    model = CLIPVAD(**config)
    state_dict = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Loaded VadCLIP text encoder checkpoint:", model_path)
    print("  missing keys:", len(missing), "| unexpected keys:", len(unexpected))
    return model


def encode_texts_with_vadclip(model, texts, batch_size):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            emb = model.encode_textprompt(batch)
            emb = F.normalize(emb.float(), dim=-1)
            chunks.append(emb.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def encode_texts_with_raw_clip(model, texts, batch_size, device):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            token_embeddings = model.clipmodel.encode_token(tokens)
            emb = model.clipmodel.encode_text(token_embeddings, tokens)
            emb = F.normalize(emb.float(), dim=-1)
            chunks.append(emb.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def encode_texts(model, texts, space_name, batch_size, device):
    if space_name == "vadclip_encode_textprompt":
        return encode_texts_with_vadclip(model, texts, batch_size)
    if space_name == "raw_clip_encode_text":
        return encode_texts_with_raw_clip(model, texts, batch_size, device)
    raise ValueError(f"Unknown embedding space: {space_name}")


def get_clip_token_rows(texts, candidate_rows):
    tokens = clip.tokenize(texts, truncate=True)
    rows = []
    for i, base in enumerate(candidate_rows):
        nonzero_ids = [int(token_id) for token_id in tokens[i].tolist() if int(token_id) != 0]
        row = dict(base)
        row.update(
            {
                "global_prompt_index": i,
                "clip_token_count_with_special": len(nonzero_ids),
                "clip_content_token_count": max(0, len(nonzero_ids) - 2),
                "clip_token_ids_nonzero": " ".join(str(token_id) for token_id in nonzero_ids),
            }
        )
        rows.append(row)
    return rows


def load_prompt_bank(prompt_json, scoring_prompt_mode):
    """Load a candidate prompt bank for scoring.

    Unlike training-time build_multi_prompt_groups, candidate pools may have
    unequal prompts per class (merged banks). Equality is only required later
    for the diversity Top-K training bank.
    """
    prompt_json = Path(prompt_json).resolve()
    payload = json.loads(prompt_json.read_text(encoding="utf-8"))
    raw_classes = payload.get("classes", {})
    class_names = list(PROMPT_LABEL_MAP.keys())

    rows = []
    encoded_prompts = []
    per_class_counts = {}
    for class_index, class_name in enumerate(class_names):
        if class_name not in raw_classes:
            raise KeyError(f"Missing class {class_name} in {prompt_json}")
        raw_group = raw_classes[class_name]
        if not isinstance(raw_group, list):
            raise ValueError(
                f"Manual candidate bank expects a list of prompts for {class_name} in {prompt_json}"
            )
        prompts = [prompt.strip() for prompt in raw_group if isinstance(prompt, str) and prompt.strip()]
        if not prompts:
            raise ValueError(f"No prompts for class {class_name} in {prompt_json}")
        per_class_counts[class_name] = len(prompts)
        for prompt_index, raw_prompt in enumerate(prompts):
            encoded_prompt = _format_prompt(PROMPT_LABEL_MAP[class_name], raw_prompt, scoring_prompt_mode)
            encoded_prompts.append(encoded_prompt)
            rows.append(
                {
                    "source_file": prompt_json.name,
                    "source_stem": prompt_json.stem,
                    "source_prompts_per_class": len(prompts),
                    "class_index": class_index,
                    "class_name": class_name,
                    "class_anchor_text": PROMPT_LABEL_MAP[class_name],
                    "prompt_index": prompt_index,
                    "raw_prompt_text": raw_prompt,
                    "encoded_prompt_text": encoded_prompt,
                }
            )

    print(
        f"Loaded candidate bank {prompt_json.name}: "
        f"{len(encoded_prompts)} prompts | per-class={per_class_counts}"
    )
    return payload, encoded_prompts, rows


def load_candidate_pool(prompt_jsons, scoring_prompt_mode):
    all_payloads = {}
    all_texts = []
    all_rows = []
    for prompt_json in prompt_jsons:
        payload, texts, rows = load_prompt_bank(prompt_json, scoring_prompt_mode)
        all_payloads[Path(prompt_json).name] = payload.get("metadata", {})
        offset = len(all_texts)
        for local_index, row in enumerate(rows):
            row = dict(row)
            row["global_prompt_index"] = offset + local_index
            all_rows.append(row)
        all_texts.extend(texts)
    return all_payloads, all_texts, all_rows


def aggregate_inter_similarity(other_scores, mode):
    if len(other_scores) == 0:
        return 0.0
    if mode == "max":
        return float(other_scores.max())
    if mode == "top3_mean":
        top_k = min(3, len(other_scores))
        return float(np.sort(other_scores)[-top_k:].mean())
    if mode == "mean":
        return float(other_scores.mean())
    raise ValueError(f"Unknown inter mode: {mode}")


def zscore_by_class(values_by_index, class_indices):
    """Return z-scored values; constant-within-class components become 0."""
    zscores = np.zeros(len(values_by_index), dtype=np.float64)
    unique_classes = sorted(set(class_indices))
    for class_index in unique_classes:
        indices = [i for i, c in enumerate(class_indices) if c == class_index]
        vals = np.asarray([values_by_index[i] for i in indices], dtype=np.float64)
        mean = vals.mean()
        std = vals.std()
        if std < 1e-8:
            for i in indices:
                zscores[i] = 0.0
        else:
            for local, i in enumerate(indices):
                zscores[i] = (vals[local] - mean) / std
    return zscores


def minmax_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    vmin = values.min()
    vmax = values.max()
    if vmax - vmin < 1e-8:
        return np.ones_like(values)
    return (values - vmin) / (vmax - vmin)


def rank_candidates(prompt_embeddings, anchor_embeddings, candidate_rows, class_names, weights, inter_mode, normalize_components):
    prompt_similarity = prompt_embeddings @ prompt_embeddings.T
    prompt_to_anchor = prompt_embeddings @ anchor_embeddings.T
    anchor_similarity = anchor_embeddings @ anchor_embeddings.T
    class_index_by_name = {class_name: idx for idx, class_name in enumerate(class_names)}

    indices_by_class = {
        class_index: [i for i, row in enumerate(candidate_rows) if row["class_index"] == class_index]
        for class_index in range(len(class_names))
    }

    # Same-class centroids (L2-normalized means).
    centroid_embeddings = np.zeros_like(anchor_embeddings)
    for class_index, indices in indices_by_class.items():
        if not indices:
            continue
        centroid = prompt_embeddings[indices].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-8)
        centroid_embeddings[class_index] = centroid

    raw_components = {
        "anchor": [],
        "anchor_margin": [],
        "centroid": [],
        "intra": [],
        "inter": [],
        "hard_negative": [],
    }
    scored_rows = []
    for row_index, base in enumerate(candidate_rows):
        class_index = base["class_index"]
        same_indices = [idx for idx in indices_by_class[class_index] if idx != row_index]
        other_indices = [
            idx for other_class, indices in indices_by_class.items()
            if other_class != class_index
            for idx in indices
        ]

        anchor_scores = prompt_to_anchor[row_index]
        best_other_anchor = max(
            [idx for idx in range(len(class_names)) if idx != class_index],
            key=lambda idx: anchor_scores[idx],
        )
        intra_similarity = float(prompt_similarity[row_index, same_indices].mean()) if same_indices else 1.0
        inter_similarity = aggregate_inter_similarity(prompt_similarity[row_index, other_indices], inter_mode)
        hard_negative_class_names = HARD_NEGATIVE_CLASS_MAP.get(base["class_name"], [])
        hard_negative_indices = [
            idx
            for class_name in hard_negative_class_names
            for idx in indices_by_class.get(class_index_by_name[class_name], [])
        ]
        if hard_negative_indices:
            hard_negative_prompt_similarity = aggregate_inter_similarity(
                prompt_similarity[row_index, hard_negative_indices],
                inter_mode,
            )
        else:
            hard_negative_prompt_similarity = inter_similarity
        anchor_similarity_own = float(prompt_to_anchor[row_index, class_index])
        anchor_margin = float(anchor_similarity_own - anchor_scores[best_other_anchor])
        centroid_similarity = float(prompt_embeddings[row_index] @ centroid_embeddings[class_index])

        raw_components["anchor"].append(anchor_similarity_own)
        raw_components["anchor_margin"].append(anchor_margin)
        raw_components["centroid"].append(centroid_similarity)
        raw_components["intra"].append(intra_similarity)
        raw_components["inter"].append(inter_similarity)
        raw_components["hard_negative"].append(hard_negative_prompt_similarity)

        scored = dict(base)
        scored.update(
            {
                "anchor_similarity": anchor_similarity_own,
                "centroid_similarity": centroid_similarity,
                "intra_prompt_similarity": intra_similarity,
                "inter_prompt_similarity": inter_similarity,
                "hard_negative_prompt_similarity": hard_negative_prompt_similarity,
                "hard_negative_classes": ", ".join(hard_negative_class_names),
                "best_other_anchor_class": class_names[best_other_anchor],
                "best_other_anchor_similarity": float(anchor_scores[best_other_anchor]),
                "anchor_margin": anchor_margin,
            }
        )
        scored_rows.append(scored)

    class_indices = [row["class_index"] for row in scored_rows]
    if normalize_components == "zscore":
        z_components = {
            name: zscore_by_class(values, class_indices)
            for name, values in raw_components.items()
        }
    elif normalize_components == "none":
        z_components = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in raw_components.items()
        }
    else:
        raise ValueError(f"Unknown normalize_components: {normalize_components}")

    for row_index, scored in enumerate(scored_rows):
        selection_score = (
            weights.get("anchor", 0.0) * z_components["anchor"][row_index]
            + weights.get("anchor_margin", 0.0) * z_components["anchor_margin"][row_index]
            + weights.get("centroid", 0.0) * z_components["centroid"][row_index]
            + weights.get("intra", 0.0) * z_components["intra"][row_index]
            - weights.get("inter", 0.0) * z_components["inter"][row_index]
            - weights.get("hard_negative", 0.0) * z_components["hard_negative"][row_index]
        )
        scored["z_anchor"] = float(z_components["anchor"][row_index])
        scored["z_anchor_margin"] = float(z_components["anchor_margin"][row_index])
        scored["z_centroid"] = float(z_components["centroid"][row_index])
        scored["z_intra"] = float(z_components["intra"][row_index])
        scored["z_inter"] = float(z_components["inter"][row_index])
        scored["z_hard_negative"] = float(z_components["hard_negative"][row_index])
        scored["selection_score"] = float(selection_score)
        scored["normalize_components"] = normalize_components

    selected_rows = []
    for class_index, class_name in enumerate(class_names):
        class_rows = [row for row in scored_rows if row["class_index"] == class_index]
        class_rows.sort(key=lambda row: row["selection_score"], reverse=True)
        for rank, row in enumerate(class_rows, start=1):
            row["rank_in_class"] = rank
        if class_rows:
            selected_rows.append(dict(class_rows[0]))
    scored_rows.sort(key=lambda row: (row["class_index"], row["rank_in_class"]))

    selected_indices = [row["global_prompt_index"] for row in selected_rows]
    selected_embeddings = prompt_embeddings[selected_indices] if selected_indices else np.zeros((0, prompt_embeddings.shape[1]))
    selected_similarity = selected_embeddings @ selected_embeddings.T if len(selected_indices) else np.zeros((0, 0))
    return scored_rows, selected_rows, prompt_similarity, prompt_to_anchor, anchor_similarity, selected_similarity


def select_diverse_topk(class_rows, prompt_embeddings, topk, diversity_mode, diversity_threshold, mmr_lambda):
    """Greedy diversity selection after quality ranking. Returns selected rows in pick order."""
    if topk <= 0:
        raise ValueError("--topk must be positive")
    candidates = sorted(class_rows, key=lambda row: row["selection_score"], reverse=True)
    if not candidates:
        return []
    if diversity_mode == "none":
        selected = [dict(row) for row in candidates[:topk]]
    else:
        quality_norm = minmax_normalize([row["selection_score"] for row in candidates])
        remaining = list(range(len(candidates)))
        selected_local = []
        while remaining and len(selected_local) < topk:
            if not selected_local:
                chosen = remaining.pop(0)
                selected_local.append(chosen)
                continue

            best_pos = None
            best_value = -1e18
            for pos, cand_idx in enumerate(remaining):
                cand = candidates[cand_idx]
                cand_emb = prompt_embeddings[cand["global_prompt_index"]]
                max_sim = max(
                    float(cand_emb @ prompt_embeddings[candidates[sel_idx]["global_prompt_index"]])
                    for sel_idx in selected_local
                )
                if max_sim >= diversity_threshold:
                    continue
                if diversity_mode == "threshold":
                    best_pos = pos
                    break
                if diversity_mode == "mmr":
                    value = mmr_lambda * float(quality_norm[cand_idx]) - (1.0 - mmr_lambda) * max_sim
                    if value > best_value:
                        best_value = value
                        best_pos = pos
                else:
                    raise ValueError(f"Unknown diversity_mode: {diversity_mode}")

            if best_pos is None:
                break
            selected_local.append(remaining.pop(best_pos))

        # If diversity constraints under-fill Top-K, pad with next-best by quality.
        if len(selected_local) < topk and remaining:
            for cand_idx in list(remaining):
                if len(selected_local) >= topk:
                    break
                selected_local.append(cand_idx)
                remaining.remove(cand_idx)

        selected = [dict(candidates[idx]) for idx in selected_local]

    for rank, row in enumerate(selected, start=1):
        row["diversity_rank_in_class"] = rank
        row["diversity_mode"] = diversity_mode
        row["diversity_threshold"] = float(diversity_threshold)
        row["mmr_lambda"] = float(mmr_lambda)
        row["selected_in_diversity_topk"] = True
    return selected


def build_diversity_topk_rows(scored_rows, prompt_embeddings, class_names, topk, diversity_mode, diversity_threshold, mmr_lambda):
    selected = []
    for class_index, class_name in enumerate(class_names):
        class_rows = [row for row in scored_rows if row["class_index"] == class_index]
        selected.extend(
            select_diverse_topk(
                class_rows,
                prompt_embeddings,
                topk,
                diversity_mode,
                diversity_threshold,
                mmr_lambda,
            )
        )
    selected_ids = {row["global_prompt_index"] for row in selected}
    for row in scored_rows:
        row["selected_in_diversity_topk"] = row["global_prompt_index"] in selected_ids
        if row["global_prompt_index"] not in selected_ids:
            row["diversity_rank_in_class"] = ""
    return selected


def softmax_np(values, temperature):
    values = np.asarray(values, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    scaled = values / temperature
    scaled = scaled - scaled.max()
    exp_values = np.exp(scaled)
    return exp_values / np.maximum(exp_values.sum(), 1e-12)


def add_prompt_weights(scored_rows, temperature, only_diversity_topk=False):
    rows = [dict(row) for row in scored_rows]
    if only_diversity_topk:
        groups = {}
        for row in rows:
            if not row.get("selected_in_diversity_topk"):
                row["prompt_weight"] = 0.0
                row["weight_rank_in_class"] = ""
                row["prompt_weight_temperature"] = float(temperature)
                continue
            groups.setdefault(row["class_index"], []).append(row)
        for group_rows in groups.values():
            group_rows.sort(key=lambda row: row.get("diversity_rank_in_class", row["rank_in_class"]))
            weights = softmax_np([row["selection_score"] for row in group_rows], temperature)
            rank_order = sorted(range(len(group_rows)), key=lambda idx: weights[idx], reverse=True)
            ranks = {idx: rank + 1 for rank, idx in enumerate(rank_order)}
            for idx, row in enumerate(group_rows):
                row["prompt_weight"] = float(weights[idx])
                row["weight_rank_in_class"] = ranks[idx]
                row["prompt_weight_temperature"] = float(temperature)
        return rows

    groups = {}
    for row in rows:
        key = (row["source_file"], row["class_index"])
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda row: row["prompt_index"])
        weights = softmax_np([row["selection_score"] for row in group_rows], temperature)
        rank_order = sorted(range(len(group_rows)), key=lambda idx: weights[idx], reverse=True)
        ranks = {idx: rank + 1 for rank, idx in enumerate(rank_order)}
        for idx, row in enumerate(group_rows):
            row["prompt_weight"] = float(weights[idx])
            row["weight_rank_in_class"] = ranks[idx]
            row["prompt_weight_temperature"] = float(temperature)
    return rows


def build_prompt_to_anchor_rows(scored_rows, prompt_to_anchor, class_names):
    rows = []
    for row_index, base in enumerate(scored_rows):
        row = {
            "global_prompt_index": base["global_prompt_index"],
            "source_file": base["source_file"],
            "source_stem": base["source_stem"],
            "class_index": base["class_index"],
            "class_name": base["class_name"],
            "prompt_index": base["prompt_index"],
            "raw_prompt_text": base["raw_prompt_text"],
            "encoded_prompt_text": base["encoded_prompt_text"],
            "own_class_similarity": base["anchor_similarity"],
            "best_other_anchor_class": base["best_other_anchor_class"],
            "best_other_anchor_similarity": base["best_other_anchor_similarity"],
            "anchor_margin": base["anchor_margin"],
        }
        for class_index, class_name in enumerate(class_names):
            row[f"sim_to_{class_name}"] = float(prompt_to_anchor[row_index, class_index])
        rows.append(row)
    return rows


def build_weighted_payload_for_rows(
    selected_rows,
    prompt_jsons,
    weights,
    score_preset,
    inter_mode,
    space_name,
    scoring_prompt_mode,
    temperature,
    topk,
    diversity_mode,
    diversity_threshold,
    mmr_lambda,
    normalize_components,
    source_label=None,
):
    class_names = list(PROMPT_LABEL_MAP.keys())
    classes = {}
    prompt_weights = {}
    scoring_details = {}
    counts = set()
    for class_name in class_names:
        class_rows = [row for row in selected_rows if row["class_name"] == class_name]
        class_rows.sort(key=lambda row: row.get("diversity_rank_in_class", row["rank_in_class"]))
        if not class_rows:
            raise ValueError(f"No diversity Top-K rows for class: {class_name}")
        classes[class_name] = [row["raw_prompt_text"] for row in class_rows]
        prompt_weights[class_name] = [float(row["prompt_weight"]) for row in class_rows]
        scoring_details[class_name] = class_rows
        counts.add(len(class_rows))

    if len(counts) != 1:
        raise ValueError(f"Diversity Top-K must keep equal prompts per class. Got sizes: {sorted(counts)}")

    return {
        "metadata": {
            "source_label": source_label,
            "source_prompt_jsons_all_inputs": [str(Path(path).resolve()) for path in prompt_jsons],
            "selection_method": SELECTION_METHOD,
            "weighting_method": WEIGHTING_METHOD,
            "embedding_space": space_name,
            "scoring_prompt_mode": scoring_prompt_mode,
            "training_prompt_mode": scoring_prompt_mode,
            "note": (
                "classes contain raw diversity Top-K prompt text; prompt_weights has the same order. "
                "Use the same prompt_mode at training/evaluation unless intentionally testing a mismatch."
            ),
            "score_weights": weights,
            "score_preset": score_preset,
            "inter_mode": inter_mode,
            "normalize_components": normalize_components,
            "hard_negative_class_map": HARD_NEGATIVE_CLASS_MAP,
            "prompt_weight_temperature": float(temperature),
            "topk": int(topk),
            "diversity_mode": diversity_mode,
            "diversity_threshold": float(diversity_threshold),
            "mmr_lambda": float(mmr_lambda),
            "prompts_per_class": len(next(iter(classes.values()))),
        },
        "classes": classes,
        "prompt_weights": prompt_weights,
        "scoring_details": scoring_details,
    }


def compare_baseline_and_selected(class_names, anchor_similarity, selected_similarity):
    rows = []
    for i, class_name in enumerate(class_names):
        other_indices = [j for j in range(len(class_names)) if j != i]
        baseline_nearest = max(other_indices, key=lambda j: anchor_similarity[i, j])
        selected_nearest = max(other_indices, key=lambda j: selected_similarity[i, j])
        rows.append(
            {
                "class_name": class_name,
                "baseline_nearest_class": class_names[baseline_nearest],
                "baseline_nearest_similarity": float(anchor_similarity[i, baseline_nearest]),
                "baseline_separation": float(1.0 - anchor_similarity[i, baseline_nearest]),
                "selected_nearest_class": class_names[selected_nearest],
                "selected_nearest_similarity": float(selected_similarity[i, selected_nearest]),
                "selected_separation": float(1.0 - selected_similarity[i, selected_nearest]),
                "separation_delta_selected_minus_baseline": float(
                    (1.0 - selected_similarity[i, selected_nearest])
                    - (1.0 - anchor_similarity[i, baseline_nearest])
                ),
            }
        )
    return sorted(rows, key=lambda row: row["separation_delta_selected_minus_baseline"])


def build_selected_payload(
    prompt_jsons,
    selected_rows,
    weights,
    score_preset,
    inter_mode,
    space_name,
    scoring_prompt_mode,
    normalize_components,
):
    return {
        "metadata": {
            "source_prompt_jsons": [str(Path(path).resolve()) for path in prompt_jsons],
            "selection_method": SELECTION_METHOD,
            "embedding_space": space_name,
            "scoring_prompt_mode": scoring_prompt_mode,
            "training_prompt_mode": scoring_prompt_mode,
            "note": "classes contain raw prompt text only; use the same prompt_mode at training/evaluation unless intentionally testing a mismatch.",
            "weights": weights,
            "score_preset": score_preset,
            "inter_mode": inter_mode,
            "normalize_components": normalize_components,
            "hard_negative_class_map": HARD_NEGATIVE_CLASS_MAP,
            "prompts_per_class": 1,
        },
        "classes": {row["class_name"]: [row["raw_prompt_text"]] for row in selected_rows},
        "selection_details": {row["class_name"]: row for row in selected_rows},
    }


def print_console_summary(space_name, selected_rows, diversity_rows, scored_rows, baseline_rows, topk):
    print()
    print("=" * 110)
    print("Embedding space:", space_name)
    print("Selected Top-1 representative prompts:")
    for row in selected_rows:
        print(
            f"  {row['class_name']:<13} "
            f"{row['source_stem']}#{row['prompt_index']:<2} "
            f"score={row['selection_score']:+.4f} "
            f"anchor={row['anchor_similarity']:.4f} "
            f"centroid={row['centroid_similarity']:.4f} "
            f"hard_neg={row['hard_negative_prompt_similarity']:.4f} "
            f"margin={row['anchor_margin']:+.4f} | {row['raw_prompt_text']}"
        )

    print()
    print(f"Diversity Top-{topk} prompts (first 2 per class):")
    for class_name in PROMPT_LABEL_MAP:
        class_rows = [row for row in diversity_rows if row["class_name"] == class_name]
        class_rows.sort(key=lambda row: row["diversity_rank_in_class"])
        for row in class_rows[:2]:
            print(
                f"  {row['class_name']:<13} "
                f"div#{row['diversity_rank_in_class']} "
                f"w={row.get('prompt_weight', 0):.3f} "
                f"score={row['selection_score']:+.4f} | {row['raw_prompt_text']}"
            )

    wrong_anchor = [row for row in scored_rows if row["anchor_margin"] < 0]
    print()
    print("Prompts whose nearest class-name anchor is not their own:", len(wrong_anchor), "/", len(scored_rows))
    for row in sorted(wrong_anchor, key=lambda item: item["anchor_margin"])[:15]:
        print(
            f"  {row['class_name']} {row['source_stem']}#{row['prompt_index']} "
            f"own={row['anchor_similarity']:.4f} "
            f"best={row['best_other_anchor_class']}:{row['best_other_anchor_similarity']:.4f} "
            f"margin={row['anchor_margin']:+.4f} | {row['raw_prompt_text']}"
        )

    print()
    print("Lowest total-score prompts:")
    for row in sorted(scored_rows, key=lambda item: item["selection_score"])[:15]:
        print(
            f"  {row['class_name']} {row['source_stem']}#{row['prompt_index']} "
            f"score={row['selection_score']:+.4f} "
            f"anchor={row['anchor_similarity']:.4f} "
            f"centroid={row['centroid_similarity']:.4f} "
            f"hard_neg={row['hard_negative_prompt_similarity']:.4f} | {row['raw_prompt_text']}"
        )

    print()
    print("Text-space baseline class-name vs selected prompt separation:")
    for row in baseline_rows:
        print(
            f"  {row['class_name']:<13} "
            f"baseline->{row['baseline_nearest_class']}:{row['baseline_nearest_similarity']:.4f} "
            f"selected->{row['selected_nearest_class']}:{row['selected_nearest_similarity']:.4f} "
            f"delta_sep={row['separation_delta_selected_minus_baseline']:+.4f}"
        )


def analyze_candidate_pool(prompt_jsons, model, args, output_root, weights):
    class_names = list(PROMPT_LABEL_MAP.keys())
    all_payloads, candidate_texts, candidate_rows = load_candidate_pool(prompt_jsons, args.scoring_prompt_mode)
    class_anchor_texts = [PROMPT_LABEL_MAP[class_name] for class_name in class_names]

    print()
    print("#" * 110)
    print("Prompt JSON inputs:")
    for prompt_json in prompt_jsons:
        print("  -", Path(prompt_json).resolve())
    print("Classes:", len(class_names))
    print("Total candidate prompts:", len(candidate_texts))
    print("Class-name anchors:", ", ".join(class_anchor_texts))
    print(
        "Quality formula:",
        SELECTION_METHOD,
        "| normalize:", args.normalize_components,
        "| topk:", args.topk,
        "| diversity:", args.diversity_mode,
        f"(tau={args.diversity_threshold}, mmr_lambda={args.mmr_lambda})",
    )

    prompts_per_class_available = {
        class_name: len([row for row in candidate_rows if row["class_name"] == class_name])
        for class_name in class_names
    }
    min_available = min(prompts_per_class_available.values())
    effective_topk = min(int(args.topk), int(min_available))
    if effective_topk < int(args.topk):
        print(
            f"Warning: reducing --topk from {args.topk} to {effective_topk} "
            f"because some classes have fewer candidates: {prompts_per_class_available}"
        )
    args.topk = effective_topk

    token_rows = get_clip_token_rows(candidate_texts, candidate_rows)
    token_fields = [
        "global_prompt_index",
        "source_file",
        "class_index",
        "class_name",
        "prompt_index",
        "clip_token_count_with_special",
        "clip_content_token_count",
        "raw_prompt_text",
        "encoded_prompt_text",
        "clip_token_ids_nonzero",
    ]
    write_csv(output_root / "all_candidate_clip_token_counts.csv", token_rows, token_fields)
    print()
    print("CLIP token counts for candidate prompts:")
    for row in token_rows:
        print(
            f"  global={row['global_prompt_index']:03d} "
            f"{row['source_stem']} {row['class_name']}#{row['prompt_index']} "
            f"tokens={row['clip_token_count_with_special']} "
            f"content={row['clip_content_token_count']} | {row['encoded_prompt_text']}"
        )

    all_summaries = []
    main_payload = None
    main_weighted_payload = None
    for space_name in args.spaces:
        space_dir = output_root / space_name
        space_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings = normalize_np(encode_texts(model, candidate_texts, space_name, args.batch_size, args.device))
        anchor_embeddings = normalize_np(encode_texts(model, class_anchor_texts, space_name, args.batch_size, args.device))

        scored_rows, selected_rows, prompt_similarity, prompt_to_anchor, anchor_similarity, selected_similarity = rank_candidates(
            prompt_embeddings,
            anchor_embeddings,
            candidate_rows,
            class_names,
            weights,
            args.inter_mode,
            args.normalize_components,
        )
        diversity_rows = build_diversity_topk_rows(
            scored_rows,
            prompt_embeddings,
            class_names,
            args.topk,
            args.diversity_mode,
            args.diversity_threshold,
            args.mmr_lambda,
        )
        # Softmax weights over diversity Top-K only (trainable bank).
        weighted_rows = add_prompt_weights(scored_rows, args.prompt_weight_temperature, only_diversity_topk=True)
        weighted_lookup = {
            row["global_prompt_index"]: {
                "prompt_weight": row["prompt_weight"],
                "weight_rank_in_class": row["weight_rank_in_class"],
                "prompt_weight_temperature": row["prompt_weight_temperature"],
                "selected_in_diversity_topk": row.get("selected_in_diversity_topk", False),
                "diversity_rank_in_class": row.get("diversity_rank_in_class", ""),
            }
            for row in weighted_rows
        }
        for row in scored_rows:
            row.update(weighted_lookup[row["global_prompt_index"]])
        for row in selected_rows:
            row.update(weighted_lookup[row["global_prompt_index"]])
        for row in diversity_rows:
            row.update(weighted_lookup[row["global_prompt_index"]])

        baseline_rows = compare_baseline_and_selected(class_names, anchor_similarity, selected_similarity)
        prompt_to_anchor_rows = build_prompt_to_anchor_rows(scored_rows, prompt_to_anchor, class_names)

        score_fields = [
            "class_index",
            "class_name",
            "rank_in_class",
            "diversity_rank_in_class",
            "selected_in_diversity_topk",
            "weight_rank_in_class",
            "source_file",
            "source_stem",
            "source_prompts_per_class",
            "prompt_index",
            "selection_score",
            "prompt_weight",
            "anchor_similarity",
            "centroid_similarity",
            "intra_prompt_similarity",
            "inter_prompt_similarity",
            "hard_negative_prompt_similarity",
            "hard_negative_classes",
            "anchor_margin",
            "z_anchor",
            "z_anchor_margin",
            "z_centroid",
            "z_intra",
            "z_inter",
            "z_hard_negative",
            "best_other_anchor_class",
            "best_other_anchor_similarity",
            "raw_prompt_text",
            "encoded_prompt_text",
        ]
        write_csv(space_dir / "prompt_selection_scores_all_candidates.csv", scored_rows, score_fields)
        write_csv(space_dir / "selected_representative_prompts.csv", selected_rows, score_fields)
        write_csv(space_dir / "diversity_topk_prompts.csv", diversity_rows, score_fields)
        prompt_to_anchor_fields = [
            "global_prompt_index",
            "source_file",
            "source_stem",
            "class_index",
            "class_name",
            "prompt_index",
            "own_class_similarity",
            "best_other_anchor_class",
            "best_other_anchor_similarity",
            "anchor_margin",
            *[f"sim_to_{class_name}" for class_name in class_names],
            "raw_prompt_text",
            "encoded_prompt_text",
        ]
        write_csv(space_dir / "prompt_to_class_anchor_similarity.csv", prompt_to_anchor_rows, prompt_to_anchor_fields)
        write_csv(
            space_dir / "baseline_classname_vs_selected_prompt_separation.csv",
            baseline_rows,
            [
                "class_name",
                "baseline_nearest_class",
                "baseline_nearest_similarity",
                "baseline_separation",
                "selected_nearest_class",
                "selected_nearest_similarity",
                "selected_separation",
                "separation_delta_selected_minus_baseline",
            ],
        )

        selected_payload = build_selected_payload(
            prompt_jsons,
            selected_rows,
            weights,
            args.score_preset,
            args.inter_mode,
            space_name,
            args.scoring_prompt_mode,
            args.normalize_components,
        )
        weighted_payload = build_weighted_payload_for_rows(
            diversity_rows,
            prompt_jsons,
            weights,
            args.score_preset,
            args.inter_mode,
            space_name,
            args.scoring_prompt_mode,
            args.prompt_weight_temperature,
            args.topk,
            args.diversity_mode,
            args.diversity_threshold,
            args.mmr_lambda,
            args.normalize_components,
            source_label="pooled_candidate_bank",
        )
        save_json(space_dir / "selected_representative_prompts.json", selected_payload)
        save_json(space_dir / f"diversity_topk{args.topk}_weighted_prompt_bank.json", weighted_payload)

        # Also write per-source diversity banks when a source alone can fill Top-K.
        for source_file in sorted({row["source_file"] for row in scored_rows}):
            # Copy rows so per-source diversity flags do not overwrite the pooled bank.
            source_scored = [dict(row) for row in scored_rows if row["source_file"] == source_file]
            for row in source_scored:
                row["selected_in_diversity_topk"] = False
                row["diversity_rank_in_class"] = ""
            source_diversity = build_diversity_topk_rows(
                source_scored,
                prompt_embeddings,
                class_names,
                args.topk,
                args.diversity_mode,
                args.diversity_threshold,
                args.mmr_lambda,
            )
            if len({row["class_name"] for row in source_diversity}) < len(class_names):
                print(f"Skip per-source Top-K bank for {source_file}: missing classes after diversity filter")
                continue
            source_counts = {
                class_name: len([row for row in source_diversity if row["class_name"] == class_name])
                for class_name in class_names
            }
            if len(set(source_counts.values())) != 1:
                print(
                    f"Skip per-source Top-K bank for {source_file}: unequal counts after diversity filter "
                    f"{source_counts}"
                )
                continue
            source_weighted = add_prompt_weights(source_scored, args.prompt_weight_temperature, only_diversity_topk=True)
            source_lookup = {row["global_prompt_index"]: row for row in source_weighted}
            for row in source_diversity:
                row.update(
                    {
                        "prompt_weight": source_lookup[row["global_prompt_index"]]["prompt_weight"],
                        "weight_rank_in_class": source_lookup[row["global_prompt_index"]]["weight_rank_in_class"],
                        "prompt_weight_temperature": source_lookup[row["global_prompt_index"]]["prompt_weight_temperature"],
                    }
                )
            try:
                source_payload = build_weighted_payload_for_rows(
                    source_diversity,
                    prompt_jsons,
                    weights,
                    args.score_preset,
                    args.inter_mode,
                    space_name,
                    args.scoring_prompt_mode,
                    args.prompt_weight_temperature,
                    args.topk,
                    args.diversity_mode,
                    args.diversity_threshold,
                    args.mmr_lambda,
                    args.normalize_components,
                    source_label=source_file,
                )
            except ValueError as exc:
                print(f"Skip per-source Top-K bank for {source_file}: {exc}")
                continue
            save_json(space_dir / f"diversity_topk{args.topk}_weighted_{Path(source_file).stem}.json", source_payload)

        if space_name == args.main_space:
            main_payload = selected_payload
            main_weighted_payload = weighted_payload

        summary = {
            "embedding_space": space_name,
            "prompt_jsons": [str(Path(path).resolve()) for path in prompt_jsons],
            "source_metadata": all_payloads,
            "classes": len(class_names),
            "total_candidate_prompts": len(candidate_texts),
            "weights": weights,
            "score_preset": args.score_preset,
            "inter_mode": args.inter_mode,
            "normalize_components": args.normalize_components,
            "topk": args.topk,
            "diversity_mode": args.diversity_mode,
            "diversity_threshold": args.diversity_threshold,
            "mmr_lambda": args.mmr_lambda,
            "selection_method": SELECTION_METHOD,
            "hard_negative_class_map": HARD_NEGATIVE_CLASS_MAP,
            "selected_prompts": selected_rows,
            "diversity_topk_prompts": diversity_rows,
            "baseline_vs_selected_separation": baseline_rows,
            "wrong_nearest_class_anchor_count": len([row for row in scored_rows if row["anchor_margin"] < 0]),
            "lowest_own_class_similarity_prompts": sorted(scored_rows, key=lambda row: row["anchor_similarity"])[:30],
            "lowest_total_score_prompts": sorted(scored_rows, key=lambda row: row["selection_score"])[:30],
        }
        save_json(space_dir / "prompt_selection_summary.json", summary)
        all_summaries.append(summary)
        print_console_summary(space_name, selected_rows, diversity_rows, scored_rows, baseline_rows, args.topk)

    if main_payload is not None:
        save_json(args.selected_output_json, main_payload)
        if main_weighted_payload is not None:
            save_json(args.weighted_output_json, main_weighted_payload)
        print()
        print("Main selected prompt bank for training:", args.selected_output_json)
        if main_weighted_payload is not None:
            print("Main weighted prompt bank for training:", args.weighted_output_json)

    save_json(output_root / "all_prompt_selection_summaries.json", all_summaries)
    return all_summaries


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_jsons = [Path(path).resolve() for path in args.prompt_jsons] if args.prompt_jsons else discover_prompt_jsons()
    if not prompt_jsons:
        raise FileNotFoundError(
            "No prompt JSONs found. Expected short noun-verb / merged / manual / keyword prompt banks."
        )
    for prompt_json in prompt_jsons:
        if not prompt_json.exists():
            raise FileNotFoundError(prompt_json)

    weights = resolve_score_weights(args)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEVICE:", args.device)
    print("MODEL_PATH:", args.model_path)
    print("OUTPUT_DIR:", output_root)
    print("SELECTED_OUTPUT_JSON:", args.selected_output_json)
    print("WEIGHTED_OUTPUT_JSON:", args.weighted_output_json)
    print(
        "WEIGHTS:",
        weights,
        "| inter_mode:", args.inter_mode,
        "| score_preset:", args.score_preset,
        "| normalize:", args.normalize_components,
        "| topk:", args.topk,
        "| diversity:", args.diversity_mode,
        "| tau:", args.diversity_threshold,
        "| mmr_lambda:", args.mmr_lambda,
        "| main_space:", args.main_space,
        "| scoring_prompt_mode:", args.scoring_prompt_mode,
        "| prompt_weight_temperature:", args.prompt_weight_temperature,
    )

    model = instantiate_model(args.model_path, args.device)
    analyze_candidate_pool(prompt_jsons, model, args, output_root, weights)
    print()
    print("Done. Main output dir:", output_root)


if __name__ == "__main__":
    main()
