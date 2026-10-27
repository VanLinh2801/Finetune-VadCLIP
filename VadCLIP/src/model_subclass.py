"""VadCLIP with per-subclass alignment; aggregation is deliberately video-level."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from clip import clip
from model import CLIPVAD


CLASS_NAMES = (
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
)


def load_subclass_bank(path, prompt_prefix=10, prompt_postfix=10):
    with Path(path).open(encoding="utf-8") as f:
        bank = json.load(f)
    classes = bank.get("classes", {})
    if set(classes) != set(CLASS_NAMES):
        raise ValueError(f"Subclass classes must equal {CLASS_NAMES}; got {tuple(classes)}")
    groups = [classes[name] for name in CLASS_NAMES]
    if any(not isinstance(group, list) or not group for group in groups):
        raise ValueError("Every class must have a nonempty subclass list")
    declared_count = bank.get("metadata", {}).get("subclasses_per_class")
    counts = [len(group) for group in groups]
    if isinstance(declared_count, int) and any(count != declared_count for count in counts):
        raise ValueError(f"Metadata declares {declared_count} subclasses per class, but found {counts}")
    if declared_count not in (None, "variable") and not isinstance(declared_count, int):
        raise ValueError("metadata.subclasses_per_class must be an integer or 'variable'")
    declared_counts = bank.get("class_subclass_counts")
    if declared_counts is not None:
        actual_counts = {name: len(group) for name, group in zip(CLASS_NAMES, groups)}
        if declared_counts != actual_counts:
            raise ValueError(f"class_subclass_counts does not match classes: {actual_counts}")
    all_text = [text for group in groups for text in group]
    if any(not isinstance(text, str) or not text.strip() for text in all_text):
        raise ValueError("Subclass names must be nonempty strings")
    if len(set(all_text)) != len(all_text):
        raise ValueError("Subclass names must be unique")
    tokens = clip.tokenize(all_text, truncate=False)
    eot = tokens.argmax(dim=-1)
    if (eot + prompt_prefix + prompt_postfix >= 77).any():
        raise ValueError("A subclass exceeds the VadCLIP learnable-prompt context")
    return groups


def subclass_group_mask(prompt_groups, device=None):
    """Return [C,Kmax] mask for ragged subclass groups."""
    if not prompt_groups or any(not group for group in prompt_groups):
        raise ValueError("Prompt groups must be nonempty")
    max_count = max(len(group) for group in prompt_groups)
    counts = torch.tensor([len(group) for group in prompt_groups], device=device)
    return torch.arange(max_count, device=device).unsqueeze(0) < counts.unsqueeze(1)


def subclass_video_scores(subclass_logits, lengths):
    """Original MIL Top-K, applied independently to every subclass."""
    if subclass_logits.ndim != 4:
        raise ValueError("Expected subclass logits [B,T,C,K]")
    scores = []
    for sample, length in zip(subclass_logits, lengths):
        valid = int(length)
        if not 1 <= valid <= sample.shape[0]:
            raise ValueError(f"Invalid snippet length: {valid}")
        k = min(valid, valid // 16 + 1)
        scores.append(sample[:valid].topk(k, dim=0).values.mean(dim=0))
    if len(scores) != subclass_logits.shape[0]:
        raise ValueError("Batch size and lengths differ")
    return torch.stack(scores)


def aggregate_class_scores(subclass_scores, mode, temperature=1.0, subclass_mask=None):
    return aggregate_subclass_logits(subclass_scores, mode, temperature, subclass_mask)


def aggregate_subclass_logits(subclass_logits, mode, temperature=1.0, subclass_mask=None,
                              return_weights=False):
    """Aggregate K subclasses at each snippet before temporal MIL Top-K."""
    if subclass_logits.ndim < 3:
        raise ValueError("Expected subclass logits with trailing dimensions [C,K]")
    if subclass_mask is None:
        subclass_mask = torch.isfinite(subclass_logits)
    else:
        subclass_mask = subclass_mask.to(device=subclass_logits.device, dtype=torch.bool)
        if subclass_mask.shape != subclass_logits.shape[-2:]:
            raise ValueError("Subclass mask must have shape [C,K]")
        view_shape = (1,) * (subclass_logits.ndim - 2) + tuple(subclass_mask.shape)
        subclass_mask = subclass_mask.view(view_shape).expand_as(subclass_logits)
    if not subclass_mask.any(dim=-1).all():
        raise ValueError("Every class must have at least one valid subclass")
    masked_logits = subclass_logits.masked_fill(~subclass_mask, -torch.inf)
    if mode == "max":
        class_logits, indices = masked_logits.max(dim=-1)
        weights = F.one_hot(indices, masked_logits.shape[-1]).to(subclass_logits.dtype)
    elif mode == "soft" and temperature > 0:
        valid_count = subclass_mask.sum(dim=-1).to(subclass_logits.dtype)
        class_logits = (temperature * torch.logsumexp(masked_logits / temperature, dim=-1)
                        - temperature * torch.log(valid_count))
        weights = torch.softmax(masked_logits / temperature, dim=-1).masked_fill(~subclass_mask, 0)
    else:
        raise ValueError("mode must be max or soft with positive temperature")
    if return_weights:
        return class_logits, weights
    return class_logits


def aggregate_video_frames(subclass_logits, mode, temperature=1.0, subclass_mask=None):
    """Aggregate subclasses per frame; video-level weights are diagnostics only."""
    scores = subclass_video_scores(subclass_logits.unsqueeze(0), [len(subclass_logits)])[0]
    frame_logits, frame_weights = aggregate_subclass_logits(
        subclass_logits, mode, temperature, subclass_mask, return_weights=True
    )
    # This summarizes which subclass contributed most across the video. It is
    # not used to construct frame logits or to choose MIL snippets.
    video_weights = frame_weights.mean(dim=0)
    return frame_logits, video_weights, scores


def text_regularization(class_text_features):
    """The paper's Normal-vs-class penalty on class representatives only."""
    class_text_features = F.normalize(class_text_features.float(), dim=-1)
    return (class_text_features[1:] @ class_text_features[0]).abs().mean() * 0.1


class CLIPVADSubclass(CLIPVAD):
    def forward(self, visual, padding_mask, prompt_groups, lengths):
        if len(prompt_groups) != self.num_class:
            raise ValueError("Prompt groups and model class count differ")
        if any(not group for group in prompt_groups):
            raise ValueError("Every class must have at least one subclass")
        k = max(len(group) for group in prompt_groups)
        subclass_mask = subclass_group_mask(prompt_groups, visual.device)

        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))
        flat_text = [text for group in prompt_groups for text in group]
        encoded_text = self.encode_textprompt(flat_text)
        chunks = encoded_text.split([len(group) for group in prompt_groups])
        original_text = torch.stack([
            F.pad(chunk, (0, 0, 0, k - len(chunk))) for chunk in chunks
        ])

        # Centroids are used only by the original text regularizer, never for alignment.
        normalized_text = F.normalize(original_text.float(), dim=-1)
        class_text = (normalized_text * subclass_mask.unsqueeze(-1)).sum(dim=1)
        class_text = class_text / subclass_mask.sum(dim=1, keepdim=True)
        class_text = F.normalize(class_text, dim=-1)
        visual_attn = logits1.transpose(1, 2) @ visual_features
        visual_attn = F.normalize(visual_attn, dim=-1).unsqueeze(2)
        conditioned_text = original_text.unsqueeze(0) + visual_attn
        conditioned_text = conditioned_text + self.mlp1(conditioned_text)
        visual_norm = F.normalize(visual_features, dim=-1)
        text_norm = F.normalize(conditioned_text, dim=-1)
        subclass_logits = torch.einsum("btd,bckd->btck", visual_norm, text_norm.to(visual_norm.dtype)) / 0.07
        subclass_logits = subclass_logits.masked_fill(~subclass_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        return class_text, logits1, subclass_logits
