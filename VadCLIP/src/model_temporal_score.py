"""VadCLIP with a small, bounded temporal correction to classifier logits."""

import torch
import torch.nn.functional as F
from torch import nn

from model import CLIPVAD


class TemporalScoreCorrection(nn.Module):
    """Produce one bounded score correction per valid temporal position."""

    def __init__(self, visual_width, mode, hidden_width=128, max_delta=0.5):
        super().__init__()
        if mode not in {"feature_sequence", "feature_difference"}:
            raise ValueError(f"Unsupported temporal input mode: {mode}")
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")

        self.mode = mode
        self.max_delta = float(max_delta)
        input_width = visual_width if mode == "feature_sequence" else visual_width * 2
        self.temporal_conv = nn.Conv1d(input_width, hidden_width, kernel_size=3)
        self.activation = nn.GELU()
        self.score_projection = nn.Conv1d(hidden_width, 1, kernel_size=1)
        nn.init.zeros_(self.score_projection.weight)
        nn.init.zeros_(self.score_projection.bias)

    @staticmethod
    def _prepare_valid_sequence(features, lengths):
        batch_size, time_steps, _ = features.shape
        if lengths is None:
            valid_lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device=features.device
            )
        else:
            valid_lengths = lengths.to(device=features.device, dtype=torch.long)
            valid_lengths = valid_lengths.clamp(min=1, max=time_steps)

        time_index = torch.arange(time_steps, device=features.device).unsqueeze(0)
        valid_mask = time_index < valid_lengths.unsqueeze(1)
        last_index = (valid_lengths - 1).view(batch_size, 1, 1)
        last_feature = features.gather(
            1, last_index.expand(-1, 1, features.shape[-1])
        )
        filled = torch.where(valid_mask.unsqueeze(-1), features, last_feature)
        return filled, valid_mask

    def _make_input(self, features, valid_lengths):
        if self.mode == "feature_sequence":
            return features

        batch_size, time_steps, width = features.shape
        positions = torch.arange(time_steps, device=features.device).unsqueeze(0)
        lengths = valid_lengths.unsqueeze(1)
        previous_index = (positions - 1).clamp_min(0)
        next_index = torch.minimum(positions + 1, lengths - 1)
        previous = features.gather(
            1, previous_index.unsqueeze(-1).expand(batch_size, -1, width)
        )
        following = features.gather(
            1, next_index.unsqueeze(-1).expand(batch_size, -1, width)
        )
        return torch.cat((features - previous, following - features), dim=-1)

    def forward(self, features, lengths=None):
        filled, valid_mask = self._prepare_valid_sequence(features, lengths)
        if lengths is None:
            valid_lengths = torch.full(
                (features.shape[0],), features.shape[1],
                dtype=torch.long, device=features.device,
            )
        else:
            valid_lengths = lengths.to(device=features.device, dtype=torch.long)
            valid_lengths = valid_lengths.clamp(min=1, max=features.shape[1])

        temporal_input = self._make_input(filled, valid_lengths).transpose(1, 2)
        temporal_input = F.pad(temporal_input, (1, 1), mode="replicate")
        hidden = self.activation(self.temporal_conv(temporal_input))
        raw_delta = self.score_projection(hidden).transpose(1, 2)
        delta = self.max_delta * torch.tanh(raw_delta)
        return delta * valid_mask.unsqueeze(-1), valid_mask


class CLIPVADTemporalScore(CLIPVAD):
    """Keep VadCLIP intact and add a bounded temporal residual to logits1."""

    def __init__(
        self,
        num_class,
        embed_dim,
        visual_length,
        visual_width,
        visual_head,
        visual_layers,
        attn_window,
        prompt_prefix,
        prompt_postfix,
        device,
        temporal_mode="feature_sequence",
        temporal_hidden_width=128,
        max_temporal_delta=0.5,
    ):
        super().__init__(
            num_class,
            embed_dim,
            visual_length,
            visual_width,
            visual_head,
            visual_layers,
            attn_window,
            prompt_prefix,
            prompt_postfix,
            device,
        )
        self.temporal_score_correction = TemporalScoreCorrection(
            visual_width,
            temporal_mode,
            hidden_width=temporal_hidden_width,
            max_delta=max_temporal_delta,
        )
        self.temporal_correction_enabled = True

    def forward(
        self, visual, padding_mask, text, lengths, return_temporal_details=False
    ):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        classifier_input = visual_features + self.mlp2(visual_features)
        base_logits1 = self.classifier(classifier_input)
        correction, valid_mask = self.temporal_score_correction(
            classifier_input, lengths
        )
        if not self.temporal_correction_enabled:
            correction = torch.zeros_like(correction)
        logits1 = base_logits1 + correction

        text_features_ori = self.encode_textprompt(text)

        # Preserve VadCLIP's original classifier-to-alignment coupling.
        logits_attn = base_logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(
            visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2]
        )
        text_features = text_features_ori.unsqueeze(0).expand(
            visual_attn.shape[0], -1, -1
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        outputs = [text_features_ori, logits1, logits2]
        if return_temporal_details:
            outputs.append(
                {
                    "base_logits1": base_logits1,
                    "temporal_delta": correction,
                    "valid_mask": valid_mask,
                }
            )
        return tuple(outputs)
