import math

import torch
import torch.nn.functional as F
from torch import nn

from model import CLIPVAD


class CLIPVADTemporalProposal(CLIPVAD):
    """VadCLIP encoder with learnable 1D temporal event proposals."""

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
        num_proposals=8,
        proposal_heads=4,
        min_width=0.02,
        max_width=0.50,
        initial_width=0.15,
        mask_temperature=0.02,
        context_scale=0.25,
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
        if num_proposals < 1:
            raise ValueError("num_proposals must be at least 1")
        if not 0 < min_width < max_width <= 1:
            raise ValueError("Proposal widths must satisfy 0 < min_width < max_width <= 1")
        if not min_width <= initial_width <= max_width:
            raise ValueError("initial_width must lie between min_width and max_width")
        if mask_temperature <= 0:
            raise ValueError("mask_temperature must be positive")

        self.num_proposals = int(num_proposals)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.mask_temperature = float(mask_temperature)
        self.context_scale = float(context_scale)

        self.event_queries = nn.Parameter(torch.empty(num_proposals, visual_width))
        nn.init.normal_(self.event_queries, std=0.02)
        self.query_attention = nn.MultiheadAttention(
            visual_width, proposal_heads, batch_first=True
        )
        self.query_norm1 = nn.LayerNorm(visual_width)
        self.query_ffn = nn.Sequential(
            nn.Linear(visual_width, visual_width * 2),
            nn.GELU(),
            nn.Linear(visual_width * 2, visual_width),
        )
        self.query_norm2 = nn.LayerNorm(visual_width)

        self.boundary_head = nn.Linear(visual_width, 2)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)
        width_fraction = (initial_width - min_width) / (max_width - min_width)
        width_fraction = min(max(width_fraction, 1e-4), 1 - 1e-4)
        self.boundary_head.bias.data[1] = math.log(width_fraction / (1 - width_fraction))

        centers = (torch.arange(num_proposals, dtype=torch.float32) + 0.5) / num_proposals
        centers = centers.clamp(1e-4, 1 - 1e-4)
        self.register_buffer("base_center_logits", torch.logit(centers))

        self.proposal_norm = nn.LayerNorm(visual_width)
        self.proposal_anomaly_head = nn.Linear(visual_width, 1)
        nn.init.zeros_(self.proposal_anomaly_head.weight)
        nn.init.constant_(self.proposal_anomaly_head.bias, -2.0)

    def _valid_mask(self, lengths, batch_size, time_steps, device, dtype):
        valid = torch.zeros(batch_size, time_steps, device=device, dtype=dtype)
        for index in range(batch_size):
            length = time_steps if lengths is None else int(lengths[index])
            valid[index, : max(1, min(length, time_steps))] = 1
        return valid

    def _soft_interval_mask(self, centers, widths, positions, temperature):
        left = centers - widths / 2
        right = centers + widths / 2
        enter = torch.sigmoid((positions - left.unsqueeze(-1)) / temperature)
        leave = torch.sigmoid((right.unsqueeze(-1) - positions) / temperature)
        return enter * leave

    def _build_proposals(self, visual_features, lengths):
        batch_size, time_steps, _ = visual_features.shape
        valid_mask = self._valid_mask(
            lengths,
            batch_size,
            time_steps,
            visual_features.device,
            visual_features.dtype,
        )
        queries = self.event_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.query_attention(
            queries,
            visual_features,
            visual_features,
            key_padding_mask=~valid_mask.bool(),
        )
        queries = self.query_norm1(queries + attended)
        queries = self.query_norm2(queries + self.query_ffn(queries))

        boundary_delta = self.boundary_head(queries)
        centers = torch.sigmoid(self.base_center_logits.view(1, -1) + boundary_delta[..., 0])
        widths = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(
            boundary_delta[..., 1]
        )

        positions = (
            (torch.arange(time_steps, device=visual_features.device, dtype=visual_features.dtype) + 0.5)
            / time_steps
        ).view(1, 1, time_steps)
        proposal_masks = self._soft_interval_mask(
            centers, widths, positions, self.mask_temperature
        )
        proposal_masks = proposal_masks * valid_mask.unsqueeze(1)

        mask_denominator = proposal_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled_visual = torch.einsum("bmt,btd->bmd", proposal_masks, visual_features)
        pooled_visual = pooled_visual / mask_denominator
        proposal_features = self.proposal_norm(pooled_visual + queries)

        context_widths = (widths * (1 + 2 * self.context_scale)).clamp(max=1.0)
        outer_masks = self._soft_interval_mask(
            centers, context_widths, positions, self.mask_temperature
        )
        context_masks = (outer_masks - proposal_masks).clamp_min(0) * valid_mask.unsqueeze(1)
        context_denominator = context_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context_features = torch.einsum("bmt,btd->bmd", context_masks, visual_features)
        context_features = context_features / context_denominator
        boundary_distance = 1 - F.cosine_similarity(
            pooled_visual.float(), context_features.float(), dim=-1
        )

        return {
            "query_features": queries,
            "proposal_features": proposal_features,
            "proposal_masks": proposal_masks,
            "context_masks": context_masks,
            "centers": centers,
            "widths": widths,
            "boundary_distance": boundary_distance,
            "valid_mask": valid_mask,
        }

    def _adapt_text_features(self, visual_features, snippet_logits, text_features_ori):
        visual_attention = snippet_logits.permute(0, 2, 1) @ visual_features
        visual_attention = F.normalize(visual_attention.float(), dim=-1).to(visual_features.dtype)
        visual_attention = visual_attention.expand(
            visual_attention.shape[0], text_features_ori.shape[0], visual_attention.shape[-1]
        )
        text_features = text_features_ori.unsqueeze(0).expand(
            visual_attention.shape[0], -1, -1
        )
        return text_features + visual_attention + self.mlp1(text_features)

    def forward(self, visual, padding_mask, text, lengths, return_proposals=False):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        proposal_outputs = self._build_proposals(visual_features, lengths)
        proposal_features = proposal_outputs["proposal_features"]
        proposal_masks = proposal_outputs["proposal_masks"]

        proposal_logits = self.proposal_anomaly_head(proposal_features).squeeze(-1)
        proposal_confidence = torch.sigmoid(proposal_logits)
        contribution = proposal_confidence.unsqueeze(-1) * proposal_masks
        snippet_probability = 1 - torch.prod((1 - contribution).clamp_min(1e-6), dim=1)
        snippet_probability = snippet_probability.clamp(1e-5, 1 - 1e-5)
        snippet_logits = torch.logit(snippet_probability).unsqueeze(-1)

        text_features_ori = self.encode_textprompt(text)
        text_features = self._adapt_text_features(
            visual_features, snippet_logits, text_features_ori
        )
        text_features_norm = F.normalize(text_features.float(), dim=-1)
        visual_features_norm = F.normalize(visual_features.float(), dim=-1)
        original_class_logits = torch.einsum(
            "btd,bcd->btc", visual_features_norm, text_features_norm
        ) / 0.07

        proposal_features_norm = F.normalize(proposal_features.float(), dim=-1)
        proposal_class_logits = torch.einsum(
            "bmd,bcd->bmc", proposal_features_norm, text_features_norm
        ) / 0.07
        class_projection_weights = contribution / contribution.sum(dim=1, keepdim=True).clamp_min(
            1e-6
        )
        projected_class_logits = torch.einsum(
            "bmt,bmc->btc", class_projection_weights, proposal_class_logits
        )
        proposal_gate = snippet_probability.unsqueeze(-1)
        snippet_class_logits = (
            (1 - proposal_gate) * original_class_logits + proposal_gate * projected_class_logits
        )

        proposal_outputs.update(
            {
                "proposal_logits": proposal_logits,
                "proposal_confidence": proposal_confidence,
                "proposal_class_logits": proposal_class_logits,
                "snippet_probability": snippet_probability,
            }
        )
        outputs = [text_features_ori, snippet_logits, snippet_class_logits]
        if return_proposals:
            outputs.append(proposal_outputs)
        return tuple(outputs)
