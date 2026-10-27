import torch
from torch import nn

from model import CLIPVAD


class ResidualClassifierAdapter(nn.Module):
    """Small head-specific feature transform with an identity residual path."""

    def __init__(self, width, bottleneck, dropout, residual_scale):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("adapter bottleneck must be positive")
        if residual_scale < 0:
            raise ValueError("adapter residual scale must be non-negative")
        self.residual_scale = float(residual_scale)
        self.adapter = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, bottleneck),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(bottleneck, width),
        )

    def forward(self, features):
        return features + self.residual_scale * self.adapter(features)


class CLIPVADMultiClassifier(CLIPVAD):
    """VadCLIP with parallel binary anomaly heads and mean-logit aggregation."""

    def __init__(
        self,
        classes_num,
        embed_dim,
        visual_length,
        visual_width,
        visual_head,
        visual_layers,
        attn_window,
        prompt_prefix,
        prompt_postfix,
        device,
        num_classifier_heads=2,
    ):
        super().__init__(
            classes_num,
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
        if num_classifier_heads < 2:
            raise ValueError("num_classifier_heads must be at least 2")
        self.num_classifier_heads = int(num_classifier_heads)
        del self.classifier
        self.classifier_heads = nn.ModuleList(
            nn.Linear(visual_width, 1) for _ in range(self.num_classifier_heads)
        )

    def classify_features(self, classifier_features):
        return torch.cat(
            [head(classifier_features) for head in self.classifier_heads], dim=-1
        )

    def forward(
        self,
        visual,
        padding_mask,
        text,
        lengths,
        return_visual=False,
        return_head_logits=False,
    ):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        classifier_features = visual_features + self.mlp2(visual_features)
        head_logits = self.classify_features(classifier_features)
        logits1 = head_logits.mean(dim=-1, keepdim=True)

        text_features_ori = self.encode_textprompt(text)
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        visual_attn = visual_attn.expand(
            visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2]
        )
        text_features = text_features_ori.unsqueeze(0).expand(
            visual_attn.shape[0], -1, -1
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        text_features_norm = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        logits2 = (
            visual_features_norm
            @ text_features_norm.permute(0, 2, 1).type(visual_features_norm.dtype)
            / 0.07
        )

        outputs = [text_features_ori, logits1, logits2]
        if return_visual:
            outputs.append(visual_features)
        if return_head_logits:
            outputs.append(head_logits)
        return tuple(outputs)


class CLIPVADAdapterMultiClassifier(CLIPVADMultiClassifier):
    """Multi-head VadCLIP with an independent residual adapter for each head."""

    def __init__(
        self,
        classes_num,
        embed_dim,
        visual_length,
        visual_width,
        visual_head,
        visual_layers,
        attn_window,
        prompt_prefix,
        prompt_postfix,
        device,
        num_classifier_heads=2,
        adapter_bottleneck=128,
        adapter_dropout=0.1,
        adapter_residual_scale=0.1,
    ):
        super().__init__(
            classes_num,
            embed_dim,
            visual_length,
            visual_width,
            visual_head,
            visual_layers,
            attn_window,
            prompt_prefix,
            prompt_postfix,
            device,
            num_classifier_heads=num_classifier_heads,
        )
        self.classifier_adapters = nn.ModuleList(
            ResidualClassifierAdapter(
                visual_width,
                adapter_bottleneck,
                adapter_dropout,
                adapter_residual_scale,
            )
            for _ in range(self.num_classifier_heads)
        )

    def classify_features(self, classifier_features):
        return torch.cat(
            [
                head(adapter(classifier_features))
                for adapter, head in zip(
                    self.classifier_adapters, self.classifier_heads
                )
            ],
            dim=-1,
        )
