import torch
import torch.nn.functional as F
from torch import nn

from model import CLIPVAD


class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, features, valid_mask):
        residual = features
        response = F.gelu(self.conv1(features.transpose(1, 2)))
        response = F.gelu(self.conv2(response)).transpose(1, 2)
        features = self.norm(residual + response)
        features = features * valid_mask.unsqueeze(-1)
        return features, response * valid_mask.unsqueeze(-1)


class TemporalCNNBackbone(nn.Module):
    """Temporal CNN that replaces LGT encode_video (no position embeddings)."""

    def __init__(self, input_width=512, hidden_width=128, kernel_size=3, num_blocks=2):
        super().__init__()
        if kernel_size not in (1, 3, 5, 7, 9):
            raise ValueError("kernel_size must be 1, 3, 5, 7, or 9")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        self.projection = nn.Sequential(
            nn.Linear(input_width, hidden_width), nn.LayerNorm(hidden_width), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(hidden_width, kernel_size) for _ in range(num_blocks)]
        )

    def forward(self, features, lengths):
        valid_mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        features = self.projection(features) * valid_mask.unsqueeze(-1)
        response = features
        for block in self.blocks:
            features, response = block(features, valid_mask)
        return features, response, valid_mask


class CLIPVADCNNReplaceLGT(CLIPVAD):
    """VadCLIP with Temporal CNN replacing the entire LGT-Adapter path."""

    def __init__(
        self,
        *args,
        kernel_size=3,
        score_mode="scoring_head",
        hidden_width=128,
        num_blocks=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if score_mode not in ("scoring_head", "kernel_response"):
            raise ValueError("score_mode must be scoring_head or kernel_response")
        self.score_mode = score_mode
        self.hidden_width = hidden_width
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size

        # Drop LGT and the original pointwise classifier; CNN owns temporal modeling.
        del self.temporal
        del self.gc1
        del self.gc2
        del self.gc3
        del self.gc4
        del self.disAdj
        del self.linear
        del self.gelu
        del self.classifier
        del self.mlp2
        del self.frame_position_embeddings

        self.cnn_backbone = TemporalCNNBackbone(
            input_width=self.visual_width,
            hidden_width=hidden_width,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
        )
        if hidden_width != self.visual_width:
            self.feature_proj = nn.Linear(hidden_width, self.visual_width)
        else:
            self.feature_proj = nn.Identity()
        if score_mode == "scoring_head":
            self.scorer = nn.Linear(hidden_width, 1)
        else:
            self.raw_scale = nn.Parameter(torch.tensor(0.54132485))
            self.bias = nn.Parameter(torch.zeros(()))

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        lengths = torch.as_tensor(lengths, device=images.device, dtype=torch.long)
        hidden, _, valid_mask = self.cnn_backbone(images, lengths)
        visual_features = self.feature_proj(hidden) * valid_mask.unsqueeze(-1)
        return visual_features

    def forward(self, visual, padding_mask, text, lengths, compute_alignment=True):
        visual = visual.to(torch.float)
        lengths = torch.as_tensor(lengths, device=visual.device, dtype=torch.long)
        hidden, response, valid_mask = self.cnn_backbone(visual, lengths)
        visual_features = self.feature_proj(hidden) * valid_mask.unsqueeze(-1)

        if self.score_mode == "scoring_head":
            logits1 = self.scorer(hidden)
        else:
            logits1 = F.softplus(self.raw_scale) * response.amax(dim=-1, keepdim=True) + self.bias
        logits1 = logits1.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        if not compute_alignment:
            return None, logits1, None

        text_features_ori = self.encode_textprompt(text)
        visual_attn = (logits1.permute(0, 2, 1) @ visual_features)
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        visual_attn = visual_attn.expand(-1, text_features_ori.shape[0], -1)
        text_features = text_features_ori.unsqueeze(0).expand(
            visual_attn.shape[0], -1, -1
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = F.normalize(visual_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        logits2 = visual_features_norm @ text_features_norm.permute(0, 2, 1).to(
            visual_features_norm.dtype
        ) / 0.07
        return text_features_ori, logits1, logits2
