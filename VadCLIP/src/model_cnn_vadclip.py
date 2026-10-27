import torch
import torch.nn.functional as F
from torch import nn

from model import CLIPVAD


class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, use_residual=True, input_channels=None):
        super().__init__()
        input_channels = channels if input_channels is None else input_channels
        self.conv1 = nn.Conv1d(input_channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels)
        self.use_residual = use_residual
        self.skip_projection = nn.Linear(input_channels, channels) if input_channels != channels and use_residual else nn.Identity()

    def forward(self, features, valid_mask):
        residual = self.skip_projection(features) if self.use_residual else None
        response = F.gelu(self.conv1(features.transpose(1, 2)))
        response = F.gelu(self.conv2(response)).transpose(1, 2)
        features = self.norm(residual + response if self.use_residual else response)
        features = features * valid_mask.unsqueeze(-1)
        return features, response * valid_mask.unsqueeze(-1)


class TemporalCNNClassifier(nn.Module):
    def __init__(self, input_width=512, hidden_width=128, kernel_size=3, score_mode="scoring_head", use_residual=True, projection_width=None):
        super().__init__()
        projection_width = hidden_width if projection_width is None else projection_width
        if input_width < 1 or hidden_width < 1 or projection_width < 1:
            raise ValueError("input_width, hidden_width and projection_width must be positive")
        if kernel_size not in (1, 3, 5, 7, 9):
            raise ValueError("kernel_size must be 1, 3, 5, 7, or 9")
        if score_mode not in ("scoring_head", "kernel_response"):
            raise ValueError("score_mode must be scoring_head or kernel_response")
        self.score_mode = score_mode
        self.projection = nn.Sequential(
            nn.Linear(input_width, projection_width), nn.LayerNorm(projection_width), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(hidden_width, kernel_size, use_residual=use_residual, input_channels=projection_width),
             TemporalConvBlock(hidden_width, kernel_size, use_residual=use_residual)]
        )
        if score_mode == "scoring_head":
            self.scorer = nn.Linear(hidden_width, 1)
        else:
            self.raw_scale = nn.Parameter(torch.tensor(0.54132485))
            self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, lengths):
        valid_mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        features = self.projection(features) * valid_mask.unsqueeze(-1)
        for block in self.blocks:
            features, response = block(features, valid_mask)
        if self.score_mode == "scoring_head":
            scores = self.scorer(features)
        else:
            scores = F.softplus(self.raw_scale) * response.amax(dim=-1, keepdim=True) + self.bias
        return scores.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class CLIPVADCNN(CLIPVAD):
    def __init__(self, *args, kernel_size=3, score_mode="scoring_head", hidden_width=128, use_residual=True, projection_width=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The original pointwise classifier path is replaced, not used as a residual.
        del self.classifier
        del self.mlp2
        self.cnn_classifier = TemporalCNNClassifier(
            input_width=self.visual_width,
            hidden_width=hidden_width,
            projection_width=projection_width,
            kernel_size=kernel_size,
            score_mode=score_mode,
            use_residual=use_residual,
        )

    def forward(self, visual, padding_mask, text, lengths, compute_alignment=True):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        lengths = torch.as_tensor(lengths, device=visual_features.device, dtype=torch.long)
        logits1 = self.cnn_classifier(visual_features, lengths)
        if not compute_alignment:
            return None, logits1, None

        # Preserve VadCLIP's score-conditioned text alignment, using the new logits1.
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
