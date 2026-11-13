"""Standalone video classifier: original VadCLIP LGT + multiscale CNN.

No CLIP weights are loaded: inputs are already extracted CLIP features.
The original LGT methods are reused, including their padding behavior.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model import CLIPVAD, Transformer, QuickGELU
from utils.layers import GraphConvolution, DistanceAdj


class DeviceDistanceAdj(DistanceAdj):
    def forward(self, batch_size, max_seqlen):
        # Same distance graph as VadCLIP, without hard-coding CUDA:0.
        positions = torch.arange(max_seqlen, device=self.sigma.device, dtype=torch.float32)
        distance = (positions[:, None] - positions[None, :]).abs()
        graph = torch.exp(-distance / torch.exp(distance.new_tensor(1.)))
        return graph.unsqueeze(0).repeat(batch_size, 1, 1)


def valid_mask(lengths, batch, time, device):
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > time)).any()):
        raise ValueError("Expected one valid length in [1,T] per sequence")
    return torch.arange(time, device=device)[None, :] < lengths[:, None]


class VadCLIPVisualLGT(nn.Module):
    build_attention_mask = CLIPVAD.build_attention_mask
    adj4 = CLIPVAD.adj4
    encode_video = CLIPVAD.encode_video

    def __init__(self, visual_length=256):
        super().__init__()
        if visual_length < 8 or visual_length % 8:
            raise ValueError("visual_length must be a positive multiple of 8")
        self.visual_length = visual_length
        self.visual_width = 512
        self.attn_window = 8
        self.temporal = Transformer(512, 2, 1, self.build_attention_mask(8))
        self.gc1 = GraphConvolution(512, 256, residual=True)
        self.gc2 = GraphConvolution(256, 256, residual=True)
        self.gc3 = GraphConvolution(512, 256, residual=True)
        self.gc4 = GraphConvolution(256, 256, residual=True)
        self.disAdj = DeviceDistanceAdj()
        self.linear = nn.Linear(512, 512)
        self.gelu = QuickGELU()
        self.frame_position_embeddings = nn.Embedding(visual_length, 512)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def forward(self, visual, lengths):
        if visual.ndim != 3 or visual.shape[1:] != (self.visual_length, 512):
            raise ValueError("LGT expects [B,visual_length,512]")
        mask = valid_mask(lengths, visual.shape[0], visual.shape[1], visual.device)
        self.device = visual.device
        # Preserve the original encode_video, including positional/context padding.
        return self.encode_video(visual.masked_fill(~mask[..., None], 0), None, lengths)


class MultiscaleVideoHead(nn.Module):
    def __init__(self, kernel_sizes=(1, 3, 5, 7), channels_per_branch=32, input_width=512, activation="gelu"):
        super().__init__()
        if not kernel_sizes or any(type(k) is not int or k < 1 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")
        if len(set(kernel_sizes)) != len(kernel_sizes) or channels_per_branch < 1:
            raise ValueError("Use distinct kernels and positive channels_per_branch")
        self.kernel_sizes = tuple(kernel_sizes)
        self.channels_per_branch = channels_per_branch
        if activation not in ("gelu", "none"):
            raise ValueError("activation must be gelu or none")
        self.activation = nn.GELU() if activation == "gelu" else nn.Identity()
        self.branches = nn.ModuleList([
            nn.Conv1d(input_width, channels_per_branch, k) for k in kernel_sizes
        ])
        self.fc = nn.Linear(channels_per_branch * len(kernel_sizes), 2)

    def forward(self, features, lengths):
        mask = valid_mask(lengths, features.shape[0], features.shape[1], features.device)
        x = features.masked_fill(~mask[..., None], 0).transpose(1, 2)
        responses = torch.cat([
            self.activation(conv(F.pad(x, ((k - 1) // 2, k // 2))))
            for k, conv in zip(self.kernel_sizes, self.branches)
        ], dim=1)
        # Both raw and GELU responses can be negative: padding must not win.
        pooled, indices = responses.masked_fill(~mask[:, None, :], -torch.inf).max(dim=-1)
        video_logits = self.fc(pooled)
        frame_logits = self.fc(responses.transpose(1, 2))
        contrast_weights = self.fc.weight[1] - self.fc.weight[0]
        return {
            "video_logits": video_logits,
            "frame_logits": frame_logits.masked_fill(~mask[..., None], 0),
            "responses": responses.masked_fill(~mask[:, None, :], 0),
            "pooled": pooled,
            "argmax": indices,
            "contributions": pooled * contrast_weights,
        }


class MultiscaleCNN(nn.Module):
    def __init__(self, kernel_sizes=(1, 3, 5, 7), channels_per_branch=32, visual_length=256, activation="gelu"):
        super().__init__()
        self.lgt = VadCLIPVisualLGT(visual_length)
        self.head = MultiscaleVideoHead(kernel_sizes, channels_per_branch, activation=activation)

    def forward(self, visual, lengths):
        return self.head(self.lgt(visual, lengths), lengths)

    def forward_video(self, chunks, lengths, micro_batch=8):
        """Encode 256-blocks, discard padding, then CNN/pool once over the video."""
        if micro_batch < 1 or chunks.shape[0] != len(lengths):
            raise ValueError("Invalid chunk batch or micro_batch")
        encoded = []
        for start in range(0, len(chunks), micro_batch):
            batch_lengths = lengths[start:start + micro_batch]
            x = self.lgt(chunks[start:start + micro_batch], batch_lengths)
            encoded.extend(row[:int(length)] for row, length in zip(x, batch_lengths))
        sequence = torch.cat(encoded, dim=0).unsqueeze(0)
        return self.head(sequence, [sequence.shape[1]])
