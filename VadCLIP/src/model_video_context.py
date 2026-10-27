"""VadCLIP with a lightweight temporal context branch over one crop view."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model import CLIPVAD


class TemporalContextHead(nn.Module):
    """Encode a variable-length snippet sequence into ordered context bins."""

    def __init__(self, visual_width, bin_size=32, hidden_width=128, max_delta=0.1):
        super().__init__()
        if bin_size < 1 or hidden_width < 1 or max_delta <= 0:
            raise ValueError("Context dimensions and max_delta must be positive.")
        self.bin_size = int(bin_size)
        self.max_delta = float(max_delta)
        self.gru = nn.GRU(
            input_size=visual_width,
            hidden_size=hidden_width,
            batch_first=True,
            bidirectional=True,
        )
        context_width = hidden_width * 2
        self.residual = nn.Sequential(
            nn.Linear(visual_width + context_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.video_classifier = nn.Linear(context_width, 1)

    def forward(self, sequences):
        if not sequences or any(sequence.ndim != 2 or sequence.shape[0] == 0 for sequence in sequences):
            raise ValueError("Each view must contain a non-empty [snippets, width] sequence.")
        token_sequences = [
            torch.stack(
                [sequence[i : i + self.bin_size].mean(dim=0)
                 for i in range(0, sequence.shape[0], self.bin_size)],
                dim=0,
            )
            for sequence in sequences
        ]
        token_lengths = torch.tensor(
            [sequence.shape[0] for sequence in token_sequences], dtype=torch.long
        )
        padded_tokens = nn.utils.rnn.pad_sequence(token_sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded_tokens, token_lengths, batch_first=True, enforce_sorted=False
        )
        packed_context, hidden = self.gru(packed)
        context_tokens, _ = pad_packed_sequence(
            packed_context, batch_first=True, total_length=padded_tokens.shape[1]
        )
        video_state = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        video_logits = self.video_classifier(video_state).squeeze(-1)

        deltas = []
        for index, sequence in enumerate(sequences):
            snippet_index = torch.arange(sequence.shape[0], device=sequence.device) // self.bin_size
            context = context_tokens[index, snippet_index]
            delta = self.max_delta * torch.tanh(
                self.residual(torch.cat((sequence, context), dim=-1))
            )
            deltas.append(delta)
        return deltas, video_logits, context_tokens, token_lengths


class CLIPVADVideoContext(CLIPVAD):
    """Add temporal context without confusing spatial crops with time segments.

    ``visual`` contains temporal blocks. ``view_block_counts`` groups consecutive
    blocks that belong to one spatial crop view of a video. A GRU runs within each
    view only; different crop/flip views are never concatenated as time.
    """

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
        context_bin_size=32,
        context_hidden_width=128,
        max_context_delta=0.1,
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
        self.context_bin_size = int(context_bin_size)
        self.context_hidden_width = int(context_hidden_width)
        self.max_context_delta = float(max_context_delta)
        self.context_head = TemporalContextHead(
            visual_width,
            bin_size=context_bin_size,
            hidden_width=context_hidden_width,
            max_delta=max_context_delta,
        )

    def _local_forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        classifier_input = visual_features + self.mlp2(visual_features)
        base_logits1 = self.classifier(classifier_input)
        text_features_ori = self.encode_textprompt(text)

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
        logits2 = visual_features_norm @ text_features_norm.type(
            visual_features_norm.dtype
        ) / 0.07
        return text_features_ori, base_logits1, logits2, classifier_input

    def forward(
        self,
        visual,
        padding_mask,
        text,
        lengths,
        view_block_counts,
        return_context_details=False,
    ):
        if visual.ndim != 3:
            raise ValueError(f"Expected [blocks, time, width], got {tuple(visual.shape)}")
        if lengths is None or len(lengths) != visual.shape[0]:
            raise ValueError("One valid length is required for every temporal block.")
        counts = [int(value) for value in view_block_counts]
        if not counts or any(value < 1 for value in counts) or sum(counts) != visual.shape[0]:
            raise ValueError("view_block_counts must partition all input blocks into views.")

        text_features, base_logits1, logits2, classifier_input = self._local_forward(
            visual, padding_mask, text, lengths
        )
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long).tolist()

        view_chunk_ranges = []
        view_sequences = []
        block_cursor = 0
        for block_count in counts:
            valid_parts = []
            block_ranges = []
            view_offset = 0
            for block_index in range(block_cursor, block_cursor + block_count):
                valid_length = max(0, min(lengths_cpu[block_index], visual.shape[1]))
                if valid_length == 0:
                    raise ValueError("A temporal block cannot have zero valid snippets.")
                part = classifier_input[block_index, :valid_length]
                valid_parts.append(part)
                block_ranges.append((block_index, view_offset, view_offset + valid_length))
                view_offset += valid_length
            sequence = torch.cat(valid_parts, dim=0)
            view_sequences.append(sequence)
            view_chunk_ranges.append(block_ranges)
            block_cursor += block_count

        delta_sequences, video_logits, context_tokens, token_lengths = self.context_head(
            view_sequences
        )

        delta_blocks = [None] * visual.shape[0]
        for view_index, block_ranges in enumerate(view_chunk_ranges):
            offset = 0
            for block_index, start, end in block_ranges:
                block_length = end - start
                valid_delta = delta_sequences[view_index][offset : offset + block_length]
                padded_delta = F.pad(valid_delta, (0, 0, 0, visual.shape[1] - block_length))
                delta_blocks[block_index] = padded_delta
                offset += block_length

        delta = torch.stack(delta_blocks, dim=0)
        valid_mask = torch.arange(visual.shape[1], device=visual.device)[None, :] < lengths.to(
            device=visual.device
        )[:, None]
        adjusted_logits1 = base_logits1 + delta

        outputs = [text_features, adjusted_logits1, logits2]
        if return_context_details:
            outputs.append(
                {
                    "base_logits1": base_logits1,
                    "context_delta": delta,
                    "valid_mask": valid_mask,
                    "video_logits_per_view": video_logits,
                    "context_tokens": context_tokens,
                    "context_token_lengths": token_lengths.to(visual.device),
                }
            )
        return tuple(outputs)
