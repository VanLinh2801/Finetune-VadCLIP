import torch
import torch.nn.functional as F

from model import CLIPVAD


class CLIPVADPromptReplacement(CLIPVAD):
    """Collapse each class prompt group into one prototype vector p_c.

    p_c = Normalize(mean(e_k)) or Normalize(sum_k w_k e_k)
    Then use the single prototype embedding in place of class-name text features.
    """

    def encode_textprompt_groups(self, prompt_groups, prompt_weights=None):
        if not prompt_groups:
            raise ValueError("prompt_groups must not be empty.")
        if isinstance(prompt_groups[0], str):
            return self.encode_textprompt(prompt_groups)

        flat_prompts = []
        group_sizes = []
        for group in prompt_groups:
            if not group:
                raise ValueError("Each prompt group must contain at least one prompt.")
            flat_prompts.extend(group)
            group_sizes.append(len(group))

        flat_features = self.encode_textprompt(flat_prompts)
        if prompt_weights is not None:
            weights = prompt_weights.to(flat_features.device, dtype=torch.float32)
            if weights.ndim != 2:
                raise ValueError(f"prompt_weights must be 2D [num_classes, K], got {tuple(weights.shape)}")
            if weights.shape[0] != len(group_sizes):
                raise ValueError(
                    f"prompt_weights rows {weights.shape[0]} != num classes {len(group_sizes)}"
                )

        grouped_features = []
        cursor = 0
        for class_index, group_size in enumerate(group_sizes):
            group_features = F.normalize(flat_features[cursor: cursor + group_size].float(), dim=-1)
            if prompt_weights is None:
                prototype = group_features.mean(dim=0, keepdim=True)
            else:
                if weights.shape[1] != group_size:
                    raise ValueError(
                        f"prompt_weights K={weights.shape[1]} != prompts for class {class_index}: {group_size}"
                    )
                class_weights = weights[class_index]
                class_weights = class_weights / class_weights.sum().clamp_min(1e-8)
                prototype = (group_features * class_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            prototype = F.normalize(prototype, dim=-1).squeeze(0)
            grouped_features.append(prototype.to(flat_features.dtype))
            cursor += group_size

        return torch.stack(grouped_features, dim=0)

    def forward(self, visual, padding_mask, text, lengths, return_visual=False, prompt_weights=None):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt_groups(text, prompt_weights=prompt_weights)

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        if return_visual:
            return text_features_ori, logits1, logits2, visual_features

        return text_features_ori, logits1, logits2
