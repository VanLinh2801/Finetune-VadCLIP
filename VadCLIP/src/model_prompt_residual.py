import torch
import torch.nn as nn
import torch.nn.functional as F

from model_prompt_replacement import CLIPVADPromptReplacement


class CLIPVADPromptResidual(CLIPVADPromptReplacement):
    """Hướng 3: learnable residual on live class prototypes.

    Each forward:
        p_c = Normalize(weighted_pool(encode(prompts)))   # encode uses learnable soft prompts
        t_c = Normalize(p_c + Delta_c)

    class_prototypes buffer is only a reference snapshot for logging (not used in forward).
    Delta_c is an nn.Parameter initialized to zeros.

    Train modes (see ucf_train_prompt_residual.py):
      - from_scratch: train VAD + soft prompt + Delta
      - warmstart_continue: load Hướng 2b, keep training VAD + soft prompt + Delta
      - warmstart_delta_only: load Hướng 2b, freeze VAD; train soft prompt + Delta
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
        self.num_class = num_class
        self.embed_dim = embed_dim
        self.register_buffer(
            "class_prototypes",
            torch.zeros(num_class, embed_dim, device=device),
            persistent=True,
        )
        self.class_residuals = nn.Parameter(torch.zeros(num_class, embed_dim, device=device))
        self.prototypes_initialized = False

    @torch.no_grad()
    def init_class_prototypes(self, prompt_groups, prompt_weights=None):
        """Snapshot p_c for logging; residuals stay learnable. Forward re-encodes every step."""
        prototypes = self.encode_textprompt_groups(prompt_groups, prompt_weights=prompt_weights)
        if prototypes.shape != self.class_prototypes.shape:
            raise ValueError(
                f"Prototype shape {tuple(prototypes.shape)} != "
                f"expected {tuple(self.class_prototypes.shape)}"
            )
        self.class_prototypes.copy_(F.normalize(prototypes.float(), dim=-1))
        self.prototypes_initialized = True

    def encode_residual_text_features(self, prompt_groups, prompt_weights=None):
        """Live encode (soft prompts trainable) then apply Delta."""
        base = self.encode_textprompt_groups(prompt_groups, prompt_weights=prompt_weights)
        base = F.normalize(base.float(), dim=-1)
        return F.normalize(base + self.class_residuals, dim=-1)

    def get_residual_text_features(self):
        """Reference path using cached snapshot + Delta (for stats only)."""
        return F.normalize(self.class_prototypes + self.class_residuals, dim=-1)

    def residual_l2_loss(self):
        return (self.class_residuals.float() ** 2).sum(dim=-1).mean()

    @torch.no_grad()
    def residual_stats(self):
        norms = self.class_residuals.float().norm(dim=-1)
        text = self.get_residual_text_features().float()
        cosine = (text * self.class_prototypes.float()).sum(dim=-1)
        return {
            "mean_norm": float(norms.mean().item()),
            "max_norm": float(norms.max().item()),
            "mean_cosine_to_pc": float(cosine.mean().item()),
            "min_cosine_to_pc": float(cosine.min().item()),
        }

    def forward(self, visual, padding_mask, text, lengths, return_visual=False, prompt_weights=None):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_residual_text_features(text, prompt_weights=prompt_weights)
        text_features_ori = text_features_ori.to(visual_features.dtype)

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(
            visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2]
        )
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(
            visual_attn.shape[0], text_features.shape[1], text_features.shape[2]
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        if return_visual:
            return text_features_ori, logits1, logits2, visual_features
        return text_features_ori, logits1, logits2
