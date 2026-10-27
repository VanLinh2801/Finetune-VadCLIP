import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_multi_prompt import CLIPVADMultiPrompt


class CLIPVADMultiPromptResidual(CLIPVADMultiPrompt):
    """Hướng 2a + 3: per-prompt residual on live multi-prompt embeddings.

    Each forward, for class c and prompt k:
        e_{c,k} = encode(prompt_{c,k})   # uses learnable soft prompts
        t_{c,k} = Normalize(e_{c,k} + Delta_{c,k})
    Then keep the same weighted logsumexp aggregation as multi-prompt (Hướng 2a).

    prompt_embeddings buffer is only a reference snapshot for logging (not used in forward).

    Train modes (see ucf_train_multi_prompt_residual.py):
      - from_scratch: train VAD + soft prompt + Delta
      - warmstart_continue: load Hướng 2a, keep training VAD + soft prompt + Delta
      - warmstart_delta_only: load Hướng 2a, freeze VAD; train soft prompt + Delta
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
        self.prompts_initialized = False

    @torch.no_grad()
    def init_prompt_embeddings(self, prompt_groups):
        """Allocate/zero Delta and snapshot embeddings for logging. Forward re-encodes every step."""
        embeddings = self.encode_textprompt_groups(prompt_groups)
        embeddings = F.normalize(embeddings.float(), dim=-1)
        if "prompt_embeddings" not in self._buffers:
            self.register_buffer("prompt_embeddings", embeddings.clone(), persistent=True)
            self.prompt_residuals = nn.Parameter(torch.zeros_like(embeddings))
        else:
            if embeddings.shape != self.prompt_embeddings.shape:
                raise ValueError(
                    f"Prompt embedding shape {tuple(embeddings.shape)} != "
                    f"buffer shape {tuple(self.prompt_embeddings.shape)}"
                )
            self.prompt_embeddings.copy_(embeddings)
            if not hasattr(self, "prompt_residuals") or self.prompt_residuals is None:
                self.prompt_residuals = nn.Parameter(torch.zeros_like(embeddings))
            elif self.prompt_residuals.shape != embeddings.shape:
                raise ValueError(
                    f"Residual shape {tuple(self.prompt_residuals.shape)} != "
                    f"embedding shape {tuple(embeddings.shape)}"
                )
        self.prompts_initialized = True

    def encode_residual_prompt_embeddings(self, prompt_groups):
        """Live encode (soft prompts trainable) then apply per-prompt Delta."""
        if not hasattr(self, "prompt_residuals") or self.prompt_residuals is None:
            raise RuntimeError("Call init_prompt_embeddings() before using residual prompts.")
        base = self.encode_textprompt_groups(prompt_groups)
        base = F.normalize(base.float(), dim=-1)
        if base.shape != self.prompt_residuals.shape:
            raise ValueError(
                f"Live embedding shape {tuple(base.shape)} != "
                f"residual shape {tuple(self.prompt_residuals.shape)}"
            )
        return F.normalize(base + self.prompt_residuals, dim=-1)

    def get_residual_prompt_embeddings(self):
        """Reference path using cached snapshot + Delta (for stats only)."""
        if "prompt_embeddings" not in self._buffers or not hasattr(self, "prompt_residuals"):
            raise RuntimeError("Call init_prompt_embeddings() before using residual prompts.")
        return F.normalize(self.prompt_embeddings + self.prompt_residuals, dim=-1)

    def residual_l2_loss(self):
        if not hasattr(self, "prompt_residuals") or self.prompt_residuals is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return (self.prompt_residuals.float() ** 2).sum(dim=-1).mean()

    @torch.no_grad()
    def residual_stats(self):
        if (
            "prompt_embeddings" not in self._buffers
            or not hasattr(self, "prompt_residuals")
            or self.prompt_residuals is None
        ):
            return {
                "mean_norm": 0.0,
                "max_norm": 0.0,
                "mean_cosine_to_e": 1.0,
                "min_cosine_to_e": 1.0,
            }
        norms = self.prompt_residuals.float().norm(dim=-1)
        text = self.get_residual_prompt_embeddings().float()
        cosine = (text * self.prompt_embeddings.float()).sum(dim=-1)
        return {
            "mean_norm": float(norms.mean().item()),
            "max_norm": float(norms.max().item()),
            "mean_cosine_to_e": float(cosine.mean().item()),
            "min_cosine_to_e": float(cosine.min().item()),
        }

    def forward(
        self,
        visual,
        padding_mask,
        text,
        lengths,
        return_visual=False,
        return_prompt_logits=False,
        prompt_weights=None,
    ):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_grouped_ori = self.encode_residual_prompt_embeddings(text)
        text_features_grouped_ori = text_features_grouped_ori.to(visual_features.dtype)

        text_features_ori = self.aggregate_group_features(text_features_grouped_ori, prompt_weights)

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        visual_attn = visual_attn.unsqueeze(2).expand(
            visual_attn.shape[0],
            text_features_grouped_ori.shape[0],
            text_features_grouped_ori.shape[1],
            visual_attn.shape[-1],
        )

        text_features = text_features_grouped_ori.unsqueeze(0).expand(
            visual_attn.shape[0],
            text_features_grouped_ori.shape[0],
            text_features_grouped_ori.shape[1],
            text_features_grouped_ori.shape[2],
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        prompt_logits = torch.einsum(
            "btd,bckd->btck",
            visual_features_norm,
            text_features_norm.type(visual_features_norm.dtype),
        ) / 0.07
        if prompt_weights is None:
            logits2 = torch.logsumexp(prompt_logits, dim=-1) - math.log(prompt_logits.shape[-1])
        else:
            weights = prompt_weights.to(prompt_logits.device, dtype=prompt_logits.dtype)
            if weights.shape != prompt_logits.shape[2:4]:
                raise ValueError(
                    f"prompt_weights shape {tuple(weights.shape)} does not match prompt logits "
                    f"class/prompt shape {tuple(prompt_logits.shape[2:4])}"
                )
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            logits2 = torch.logsumexp(
                prompt_logits + weights.clamp_min(1e-8).log().view(1, 1, *weights.shape),
                dim=-1,
            )

        outputs = [text_features_ori, logits1, logits2]
        if return_prompt_logits:
            outputs.append(prompt_logits)
        if return_visual:
            outputs.append(visual_features)
        return tuple(outputs)
