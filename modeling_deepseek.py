# deepseek/modeling_deepseek.py

import torch
import torch.nn as nn

from .config import DeepSeekConfig
from .rmsnorm import RMSNorm
from .mlha import DeepSeekMLHAAttention
from .moe import MoEFeedForward


# -------------------------------------------------
# Decoder Layer
# -------------------------------------------------
class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, cfg: DeepSeekConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DeepSeekMLHAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.moe = MoEFeedForward(cfg)

    def forward(self, x: torch.Tensor, attn_mask=None):
        # Attention block
        x = x + self.self_attn(self.input_layernorm(x), attn_mask=attn_mask)

        # MoE block
        moe_out, aux_loss = self.moe(self.post_attention_layernorm(x))
        x = x + moe_out

        return x, aux_loss


# -------------------------------------------------
# Base Model
# -------------------------------------------------
class DeepSeekModel(nn.Module):
    def __init__(self, cfg: DeepSeekConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attn_mask=None):
        x = self.embed_tokens(input_ids)

        total_aux = x.new_zeros(())

        for layer in self.layers:
            x, aux = layer(x, attn_mask=attn_mask)
            total_aux = total_aux + aux

        x = self.norm(x)
        return x, total_aux


# -------------------------------------------------
# Causal LM
# -------------------------------------------------
class DeepSeekForCausalLM(nn.Module):
    def __init__(self, cfg: DeepSeekConfig):
        super().__init__()
        self.cfg = cfg
        self.model = DeepSeekModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attn_mask=None):
        hidden, aux_loss = self.model(input_ids, attn_mask=attn_mask)
        logits = self.lm_head(hidden)
        return logits, aux_loss