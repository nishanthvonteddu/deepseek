# deepseek/mlha.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import precompute_rope_freqs, apply_rope


class DeepSeekMLHAAttention(nn.Module):
    """
    Multi-Latent Head Attention (MLHA) / MLA-style compressed KV attention.

    Design:
      - Q is standard:  hidden -> (num_heads * head_dim)
      - KV is compressed to a latent per token: hidden -> latent_dim
      - K and V are expanded from latent: latent -> (num_kv_heads * head_dim)
      - Apply RoPE to Q and expanded K
      - GQA: repeat KV heads to match Q heads
      - Causal mask always; optional additive attn_mask supported
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_key_value_heads"

        self.head_dim = self.hidden_size // self.num_heads
        self.q_dim = self.num_heads * self.head_dim  # == hidden_size
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Latent dimension (compressed KV)
        cr = getattr(cfg, "mlha_compression_ratio", 8)
        assert cr > 0, "mlha_compression_ratio must be > 0"
        self.latent_dim = self.hidden_size // cr
        assert self.latent_dim > 0, "latent_dim must be > 0 (check compression_ratio)"

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=False)

        # Compress hidden -> latent
        self.kv_latent_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        # Expand latent -> K/V
        self.k_from_latent = nn.Linear(self.latent_dim, self.kv_dim, bias=False)
        self.v_from_latent = nn.Linear(self.latent_dim, self.kv_dim, bias=False)

        # Output proj
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

        # RoPE caches
        cos, sin = precompute_rope_freqs(
            dim=self.head_dim,
            max_position=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads for GQA.
        x: (B, kv_heads, T, head_dim) -> (B, heads, T, head_dim)
        """
        if self.num_kv_heads == self.num_heads:
            return x
        repeat_factor = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat_factor, dim=1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, hidden)
        attn_mask: optional additive mask broadcastable to (B, heads, T, T)
                  where masked positions are large negative (e.g., -1e9)
        """
        B, T, _ = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        # Q
        q = self.q_proj(x)  # (B, T, q_dim)

        # Latent KV
        kv_latent = self.kv_latent_proj(x)  # (B, T, latent_dim)

        # Expand to K/V
        k = self.k_from_latent(kv_latent)  # (B, T, kv_dim)
        v = self.v_from_latent(kv_latent)  # (B, T, kv_dim)

        # Reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # (B, H, T, D)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (B, KvH, T, D)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (B, KvH, T, D)

        # RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin, position_ids)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_ids)

        # GQA repeat
        k = self._repeat_kv(k)  # (B, H, T, D)
        v = self._repeat_kv(v)  # (B, H, T, D)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        attn_scores = attn_scores * (1.0 / math.sqrt(self.head_dim))

        # Causal mask always
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        # Optional additional mask (e.g., padding)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)  # (B, H, T, D)

        # Back to (B, T, hidden)
        out = out.transpose(1, 2).contiguous().view(B, T, self.q_dim)
        out = self.o_proj(out)
        return out