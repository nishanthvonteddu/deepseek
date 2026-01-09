# deepseek/rope.py

import torch


def precompute_rope_freqs(
    dim: int,
    max_position: int,
    theta: float,
):
    """
    Precompute rotary embeddings (cos, sin).

    Returns:
        cos, sin: tensors of shape (max_position, dim)
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_position).float()
    freqs = torch.outer(positions, inv_freq)  # (max_position, dim/2)

    # Duplicate for HF-style RoPE
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_position, dim)

    return torch.cos(emb), torch.sin(emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper for RoPE rotation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings.

    Args:
        x: (B, H, T, D)
        cos, sin: (max_seq_len, D)
        position_ids: (B, T)
    """
    cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, D)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)