# deepseek/moe.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """
    LLaMA-style SwiGLU MLP expert:
      down( silu(gate(x)) * up(x) )
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TopKRouter(nn.Module):
    """
    Router: hidden -> logits(num_experts), softmax -> probs, take top-k.

    Returns:
      topk_idx: (B*T, k)
      topk_prob: (B*T, k)  (normalized across top-k)
      probs: (B*T, E) full distribution (for aux loss)
      z_loss: scalar (optional stabilization)
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, z_loss_coef: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.z_loss_coef = float(z_loss_coef)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x_flat: torch.Tensor):
        # x_flat: (N, hidden)
        logits = self.router(x_flat)  # (N, E)
        probs = F.softmax(logits, dim=-1)  # (N, E)

        # top-k selection
        topk_prob, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # (N, k), (N, k)

        # Normalize top-k probs to sum to 1 (common in MoE implementations)
        topk_prob = topk_prob / (topk_prob.sum(dim=-1, keepdim=True) + 1e-9)

        # Router z-loss (encourages small logits, stabilizes training)
        z_loss = None
        if self.z_loss_coef > 0.0:
            z = torch.logsumexp(logits, dim=-1)  # (N,)
            z_loss = self.z_loss_coef * (z * z).mean()

        return topk_idx, topk_prob, probs, z_loss


def load_balance_loss(probs: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Switch-style load balancing loss (no token dropping).
    - probs: (N, E) router softmax probabilities
    - topk_idx: (N, k) selected experts per token

    Computes:
      importance[e] = sum_n probs[n,e]
      load[e] = count of tokens routed to expert e (using top-1 assignment from top-k)
    """
    N, E = probs.shape
    assert E == num_experts

    # Importance: sum of probabilities per expert
    importance = probs.sum(dim=0)  # (E,)

    # Load: count of tokens assigned to each expert (use top-1 route for load metric)
    top1 = topk_idx[:, 0]  # (N,)
    load = torch.bincount(top1, minlength=E).float()  # (E,)

    # Normalize
    importance = importance / (importance.sum() + 1e-9)
    load = load / (load.sum() + 1e-9)

    # Switch loss: E * sum_e importance[e] * load[e]
    return (E * (importance * load).sum())


class MoEFeedForward(nn.Module):
    """
    MoE FFN with:
      - num_shared_experts always-on (we implement as one shared expert by default)
      - num_experts routed experts (top-k routing)
      - loss-less: no capacity limit, no token dropping
      - aux losses: load-balance + optional z-loss

    Forward returns:
      y: (B, T, hidden)
      aux_loss: scalar tensor
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size

        self.num_experts = int(getattr(cfg, "num_experts", 8))
        self.num_shared_experts = int(getattr(cfg, "num_shared_experts", 1))
        self.top_k = int(getattr(cfg, "top_k_experts", 2))

        assert self.num_experts > 0
        assert self.top_k > 0 and self.top_k <= self.num_experts
        assert self.num_shared_experts == 1, "This implementation supports exactly 1 shared expert for now."

        self.moe_aux_loss_coef = float(getattr(cfg, "moe_aux_loss_coef", 0.01))
        self.router_z_loss_coef = float(getattr(cfg, "router_z_loss_coef", 0.001))

        # Shared expert (always on) — can be warm-started from SmolLM2 MLP
        self.shared_expert = SwiGLUExpert(self.hidden_size, self.intermediate_size)

        # Routed experts
        self.experts = nn.ModuleList(
            [SwiGLUExpert(self.hidden_size, self.intermediate_size) for _ in range(self.num_experts)]
        )

        # Router
        self.router = TopKRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            z_loss_coef=self.router_z_loss_coef,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, hidden)
        """
        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)  # (N, H)
        N = x_flat.shape[0]

        # Shared path
        shared_out = self.shared_expert(x_flat)  # (N, H)

        # Route
        topk_idx, topk_prob, probs, z_loss = self.router(x_flat)  # idx/prob: (N,k)

        # Compute routed experts output loss-less (no dropping):
        # We compute expert outputs only for tokens assigned to that expert (saves compute).
        routed_out = torch.zeros_like(shared_out)  # (N, H)

        # For each k slot, dispatch tokens to their selected experts
        # Note: a token may appear multiple times across k with different experts.
        for slot in range(self.top_k):
            idx_e = topk_idx[:, slot]        # (N,)
            w_e = topk_prob[:, slot].unsqueeze(-1)  # (N,1)

            # Process per expert
            for e in range(self.num_experts):
                mask = (idx_e == e)
                if not mask.any():
                    continue
                xe = x_flat[mask]                 # (Ne, H)
                ye = self.experts[e](xe)          # (Ne, H)
                routed_out[mask] += ye * w_e[mask]

        # Combine shared + routed
        y = shared_out + routed_out
        y = y.reshape(B, T, H)

        # Aux load-balance loss
        lb = load_balance_loss(probs, topk_idx, self.num_experts)  # scalar
        aux = self.moe_aux_loss_coef * lb

        if z_loss is not None:
            aux = aux + z_loss

        return y, aux