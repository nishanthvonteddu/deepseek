# deepseek/config.py

from config import SmolLM2Config


class DeepSeekConfig(SmolLM2Config):
    """
    Configuration for DeepSeek-style SmolLM2 variant.

    Inherits all base SmolLM2 parameters and adds:
    - MLHA (Multi-Latent Head Attention)
    - MoE (Mixture of Experts)
    """

    # -------------------------
    # MLHA (Attention)
    # -------------------------
    mlha_compression_ratio: int = 8

    # -------------------------
    # MoE (Feed-Forward)
    # -------------------------
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k_experts: int = 2

    # -------------------------
    # MoE Loss coefficients
    # -------------------------
    moe_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # -------------------------
    # Identification (useful for export)
    # -------------------------
    model_type: str = "deepseek-smollm2"