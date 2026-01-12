# config.py

from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    # --- tokenizer / vocab ---
    vocab_size: int = 49152

    # --- model size ---
    hidden_size: int = 576
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    intermediate_size: int = 1536

    # --- sequence / RoPE ---
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # --- normalization ---
    rms_norm_eps: float = 1e-5

    # --- MLHA ---
    mlha_compression_ratio: int = 8

    # --- MoE ---
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k_experts: int = 2
    moe_aux_loss_coef: float = 0.01

    # --- misc ---
    tie_word_embeddings: bool = True
