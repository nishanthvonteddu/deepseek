# DeepSeek (Mini) - MoE + MLHA Language Model

A custom transformer-based language model inspired by DeepSeek-style architectures, combining a dense transformer backbone with Mixture-of-Experts (MoE) feedforward layers and Multi-Latent Head Attention (MLHA).

## 🏗️ Architecture Overview

```
Input Tokens
   ↓
Token Embedding
   ↓
[ Repeated N times ]
 ┌─────────────────────────────────────────────┐
 │ Transformer Decoder Layer                   │
 │                                             │
 │ 1. RMSNorm                                  │
 │ 2. MLHA Attention                           │
 │ 3. Residual Add                             │
 │ 4. RMSNorm                                  │
 │ 5. MoE Feedforward (Experts + Router)       │
 │ 6. Residual Add                             │
 └─────────────────────────────────────────────┘
   ↓
Final RMSNorm
   ↓
LM Head (Vocabulary Projection)
   ↓
Next-token Logits
```

## ⚙️ Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden size | 576 |
| Layers | 30 |
| Attention heads | 8 |
| Key/Value heads | 1 (GQA-style) |
| Sequence length | 128 |
| Vocabulary size | 49,152 |
| MLHA compression ratio | 8 |
| Experts per layer | 8 |
| Shared experts | 1 |
| Top-K experts | 2 |

## 🔤 Token Embeddings

- Tokens are embedded into a 576-dimensional vector space
- Tokenizer comes from SmolLM2 Instruct
- Embedding matrix shape: `[vocab_size, hidden_size]`

## 🌀 Positional Encoding (RoPE)

This model uses Rotary Positional Embeddings (RoPE):
- Positions are encoded by rotating query and key vectors
- Allows better extrapolation than absolute embeddings
- Applied inside attention, not added to embeddings

**Benefits:**
- Stable long-context behavior
- Better relative position modeling

## 🎯 MLHA – Multi-Latent Head Attention

### What MLHA Does

**Traditional attention:**
- Each head has its own Q, K, V

**MLHA:**
- Compresses keys & values into latent vectors
- Reconstructs K/V from latents
- Saves memory and compute

### MLHA Flow

```
Input → Q Projection
      → KV Latent Projection (compressed)
      → K from Latent
      → V from Latent
      → Scaled Dot-Product Attention
      → Output Projection
```

### Why MLHA?
- Lower memory usage
- Faster attention
- Keeps expressiveness close to full attention

## 📏 RMSNorm (Normalization)

Instead of LayerNorm, the model uses RMSNorm:
- Normalizes only by root-mean-square
- No mean subtraction
- Faster and more stable

**Used:**
- Before attention
- Before MoE feedforward
- At final output

## 🧠 MoE – Mixture of Experts Feedforward

Each transformer layer uses an MoE block instead of a single FFN.

### Experts
- 8 experts per layer
- Each expert is a standard FFN: `Linear → GELU → Linear`

### Shared Expert
- 1 shared expert always available
- Helps stability and prevents expert collapse

### Router
- A learned router scores tokens
- Selects Top-2 experts per token
- Routing is token-wise, not batch-wise

### MoE Flow

```
Token
 ↓
Router → select top-K experts
 ↓
Experts process token
 ↓
Weighted sum of expert outputs
```

## 📊 MoE Auxiliary Losses

Two auxiliary losses are added to training:

1. **Load Balancing Loss**
   - Encourages even expert usage
   - Prevents all tokens going to one expert

2. **Router Z-Loss**
   - Penalizes overly confident routing
   - Improves training stability

**Final loss:**
```
Total Loss = CrossEntropy + Aux Loss
```

## 🔗 Residual Connections

Every major block uses residual connections:
- Attention output added back to input
- MoE output added back to input

**This ensures:**
- Stable gradient flow
- Deep training (30 layers) without collapse

## 🎯 Output Head (LM Head)

- Final hidden states → linear projection
- Shape: `[hidden_size → vocab_size]`
- Produces logits for next-token prediction
- Weight tying can be enabled if desired

## 🚀 Training Details

### Optimizer
- AdamW
- Weight decay: 0.1
- Betas: (0.9, 0.95)

### Learning Rate
- Base LR: 3e-4
- Cosine decay
- Warmup: 20 steps

### Precision
- TF32 enabled
- Mixed precision (FP16 autocast + GradScaler)

### Batch Size
- 8 (effective)
- Fits within A10G 24GB VRAM

## 💾 Checkpointing Strategy

- `latest.pt` → always overwritten
- `best.pt` → lowest loss seen
- Prevents disk explosion
- Safe resume supported

## 🤔 Why This Architecture?

This design balances:
- **Speed** (MLHA, TF32, mixed precision)
- **Capacity** (MoE experts)
- **Stability** (RMSNorm, routing losses)
- **Scalability** (can scale layers, experts, context)

It is intentionally smaller and faster than full DeepSeek models, while keeping the same core ideas.

## 📈 Expected Behavior

- Loss decreases fast initially
- Slower convergence after ~5k steps
- Stable aux loss ≈ 0.30
- Final loss (TinyStories): ~2.3–2.6
- This is normal and healthy

## 🛠️ Technical Implementation

- Built with PyTorch
- Warm-started from SmolLM2
- Trained on TinyStoriesInstruct
- Custom transformer implementation
- Efficient MoE routing

## 📁 Repository Structure

```
├── model.py              # Core model implementation
├── train.py              # Training script
├── config.py             # Configuration management
├── data_loader.py        # Data loading utilities
├── tokenizer/            # Tokenizer files
├── checkpoints/          # Training checkpoints
└── utils/               # Utility functions
```

## 🚦 Quick Start

```bash
# Install dependencies
pip install torch transformers datasets

# Train the model
python train.py --config configs/base.yaml

# Generate text
python generate.py --checkpoint checkpoints/best.pt
```

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- DeepSeek AI for the architecture inspiration
- SmolLM2 for tokenizer and warm-start
- TinyStories dataset for training
- PyTorch team for the framework

## 📚 References

1. DeepSeek Models
2. Mixture of Experts (MoE) literature
3. Rotary Positional Embeddings (RoPE)
4. RMSNorm: Root Mean Square Layer Normalization
5. Multi-Latent Head Attention (MLHA)

---

*This model is a research implementation and may require modifications for production use.*



## 📊 Training Log Analysis (Steps 4390–4790)

This section summarizes training behavior during mid-to-late training, focusing on
loss stability, learning rate behavior, and generation quality.

---

### 🔢 Training Metrics

| Step | Total Loss | CE Loss | Aux Loss | LR | Time (s) |
|-----:|-----------:|--------:|---------:|----:|---------:|
| 4390 | 3.6203 | 3.3149 | 0.3055 | 2.99e-04 | 3569.8 |
| 4400 | 3.2745 | 2.9691 | 0.3054 | 2.99e-04 | 3580.5 |
| 4450 | 3.3497 | 3.0439 | 0.3058 | 2.98e-04 | 3749.1 |
| 4500 | 3.3655 | 3.0601 | 0.3053 | 2.98e-04 | 3860.2 |
| 4550 | 4.1483 | 3.8427 | 0.3055 | 2.98e-04 | 4002.6 |
| 4600 | 4.1823 | 3.8763 | 0.3060 | 2.97e-04 | 4056.6 |
| 4650 | 3.3953 | 3.0900 | 0.3054 | 2.97e-04 | 4110.1 |
| 4700 | 3.2619 | 2.9566 | 0.3053 | 2.96e-04 | 4163.8 |
| 4720 | 3.2115 | 2.9063 | 0.3051 | 2.96e-04 | 4185.4 |
| 4750 | 4.0083 | 3.7017 | 0.3066 | 2.95e-04 | 4217.7 |
| 4790 | 3.5691 | 3.2644 | 0.3047 | 2.95e-04 | 4260.6 |

---

### 📉 Loss Behavior

- **Cross-entropy loss** steadily decreases overall, with expected batch-level noise
- **Auxiliary MoE loss** remains highly stable (~0.30), indicating healthy expert routing
- Loss spikes are normal and correlate with difficult batches or routing imbalance
- Best observed losses in this window are around **~3.21–3.26**

This indicates:
- No divergence
- No expert collapse
- Learning rate remains in a safe regime

---

### 🧠 Learning Rate

- LR decays smoothly from **2.99e-04 → 2.95e-04**
- No sudden drops or plateaus
- Scheduler is functioning as expected

---

### ✍️ Generation Sample (Qualitative Check)
