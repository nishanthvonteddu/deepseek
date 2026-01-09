import os
import time
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from safetensors.torch import load_file

from dataset import StreamingTokenDataset

from deepseek.config import DeepSeekConfig
from deepseek.modeling_deepseek import DeepSeekForCausalLM


# ------------------------------------------------
# Logging
# ------------------------------------------------
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/train.log"),
            logging.StreamHandler(),
        ],
    )


# ------------------------------------------------
# Checkpoints
# ------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, step, best_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_loss"]


# ------------------------------------------------
# Pretrained weights (warm start from SmolLM2)
# ------------------------------------------------
def load_pretrained_smol(model, model_dir):
    """
    Loads SmolLM2 safetensors into DeepSeek model with strict=False.
    This will:
      - load matching weights (embeddings / lm_head / norms etc. if names match)
      - leave new DeepSeek params (router/experts/mlha latent/expand) randomly init
    """
    path = os.path.join(model_dir, "model.safetensors")
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info(f"Warm start loaded from {path}")
    logging.info(f"Missing keys (expected for new DeepSeek params): {len(missing)}")
    logging.info(f"Unexpected keys: {len(unexpected)}")


# ------------------------------------------------
# Generation (monitoring only)
# ------------------------------------------------
@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, max_new_tokens=80):
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        logits, _aux = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return text


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Config
    # -------------------------
    cfg = DeepSeekConfig()

    # Your target params (you can hard-set them here to avoid surprises)
    # vocab_size stays same as SmolLM2 instruct tokenizer vocab
    # (we’ll set cfg.vocab_size after tokenizer loads)
    cfg.hidden_size = 768
    cfg.num_hidden_layers = 30
    cfg.num_attention_heads = 8
    # Set kv heads to a valid GQA factor. If your SmolLM2Config uses a different default,
    # keep it consistent. 4 is common when heads=8.
    cfg.num_key_value_heads = getattr(cfg, "num_key_value_heads", 4) or 4
    cfg.intermediate_size = 1536
    cfg.max_position_embeddings = 2048

    # MLHA / MoE params
    cfg.mlha_compression_ratio = 8
    cfg.num_experts = 8
    cfg.num_shared_experts = 1
    cfg.top_k_experts = 2
    cfg.moe_aux_loss_coef = 0.01
    cfg.router_z_loss_coef = 0.001

    # Training params
    seq_len = 512
    batch_size = 2
    max_steps = 10  # you asked 10k+ for fun
    log_interval = 10
    gen_interval = 50
    ckpt_interval = 500
    warmup_steps = 20
    base_lr = 3e-4

    gen_prompt = "Summary: A girl and her dog went on an adventure."

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "smollm2_135m_instruct",
        use_fast=True,
    )

    # Make sure vocab size matches tokenizer
    cfg.vocab_size = len(tokenizer)

    # -------------------------
    # Dataset
    # -------------------------
    hf_ds = load_dataset(
        "roneneldan/TinyStoriesInstruct",
        split="train",
        streaming=True,
    )

    train_ds = StreamingTokenDataset(
        hf_dataset=hf_ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
    )

    # -------------------------
    # Model
    # -------------------------
    model = DeepSeekForCausalLM(cfg).to(device)

    # Warm start (SmolLM2 instruct weights)
    # Keep this path the same as your existing repo folder
    if os.path.isdir("smollm2_135m_instruct"):
        load_pretrained_smol(model, "smollm2_135m_instruct")
    else:
        logging.warning("smollm2_135m_instruct folder not found; training from scratch.")

    model.train()

    logging.info("Setup complete.")
    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Seq len: {seq_len}")
    logging.info(f"Max steps: {max_steps}")
    logging.info(f"Vocab size: {cfg.vocab_size}")
    logging.info(
        f"Model: layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}, "
        f"kv_heads={cfg.num_key_value_heads}, mlha_cr={cfg.mlha_compression_ratio}, "
        f"experts={cfg.num_experts}, topk={cfg.top_k_experts}"
    )

    # -------------------------
    # Optimizer + Scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # -------------------------
    # Checkpoint paths
    # -------------------------
    os.makedirs("checkpoints", exist_ok=True)
    latest_ckpt = "checkpoints/latest.pt"
    best_ckpt = "checkpoints/best.pt"

    step = 0
    best_loss = float("inf")

    if os.path.exists(latest_ckpt):
        logging.info(f"Resuming from {latest_ckpt}")
        step, best_loss = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler
        )
        logging.info(f"Resumed at step={step}, best_loss={best_loss:.4f}")

    # -------------------------
    # Training loop
    # -------------------------
    start_time = time.time()

    for batch in train_loader:
        if step >= max_steps:
            break

        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, aux_loss = model(input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss = ce_loss + aux_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        loss_val = loss.detach().item()
        ce_val = ce_loss.detach().item()
        aux_val = aux_loss.detach().item()
        lr = scheduler.get_last_lr()[0]

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            logging.info(
                f"step={step:04d} "
                f"loss={loss_val:.4f} "
                f"ce={ce_val:.4f} "
                f"aux={aux_val:.4f} "
                f"lr={lr:.2e} "
                f"time={elapsed:.1f}s"
            )

        if step % gen_interval == 0:
            logging.info("=== Generation sample ===")
            text = generate_sample(model, tokenizer, device, gen_prompt)
            logging.info(text)
            logging.info("=== End sample ===")

        if step > 0 and step % ckpt_interval == 0:
            save_checkpoint(
                latest_ckpt,
                model,
                optimizer,
                scheduler,
                step,
                best_loss,
            )
            logging.info(f"Saved latest checkpoint at step {step}")

        if step > 0 and loss_val < best_loss:
            best_loss = loss_val
            save_checkpoint(
                best_ckpt,
                model,
                optimizer,
                scheduler,
                step,
                best_loss,
            )
            logging.info(f"New best model saved (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()