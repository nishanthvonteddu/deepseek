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
from config import DeepSeekConfig
from modeling_deepseek import DeepSeekForCausalLM

# ============================================================
# Enable TF32 (FREE SPEED)
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Logging
# ============================================================
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

# ============================================================
# Checkpoint helpers
# ============================================================
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

# ============================================================
# Warm start from SmolLM2
# ============================================================
def load_pretrained_smol(model, model_dir):
    path = os.path.join(model_dir, "model.safetensors")
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info(f"Warm start loaded from {path}")
    logging.info(f"Missing keys (expected): {len(missing)}")
    logging.info(f"Unexpected keys: {len(unexpected)}")

# ============================================================
# Generation (monitoring)
# ============================================================
@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, max_new_tokens=80):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return text

# ============================================================
# Main
# ============================================================
def main():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Config ----------------
    cfg = DeepSeekConfig()
    cfg.hidden_size = 576
    cfg.num_hidden_layers = 30
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 1
    cfg.intermediate_size = 1536
    cfg.max_position_embeddings = 2048
    cfg.mlha_compression_ratio = 8
    cfg.num_experts = 8
    cfg.num_shared_experts = 1
    cfg.top_k_experts = 2
    cfg.moe_aux_loss_coef = 0.01

    # ---------------- Training params ----------------
    seq_len = 128
    batch_size = 8
    max_steps = 10000
    log_interval = 10
    gen_interval = 500
    ckpt_interval = 500
    warmup_steps = 500
    base_lr = 3e-4

    gen_prompt = "Summary: A girl and her dog went on an adventure."

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ec2-user/projects/SmolLM2/smollm2_135m_instruct",
        use_fast=True,
    )
    cfg.vocab_size = len(tokenizer)

    # ---------------- Dataset ----------------
    hf_ds = load_dataset(
        "roneneldan/TinyStoriesInstruct",
        split="train",
        streaming=True,
    )

    train_ds = StreamingTokenDataset(
        hf_ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = DeepSeekForCausalLM(cfg).to(device)

    resume_path = "checkpoints/step_3500.pt"
    if os.path.exists(resume_path):
        logging.info(f"Resuming from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location="cpu"))
        start_step = 3500
    else:
        start_step = 0

    if os.path.isdir("/home/ec2-user/projects/SmolLM2/smollm2_135m_instruct"):
        load_pretrained_smol(
            model,
            "/home/ec2-user/projects/SmolLM2/smollm2_135m_instruct",
        )

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

    scaler = torch.amp.GradScaler("cuda")

    # ---------------- Checkpoints ----------------
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

    logging.info("Setup complete.")
    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Seq len: {seq_len}")
    logging.info(f"Max steps: {max_steps}")

    start_time = time.time()

    # ---------------- Training loop ----------------
    step = start_step

    for batch in train_loader:
        if step >= max_steps:
            break
        input_ids, labels = [x.to(device) for x in batch]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, aux_loss = model(input_ids)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss = ce_loss + aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            logging.info(
                f"step={step:05d} "
                f"loss={loss.item():.4f} "
                f"ce={ce_loss.item():.4f} "
                f"aux={aux_loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"time={elapsed:.1f}s"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(best_ckpt, model, optimizer, scheduler, step, best_loss)
            logging.info(f"New best model saved (loss={best_loss:.4f})")

        if step % ckpt_interval == 0:
            save_checkpoint(latest_ckpt, model, optimizer, scheduler, step, best_loss)
            logging.info(f"Checkpoint saved at step {step}")

        if step % gen_interval == 0:
            logging.info("=== Generation sample ===")
            logging.info(generate_sample(model, tokenizer, device, gen_prompt))

    logging.info("Training finished.")

if __name__ == "__main__":
    main()
