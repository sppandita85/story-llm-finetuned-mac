"""Phase 5 -- Pretraining: train the GPT to predict the next token.

This is THE core deliverable. The loop here is structurally identical to a real
trillion-token pretraining run -- only the scale differs:

  - AdamW optimizer with decoupled weight decay (no decay on 1-D params)
  - linear warmup then cosine decay of the learning rate
  - gradient clipping for stability
  - periodic train/val loss estimation on held-out data
  - sample text generation to watch the model learn
  - checkpointing the best model by validation loss

Random fixed-length windows are streamed from the Phase-3 token shards via a
memmap, exactly like nanoGPT / production data loaders.

Output (this folder):
  pretrained.pt -- the pretrained LLM checkpoint (model weights + config)
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (  # noqa: E402
    TRAIN_BIN, VAL_BIN, TOKENIZER_JSON, PRETRAINED_CKPT, cfg,
)
from common.bpe_tokenizer import BPETokenizer  # noqa: E402
from common.model import build_model  # noqa: E402


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    """Sample `batch_size` random contiguous windows -> (x, y) tensors."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device):
    """Average loss over `eval_iters` batches for each split (no grad)."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it: int, t) -> float:
    """Linear warmup then cosine decay down to min_lr."""
    if it < t.warmup_iters:
        return t.learning_rate * (it + 1) / t.warmup_iters
    if it >= t.max_iters:
        return t.min_lr
    ratio = (it - t.warmup_iters) / (t.max_iters - t.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return t.min_lr + coeff * (t.learning_rate - t.min_lr)


def configure_optimizer(model, t, device):
    """AdamW with weight decay only on >=2-D tensors (matrices), not biases/norms."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": t.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=t.learning_rate, betas=(t.beta1, t.beta2))


def main() -> None:
    torch.manual_seed(cfg.seed)
    t = cfg.train
    device = cfg.device

    tok = BPETokenizer.load(TOKENIZER_JSON)
    cfg.model.vocab_size = tok.n_vocab
    model = build_model(cfg.model, device=device)
    print(f"Pretraining a {model.num_params():,}-param GPT on {device} "
          f"(vocab {tok.n_vocab}, block {cfg.model.block_size}).")

    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    splits = {"train": train_data, "val": val_data}

    optimizer = configure_optimizer(model, t, device)
    prompt_ids = torch.tensor([tok.encode("Once upon a time")], dtype=torch.long, device=device)

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for it in range(t.max_iters + 1):
        # set the schedule's learning rate for this step
        lr = lr_at(it, t)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # periodic evaluation + checkpointing
        if it % t.eval_interval == 0 or it == t.max_iters:
            losses = estimate_loss(model, splits, cfg.model.block_size,
                                   t.batch_size, t.eval_iters, device)
            dt = time.time() - t0
            print(f"iter {it:4d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| lr {lr:.2e} | {dt:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "model_cfg": vars(cfg.model),
                    "val_loss": best_val,
                    "iter": it,
                }, PRETRAINED_CKPT)

        if it == t.max_iters:
            break

        # one optimization step
        x, y = get_batch(train_data, cfg.model.block_size, t.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()

        # show a generated sample now and then to watch learning progress
        if it > 0 and it % (t.eval_interval * 2) == 0:
            gen = model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
            print(f"    sample: {tok.decode(gen[0].tolist())!r}")

    print(f"\nBest val loss: {best_val:.3f}")
    print(f"Saved pretrained LLM -> {os.path.relpath(PRETRAINED_CKPT)}")
    print("Phase 5 complete. You now have a PRETRAINED base model.")


if __name__ == "__main__":
    main()
