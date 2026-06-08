"""Phase 8 -- Supervised fine-tuning (SFT) on the instruction dataset.

Real-world equivalent: instruction tuning / SFT. We take the pretrained base
model and continue training it on (prompt, response) pairs, but crucially we
mask the loss so it is computed ONLY on the response tokens. This teaches the
model to follow instructions while reusing everything it learned in pretraining.
Same recipe as production SFT, just smaller.

Loads:  phase5_pretrain/pretrained.pt + phase7 qa_train/qa_val
Output: finetuned.pt -- the instruction-following model
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (  # noqa: E402
    TOKENIZER_JSON, PRETRAINED_CKPT, FINETUNED_CKPT,
    QA_TRAIN_JSONL, QA_VAL_JSONL, cfg,
)
from common.bpe_tokenizer import BPETokenizer  # noqa: E402
from common.model import build_model  # noqa: E402
from common.chat_format import encode_example  # noqa: E402


def load_examples(path: str, tok, block_size: int):
    """Encode each jsonl row into (ids, mask), truncated to block_size."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids, mask = encode_example(tok, r["instruction"], r.get("input", ""), r["output"])
            ids, mask = ids[:block_size], mask[:block_size]
            if sum(mask) == 0:
                continue  # no answer tokens left after truncation
            examples.append((ids, mask))
    return examples


def make_batch(examples, idxs, pad_id, device):
    """Pad a set of variable-length examples into (x, y, loss_mask) tensors."""
    seqs = [torch.tensor(examples[i][0], dtype=torch.long) for i in idxs]
    masks = [torch.tensor(examples[i][1], dtype=torch.long) for i in idxs]
    seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    # next-token setup: predict token t+1 from token t
    x = seqs[:, :-1]
    y = seqs[:, 1:].clone()
    loss_mask = masks[:, 1:]
    y[loss_mask == 0] = -1            # ignore_index in cross-entropy
    return x.to(device), y.to(device), loss_mask.to(device)


def lr_at(it, t):
    if it < t.ft_warmup_iters:
        return t.ft_learning_rate * (it + 1) / t.ft_warmup_iters
    if it >= t.ft_max_iters:
        return t.ft_min_lr
    ratio = (it - t.ft_warmup_iters) / (t.ft_max_iters - t.ft_warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return t.ft_min_lr + coeff * (t.ft_learning_rate - t.ft_min_lr)


@torch.no_grad()
def eval_loss(model, examples, pad_id, batch_size, device, n_batches=10):
    model.eval()
    losses = []
    for b in range(n_batches):
        idxs = torch.randint(len(examples), (min(batch_size, len(examples)),)).tolist()
        x, y, m = make_batch(examples, idxs, pad_id, device)
        _, loss = model(x, y, loss_mask=m)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    torch.manual_seed(cfg.seed)
    t = cfg.train
    device = cfg.device
    tok = BPETokenizer.load(TOKENIZER_JSON)
    pad_id = tok.special_tokens["<|pad|>"]

    # Load the PRETRAINED base model and continue training from it.
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    for k, v in ckpt["model_cfg"].items():
        setattr(cfg.model, k, v)
    model = build_model(cfg.model, device=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded pretrained base (val loss {ckpt['val_loss']:.3f}). Starting SFT.")

    train_ex = load_examples(QA_TRAIN_JSONL, tok, cfg.model.block_size)
    val_ex = load_examples(QA_VAL_JSONL, tok, cfg.model.block_size)
    print(f"Fine-tuning examples: {len(train_ex)} train / {len(val_ex)} val")

    # AdamW; lower LR than pretraining so we don't wash out pretrained knowledge.
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.ft_learning_rate,
                                  betas=(t.beta1, t.beta2), weight_decay=t.weight_decay)

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for it in range(t.ft_max_iters + 1):
        lr = lr_at(it, t)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % t.ft_eval_interval == 0 or it == t.ft_max_iters:
            vl = eval_loss(model, val_ex, pad_id, t.ft_batch_size, device)
            tl = eval_loss(model, train_ex, pad_id, t.ft_batch_size, device)
            print(f"iter {it:4d} | train {tl:.3f} | val {vl:.3f} | lr {lr:.2e} "
                  f"| {time.time() - t0:.0f}s")
            if vl < best_val:
                best_val = vl
                torch.save({
                    "model_state": model.state_dict(),
                    "model_cfg": vars(cfg.model),
                    "val_loss": best_val,
                    "iter": it,
                }, FINETUNED_CKPT)

        if it == t.ft_max_iters:
            break

        idxs = torch.randint(len(train_ex), (t.ft_batch_size,)).tolist()
        x, y, m = make_batch(train_ex, idxs, pad_id, device)
        _, loss = model(x, y, loss_mask=m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()

    print(f"\nBest val loss: {best_val:.3f}")
    print(f"Saved fine-tuned model -> {os.path.relpath(FINETUNED_CKPT)}")
    print("Phase 8 complete. The model now follows instructions.")


if __name__ == "__main__":
    main()
