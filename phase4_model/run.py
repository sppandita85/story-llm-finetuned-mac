"""Phase 4 -- Build the GPT model and sanity-check the architecture.

Real-world equivalent: defining the model and verifying a forward/backward pass
on a single batch before committing thousands of GPU-hours. We instantiate the
decoder-only transformer (common/model.py) sized to our tokenizer's vocab, print
the parameter count and per-component breakdown, then run one forward pass and
one backward pass to prove gradients flow.

This phase trains nothing -- it is the "does the model even run?" gate.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import TOKENIZER_JSON, cfg  # noqa: E402
from common.bpe_tokenizer import BPETokenizer  # noqa: E402
from common.model import build_model  # noqa: E402


def main() -> None:
    torch.manual_seed(cfg.seed)

    # Size the model's vocab to match the real tokenizer.
    tok = BPETokenizer.load(TOKENIZER_JSON)
    cfg.model.vocab_size = tok.n_vocab
    model = build_model(cfg.model, device=cfg.device)

    print("--- model configuration ---")
    for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout"):
        print(f"  {k:11s}: {getattr(cfg.model, k)}")

    total = model.num_params()
    embed = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    print("\n--- parameter count ---")
    print(f"  total parameters     : {total:,}")
    print(f"  embeddings (tok+pos) : {embed:,}")
    print(f"  transformer + head   : {total - embed:,}")

    # One forward pass on a random batch, then one backward pass.
    B, T = 4, cfg.model.block_size
    idx = torch.randint(0, cfg.model.vocab_size, (B, T), device=cfg.device)
    targets = torch.randint(0, cfg.model.vocab_size, (B, T), device=cfg.device)
    logits, loss = model(idx, targets)
    loss.backward()

    grad_ok = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    # A randomly initialized model over V classes should start near ln(V).
    import math
    expected = math.log(cfg.model.vocab_size)
    print("\n--- forward / backward sanity check ---")
    print(f"  input  shape : {tuple(idx.shape)}")
    print(f"  logits shape : {tuple(logits.shape)}  (batch, time, vocab)")
    print(f"  initial loss : {loss.item():.3f}  (random-init expectation ~ln(V) = {expected:.3f})")
    print(f"  gradients flow to all params : {grad_ok}")
    assert logits.shape == (B, T, cfg.model.vocab_size)
    assert grad_ok
    print("Phase 4 complete. Architecture is ready to train.")


if __name__ == "__main__":
    main()
