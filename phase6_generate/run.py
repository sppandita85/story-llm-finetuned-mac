"""Phase 6 -- Evaluate the pretrained base model by generating text.

Real-world equivalent: after pretraining you probe the base model with prompts
to qualitatively check what it learned (before any instruction tuning). A base
LLM is a text *continuer*, not yet a chat assistant -- it just keeps writing in
the style of its training data. We load pretrained.pt and continue a few prompts.

Usage:
  .venv/bin/python phase6_generate/run.py                # built-in prompts
  .venv/bin/python phase6_generate/run.py "A clever fox"  # custom prompt
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import TOKENIZER_JSON, PRETRAINED_CKPT, cfg  # noqa: E402
from common.bpe_tokenizer import BPETokenizer  # noqa: E402
from common.model import build_model  # noqa: E402


def load_pretrained():
    ckpt = torch.load(PRETRAINED_CKPT, map_location=cfg.device)
    for k, v in ckpt["model_cfg"].items():
        setattr(cfg.model, k, v)
    model = build_model(cfg.model, device=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main() -> None:
    torch.manual_seed(cfg.seed)
    tok = BPETokenizer.load(TOKENIZER_JSON)
    model, ckpt = load_pretrained()
    print(f"Loaded pretrained.pt (iter {ckpt['iter']}, val loss {ckpt['val_loss']:.3f}).\n")

    if len(sys.argv) > 1:
        prompts = [" ".join(sys.argv[1:])]
    else:
        prompts = [
            "Once upon a time",
            "The lion",
            "Moral:",
            "A greedy",
        ]

    for p in prompts:
        ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=cfg.device)
        out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40)
        text = tok.decode(out[0].tolist())
        # cut at the first document boundary for readability
        text = text.split("<|endoftext|>")[0].strip()
        print(f"PROMPT: {p!r}")
        print(f"  -> {text}\n")

    print("Phase 6 complete.")


if __name__ == "__main__":
    main()
