"""Phase 9 -- Inference: ask the fine-tuned model questions.

Real-world equivalent: serving the aligned model. We wrap the fine-tuned weights
with the SAME instruction template used in training (common/chat_format.py),
feed a question, and decode the response up to the <|endoftext|> stop token.

Usage:
  .venv/bin/python phase9_chat/run.py "What is the moral of 'The Boy Who Cried Wolf'?"
  .venv/bin/python phase9_chat/run.py            # interactive REPL
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import TOKENIZER_JSON, FINETUNED_CKPT, cfg  # noqa: E402
from common.bpe_tokenizer import BPETokenizer  # noqa: E402
from common.model import build_model  # noqa: E402
from common.chat_format import build_prompt  # noqa: E402


def load_model():
    ckpt = torch.load(FINETUNED_CKPT, map_location=cfg.device)
    for k, v in ckpt["model_cfg"].items():
        setattr(cfg.model, k, v)
    model = build_model(cfg.model, device=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def answer(model, tok, question: str) -> str:
    prompt = build_prompt(question)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=cfg.device)
    eot = tok.special_tokens["<|endoftext|>"]
    out = model.generate(ids, max_new_tokens=120, temperature=0.7, top_k=40, stop_id=eot)
    # keep only the newly generated answer tokens
    gen = out[0, ids.shape[1]:].tolist()
    text = tok.decode(gen)
    return text.split("<|endoftext|>")[0].strip()


def main() -> None:
    torch.manual_seed(cfg.seed)
    tok = BPETokenizer.load(TOKENIZER_JSON)
    model, ckpt = load_model()
    print(f"Loaded finetuned.pt (iter {ckpt['iter']}, val loss {ckpt['val_loss']:.3f}).\n")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"Q: {q}\nA: {answer(model, tok, q)}")
        return

    print("Interactive chat. Type a question (or 'quit').")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", ""}:
            break
        print(f"Model: {answer(model, tok, q)}")
    print("\nPhase 9 complete.")


if __name__ == "__main__":
    main()
