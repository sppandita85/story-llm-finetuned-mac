"""Phase 2 -- Train a custom BPE tokenizer on the corpus.

Real-world equivalent: before any LLM is trained, a tokenizer is learned on a
representative sample of the data. It decides how raw text is chopped into the
integer tokens the model actually sees. We train the exact same Byte-Pair-
Encoding algorithm GPT uses (see common/bpe_tokenizer.py), just to a small vocab
because the corpus is tiny.

Outputs (this folder):
  tokenizer.json -- learned merges + special tokens (loadable everywhere)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import CORPUS_TXT, TOKENIZER_JSON, SPECIAL_TOKENS, cfg  # noqa: E402
from common.bpe_tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    with open(CORPUS_TXT, "r", encoding="utf-8") as f:
        text = f.read()

    target_vocab = cfg.tokenizer.vocab_size
    print(f"Training BPE tokenizer: target vocab = {target_vocab}, "
          f"special tokens = {SPECIAL_TOKENS}")

    tok = BPETokenizer()
    # Train on the raw text WITHOUT the special-token strings being merged: we
    # strip them so merges are learned over natural language only, then register
    # the specials separately.
    train_text = text
    for s in SPECIAL_TOKENS:
        train_text = train_text.replace(s, "\n")
    tok.train(train_text, vocab_size=target_vocab, special_tokens=SPECIAL_TOKENS)

    tok.save(TOKENIZER_JSON)
    print(f"\nFinal vocab size: {tok.n_vocab}")
    print(f"  byte tokens   : 256")
    print(f"  learned merges: {len(tok.merges)}")
    print(f"  special tokens: {len(tok.special_tokens)} -> {tok.special_tokens}")
    print(f"Saved -> {os.path.relpath(TOKENIZER_JSON)}")

    # Round-trip sanity check: decode(encode(x)) must reproduce x exactly.
    sample = "The tortoise kept walking, slowly and steadily. <|endoftext|>"
    ids = tok.encode(sample)
    back = tok.decode(ids)
    ratio = len(sample) / max(1, len(ids))
    print("\n--- round-trip check ---")
    print(f"  sample      : {sample!r}")
    print(f"  token ids   : {ids}")
    print(f"  decoded back: {back!r}")
    print(f"  lossless    : {back == sample}")
    print(f"  compression : {ratio:.2f} chars/token")
    assert back == sample, "Tokenizer round-trip is not lossless!"
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
