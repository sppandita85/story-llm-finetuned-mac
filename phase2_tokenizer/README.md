# Phase 2 — Tokenizer Training (custom BPE)

**What it does:** Trains a Byte-Pair-Encoding tokenizer from scratch on
`phase1_data_prep/corpus.txt` (implementation in `common/bpe_tokenizer.py`),
registers the special tokens, saves `tokenizer.json`, and verifies that
`decode(encode(x)) == x` (lossless round-trip).

**Real-world equivalent:** Every LLM learns a tokenizer on a data sample before
training. It defines the integer vocabulary the model sees. Same BPE algorithm
as GPT-2/3/4 — only the vocab size is small here because the corpus is tiny.

**Run (after Phase 1):**
```bash
.venv/bin/python phase2_tokenizer/run.py
```

**Output (this folder):** `tokenizer.json` (merges + special tokens).
