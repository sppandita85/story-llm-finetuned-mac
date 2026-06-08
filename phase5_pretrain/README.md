# Phase 5 — Pretraining (the core deliverable)

**What it does:** Trains the GPT to predict the next token on the Phase-3 token
shards. The loop is structurally identical to a real pretraining run: AdamW with
decoupled weight decay, linear-warmup + cosine-decay learning rate, gradient
clipping, periodic train/val loss estimation, sample-generation logging, and
best-by-val-loss checkpointing.

**Real-world equivalent:** This *is* LLM pretraining — only the data volume,
model size, and cluster differ. The same code scales up by changing the numbers
in `common/config.py` and pointing at a larger corpus.

**Run (after Phase 3):**
```bash
.venv/bin/python phase5_pretrain/run.py
```

**Output (this folder):** `pretrained.pt` — the pretrained base LLM (weights +
config). Expect train/val loss to fall steadily and samples to become
story-like (and, on this tiny corpus, to memorize — that is expected).
