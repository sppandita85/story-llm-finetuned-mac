# Phase 3 — Encoding & Sharding

**What it does:** Loads the Phase-2 tokenizer, encodes `train.txt`/`val.txt` into
flat `uint16` token arrays (`train.bin`, `val.bin`), and demonstrates the
sliding-window `Dataset`/`DataLoader` that turns a flat token stream into
`(x, y)` next-token prediction pairs (`y` = `x` shifted by one).

**Real-world equivalent:** Corpora are tokenized once and written as binary
shards; training then streams random fixed-length windows from them. Same here.

**Run (after Phase 2):**
```bash
.venv/bin/python phase3_encoding/run.py
```

**Outputs (this folder):** `train.bin`, `val.bin`.
