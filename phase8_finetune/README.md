# Phase 8 — Supervised Fine-Tuning (SFT)

**What it does:** Loads the pretrained base model (`phase5_pretrain/pretrained.pt`)
and continues training it on the Phase-7 instruction pairs, with the loss
**masked to the response tokens only** (prompt tokens are ignored via
`ignore_index`). Uses a lower learning rate so pretrained knowledge is preserved.
Saves the best-by-val-loss `finetuned.pt`.

**Real-world equivalent:** Instruction tuning / SFT — the alignment step that
turns a text-continuing base model into one that follows instructions. The
prompt template lives in `common/chat_format.py` and is shared with Phase 9.

**Run (after Phases 5 and 7):**
```bash
.venv/bin/python phase8_finetune/run.py
```

**Output (this folder):** `finetuned.pt` — the instruction-following model.
