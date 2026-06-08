# Phase 7 — Instruction (Q&A) Dataset

**What it does:** Parses the structured fields of each story (title, moral, body,
lesson) from the original markdown and programmatically emits several
instruction/response pairs per story (moral, lesson, retell, summarize, reverse
lookup). Writes `qa_dataset.jsonl` plus a 90/10 `qa_train`/`qa_val` split. Each
row is `{"instruction", "input", "output"}`.

**Real-world equivalent:** Collecting supervised instruction data — (prompt,
ideal response) pairs that teach a base model to *follow instructions* instead
of merely continuing text.

**Run (after Phase 1; uses the original input file):**
```bash
.venv/bin/python phase7_qa_dataset/run.py
```

**Outputs (this folder):** `qa_dataset.jsonl`, `qa_train.jsonl`, `qa_val.jsonl`.
