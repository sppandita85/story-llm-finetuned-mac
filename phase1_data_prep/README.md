# Phase 1 — Data Preparation

**What it does:** Reads `input/childhood_stories_50_input.md`, strips the markdown
scaffolding into clean running text, inserts an `<|endoftext|>` boundary between
stories, prints corpus statistics, and writes a 90/10 train/val split.

**Real-world equivalent:** The data-curation stage. A trillion-token run crawls,
filters, dedupes, and normalizes huge volumes of text and inserts document
separators before tokenization. Same idea, miniature scale.

**Run:**
```bash
.venv/bin/python phase1_data_prep/run.py
```

**Outputs (this folder):** `corpus.txt`, `train.txt`, `val.txt`.
