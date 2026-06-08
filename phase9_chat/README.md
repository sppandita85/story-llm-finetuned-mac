# Phase 9 — Inference / Chat

**What it does:** Loads `phase8_finetune/finetuned.pt`, wraps a question in the
**same** instruction template used during fine-tuning (`common/chat_format.py`),
generates a response, and stops at `<|endoftext|>`.

**Real-world equivalent:** Serving the aligned model behind an API/REPL.

**Run (after Phase 8):**
```bash
.venv/bin/python phase9_chat/run.py "What is the moral of 'The Boy Who Cried Wolf'?"
.venv/bin/python phase9_chat/run.py        # interactive REPL
```

> Note: on a ~6K-token corpus the model heavily memorizes; answers will echo the
> source stories. That is expected at this scale — the value is that the full
> pretrain → SFT → serve pipeline is real and runs end-to-end.
