# Phase 6 — Generation / Base-Model Evaluation

**What it does:** Loads `phase5_pretrain/pretrained.pt` and continues several
prompts. A base LLM is a text *continuer* (not yet a chat assistant); this is a
qualitative check of what pretraining learned.

**Real-world equivalent:** Probing the base model with prompts after pretraining,
before any instruction tuning.

**Run (after Phase 5):**
```bash
.venv/bin/python phase6_generate/run.py                 # built-in prompts
.venv/bin/python phase6_generate/run.py "A clever fox"  # custom prompt
```

**Outputs:** none (prints generations).
