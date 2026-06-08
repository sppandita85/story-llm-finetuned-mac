# Phase 4 — Model Architecture (sanity check)

**What it does:** Instantiates the decoder-only GPT transformer
(`common/model.py`) sized to the Phase-2 vocab, prints the parameter count and
breakdown, and runs one forward + one backward pass on a random batch to prove
shapes are correct and gradients flow. The initial loss should be near `ln(V)`,
the expected loss of a randomly initialized model over `V` classes.

**Real-world equivalent:** Defining the model and validating a forward/backward
pass on a single batch before committing large compute to training.

**Run (after Phase 2):**
```bash
.venv/bin/python phase4_model/run.py
```

**Outputs:** none (verification only).
