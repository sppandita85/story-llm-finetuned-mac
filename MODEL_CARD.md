# Model Card — story-llm

A **GPT-style language model built entirely from scratch** (custom transformer +
custom BPE tokenizer, no HuggingFace/nanoGPT for the model) and trained on 50
classic children's moral stories. It is an educational, end-to-end reproduction
of the full LLM lifecycle — data prep → tokenizer → pretraining → fine-tuning →
serving — at miniature scale on a laptop CPU.

> ⚠️ **This is a learning artifact, not a usable assistant.** At 0.94M parameters
> trained on ~6K tokens it memorizes rather than reasons, so its output is
> garbled and unreliable. The value is the *pipeline*, which scales unchanged to
> a real model.

## Architecture

| Property | Value |
|---|---|
| Type | Decoder-only transformer (GPT-2 family) |
| Parameters | **940,800** (~0.94M) |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dim | 128 |
| Context length | 128 tokens |
| Vocabulary | 1,024 tokens |
| Serving precision | f16 (GGUF) |
| Notable | causal self-attention, pre-LayerNorm, GELU MLP, learned positional embeddings, weight-tied head |

## Tokenizer

- **Custom byte-level BPE**, trained from scratch on the corpus
- 256 byte tokens + 764 learned merges + 4 special tokens
  (`<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`)
- ~3.4 chars/token, lossless round-trip

## Training data & procedure

- **Dataset:** 50 classic moral stories (~4,256 words ≈ 6,760 train / 837 val tokens)
- **Pretraining:** next-token prediction; AdamW; linear-warmup + cosine-decay LR;
  gradient clipping → train loss **6.95 → 0.07**
- **Fine-tuning (SFT):** 250 auto-generated instruction Q&A pairs; loss masked to
  answer tokens only → val loss **4.11 → 3.17**
- **Compute:** CPU-only, fixed seed, fully reproducible; trains in minutes

## Intended use & limitations

- ✅ **Intended:** learning how an LLM is built and trained; demonstrating a real,
  complete pipeline (data → BPE → encoding → model → pretrain → SFT → Ollama).
- ❌ **Not intended:** any production or factual use. Output is unreliable by
  design at this scale.
- **Bias/scope:** trained only on 50 short moral fables; it knows nothing else.

## How to run

**Ollama (easiest):**
```bash
ollama run sppandita85/story-llm "What is the moral of 'The Boy Who Cried Wolf'?"
```

**From source (PyTorch):**
```bash
git clone https://github.com/sppandita85/story-llm-finetuned-mac.git
cd story-llm-finetuned-mac
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python phase9_chat/run.py "What lesson does 'The Golden Goose' teach?"
```

## Scaling toward a real model

Edit `common/config.py` (more layers/width, longer context, larger vocab, far
more data, more iterations) and re-run the same 10 phases. The pipeline is
unchanged — only the numbers grow.

## Links

- **Source & full 10-phase write-up:** https://github.com/sppandita85/story-llm-finetuned-mac
- **Hosted model:** https://ollama.com/sppandita85/story-llm

## License / attribution

Built as an educational project. Training data: 50 classic public-domain-style
moral fables included in the repository (`input/`).
