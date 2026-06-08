# Building a Large Language Model — from scratch, end to end

A miniature but **architecturally faithful** reproduction of the full LLM
life-cycle, trained on `input/childhood_stories_50_input.md` (50 moral stories,
~6K tokens). Every phase you'd run to pretrain a model on *trillions* of tokens
is here — only the data volume, model size, and hardware differ. The model is
hand-coded in PyTorch (no HuggingFace / nanoGPT), runs **CPU-only**, and is fully
seeded for reproducibility.

> Because the corpus is tiny, the model **memorizes/overfits** — that's expected.
> The point is that the *pipeline* is real: the same code scales up by changing
> the numbers in `common/config.py` and pointing at a bigger corpus.

## The phases

| Phase | Folder | What happens | Real-world analogue |
|------:|--------|--------------|---------------------|
| 1 | `phase1_data_prep/` | clean markdown → corpus, insert `<|endoftext|>`, train/val split | data curation |
| 2 | `phase2_tokenizer/` | train a custom **BPE** tokenizer | learning the tokenizer |
| 3 | `phase3_encoding/` | encode → `train.bin`/`val.bin`, sliding-window DataLoader | tokenization + sharding |
| 4 | `phase4_model/` | build GPT, param count, forward/backward check | model definition gate |
| 5 | `phase5_pretrain/` | **pretrain** (AdamW, warmup+cosine LR, clipping, checkpoints) | pretraining ← core |
| 6 | `phase6_generate/` | sample text from the base model | base-model eval |
| 7 | `phase7_qa_dataset/` | derive instruction **Q&A** pairs | SFT data collection |
| 8 | `phase8_finetune/` | **supervised fine-tuning** (loss masked to answers) | instruction tuning |
| 9 | `phase9_chat/` | ask the fine-tuned model questions | serving |

Shared code lives in `common/`: `config.py` (all hyperparameters + paths),
`bpe_tokenizer.py` (the BPE algorithm), `model.py` (the GPT transformer), and
`chat_format.py` (the instruction template used by both Phase 8 and Phase 9).

## Setup

```bash
cd building-LargeLanguageModel
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run everything

```bash
bash run_all.sh                          # phases 1 → 8
.venv/bin/python phase9_chat/run.py "What is the moral of 'The Boy Who Cried Wolf'?"
```

Or run any phase on its own (each reads the previous phase's outputs):

```bash
.venv/bin/python phase1_data_prep/run.py
.venv/bin/python phase2_tokenizer/run.py
# ... etc
```

## Deliverables

- `phase5_pretrain/pretrained.pt` — the **pretrained base LLM**.
- `phase7_qa_dataset/qa_dataset.jsonl` — the **instruction dataset** for fine-tuning.
- `phase8_finetune/finetuned.pt` — the **instruction-tuned LLM**.

## Scaling up (the only changes needed for a "real" run)

Edit `common/config.py`: raise `vocab_size`, `n_layer`/`n_head`/`n_embd`,
`block_size`, and `max_iters`; point the data paths at a large corpus; switch
`device` to a GPU. The phase scripts do not change.
