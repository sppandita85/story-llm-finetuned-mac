# Project Plan — Building a Large Language Model from Scratch

A plain-language walkthrough of this project: what we built, why, and what each
phase does.

---

## What is this project?

We built a **small Large Language Model (LLM)** — the same kind of technology
behind ChatGPT — completely **from scratch in Python/PyTorch**, and trained it on
a tiny dataset of 50 children's moral stories.

The goal was **not** to build something as smart as ChatGPT. The goal was to walk
through **every single step** that a real company follows when training an LLM on
*trillions* of words — but shrunk down so it runs on a normal laptop CPU in a few
minutes.

Think of it like building a **working model airplane** to learn how real planes
fly. It won't carry passengers, but every part — wings, engine, controls — is
real and works the same way.

### Why the model "memorizes" instead of being smart

Real LLMs read trillions of words. Ours read about **6,000**. With so little
data, the model simply **memorizes** the stories rather than learning to reason.
That is completely expected and is actually useful for learning — you can watch
it soak up the data. To make it genuinely smart you would just feed the *same
code* far more data and use a bigger model.

---

## How the project is organized

The project is split into **9 phases**, and **each phase has its own folder** so
you can read it top to bottom like chapters in a book:

```
input/              <- the raw dataset (50 stories)
common/             <- shared building blocks used by every phase
phase1_data_prep/   <- clean the data
phase2_tokenizer/   <- build the "vocabulary"
phase3_encoding/    <- turn text into numbers
phase4_model/       <- build the AI brain (untrained)
phase5_pretrain/    <- TRAIN the brain  (the main event)
phase6_generate/    <- test what it learned
phase7_qa_dataset/  <- create question/answer practice material
phase8_finetune/    <- teach it to answer questions
phase9_chat/        <- talk to the finished model
```

The **`common/`** folder holds the three reusable pieces so we never write the
same code twice:
- `config.py` — every setting and file location in one place (the "control panel")
- `bpe_tokenizer.py` — the code that splits text into tokens
- `model.py` — the actual neural network (the GPT)
- `chat_format.py` — the template for question/answer conversations

Everything installs into a **project-local `.venv`** (virtual environment), so
all the software stays inside this project and doesn't touch the rest of your
computer.

---

## The 9 phases, in simple language

### Phase 1 — Data Preparation 🧹
**Folder:** `phase1_data_prep/`

We take the raw story file and **clean it up**: remove formatting symbols, tidy
the spacing, and put a special marker (`<|endoftext|>`) between stories so the
model knows where one story ends and the next begins. Then we set aside a small
part of the data (5 stories) as a **test set** the model never trains on, so we
can later check if it's actually learning or just cheating.

> *Real-world version:* Companies collect text from the internet, filter out junk,
> remove duplicates, and clean it — usually the **most expensive and important**
> step of all.

### Phase 2 — Tokenizer (Building the Vocabulary) 🔤
**Folder:** `phase2_tokenizer/`

Computers don't understand letters — they understand numbers. A **tokenizer**
decides how to chop text into small pieces called **tokens** (roughly: common
chunks of words), and gives each a number. We *train* this on our stories using
the **BPE (Byte-Pair Encoding)** method — the exact same technique GPT uses. It
starts with single characters and repeatedly glues together the most common
pairs (like `t`+`h` → `th`) until it has a vocabulary of ~1,000 tokens.

> *Real-world version:* Identical method, just a much bigger vocabulary
> (50,000–250,000 tokens).

### Phase 3 — Encoding & Sharding 🔢
**Folder:** `phase3_encoding/`

Now we run all our cleaned text through the tokenizer to convert it into a long
list of numbers, and save those numbers to disk (`train.bin` / `val.bin`). We
also build a **data loader** — a conveyor belt that feeds the model little
windows of text and asks it to **predict the next token**. That "predict the next
word" game is the heart of how LLMs learn.

> *Real-world version:* Exactly the same — text is tokenized once and stored as
> number files that training reads from.

### Phase 4 — Building the Model 🧠
**Folder:** `phase4_model/`

Here we assemble the **neural network** itself — a "GPT" (the same architecture
family as ChatGPT). It's built from:
- **Attention** — lets each word look back at earlier words to understand context
- **Layers** stacked on top of each other (ours has 4; big models have 100+)

This phase doesn't train anything; it just **builds the brain and checks it
works** (correct shapes, ~940,000 adjustable knobs called *parameters*). At this
point the model knows nothing — its answers are random.

> *Real-world version:* Same design; real models have **billions** of parameters.

### Phase 5 — Pretraining (The Main Event) 🏋️
**Folder:** `phase5_pretrain/`

This is where the model **actually learns**. We show it text over and over and it
plays the "guess the next token" game thousands of times. Each time it guesses
wrong, it **slightly adjusts its parameters** to do better next time. We use all
the professional techniques:
- **AdamW optimizer** — the smart adjustment method
- **Learning-rate schedule** (warm up, then cool down) — how big the adjustments
  are over time
- **Gradient clipping** — a safety brake so training doesn't blow up
- **Checkpoints** — saving the best version as it improves

We watched the error drop from **6.95 down to 0.07** — proof it learned the data.
The result, `pretrained.pt`, **is the trained LLM** — the core deliverable.

> *Real-world version:* This is the step that costs millions of dollars and runs
> for weeks on thousands of GPUs. Same code, vastly bigger scale.

### Phase 6 — Generating Text ✍️
**Folder:** `phase6_generate/`

We give the trained model a starting phrase like *"Once upon a time"* and let it
**write a continuation** one token at a time. This shows what it learned. Our
model produces story-flavored text with morals and lessons — rough, but clearly
in the style of the data. A freshly pretrained model is a **text continuer**, not
yet a question-answerer.

> *Real-world version:* The same kind of "let's see what the base model can do"
> testing before any further tuning.

### Phase 7 — Question & Answer Dataset 📝
**Folder:** `phase7_qa_dataset/`

A base model just continues text; it doesn't **follow instructions**. To teach
that, we need examples of good questions and answers. We automatically create
**250 Q&A pairs** from the stories — e.g. *"What is the moral of 'The Boy Who
Cried Wolf'?"* → *"Honesty is the best policy."* These become the practice
material for the next phase.

> *Real-world version:* Companies hire people to write thousands of high-quality
> instruction/answer examples (this is the "alignment" data).

### Phase 8 — Fine-Tuning 🎯
**Folder:** `phase8_finetune/`

We take the pretrained model from Phase 5 and **continue training it**, but now
on the Q&A pairs from Phase 7. The trick: we only grade the model on the
**answer** part, not the question — so it learns to *produce answers*, not repeat
questions. This turns a text-continuer into something that **responds to
instructions**. The result is saved as `finetuned.pt`.

> *Real-world version:* Called **SFT (Supervised Fine-Tuning)** — the step that
> turns a raw model into a helpful assistant.

### Phase 9 — Chatting with the Model 💬
**Folder:** `phase9_chat/`

Finally, we **talk to the finished model**. We wrap a question in the same
template used during fine-tuning, the model generates an answer, and stops when
it's done. You can ask it questions from the command line.

> *Real-world version:* "Serving" — putting the model behind an app or API so
> people can use it.

---

## The end products

After running all phases you get three things:

| File | What it is |
|------|------------|
| `phase5_pretrain/pretrained.pt` | The **pretrained LLM** (the main goal) |
| `phase7_qa_dataset/qa_dataset.jsonl` | The **Q&A dataset** for fine-tuning |
| `phase8_finetune/finetuned.pt` | The **fine-tuned**, instruction-following LLM |

---

## How to run it

**One-time setup:**
```bash
cd building-LargeLanguageModel
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Run the whole pipeline (phases 1–8):**
```bash
bash run_all.sh
```

**Then chat with your model:**
```bash
.venv/bin/python phase9_chat/run.py "What is the moral of 'The Boy Who Cried Wolf'?"
```

**Or run any single phase** (each one reads the previous phase's output):
```bash
.venv/bin/python phase1_data_prep/run.py
.venv/bin/python phase2_tokenizer/run.py
# ...and so on
```

---

## How to make it smarter

Everything is controlled from `common/config.py`. To go from "toy" toward "real",
you only change numbers — **not the code**:
- Bigger brain: increase `n_layer`, `n_head`, `n_embd`
- Longer memory: increase `block_size`
- More learning: increase `max_iters`
- More data: point the input path at a much larger text collection
- Faster training: switch `device` from `"cpu"` to a GPU

The same 9 phases would then train a genuinely capable model.
