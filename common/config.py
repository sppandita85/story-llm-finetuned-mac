"""Central configuration: hyperparameters + every file path in one place.

In a real trillion-token training run this is exactly where the "single source
of truth" lives -- a config that the data pipeline, the model, the optimizer,
and the eval harness all read from. Scaling this project up to a serious run is
mostly a matter of changing the numbers in here (bigger model, more iters,
bigger vocab) and pointing the paths at a real corpus.

Everything is CPU-only and seeded so the whole build is reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Paths. PROJECT_ROOT is the folder that contains this `common/` package.
# Each phase writes its artifacts into its OWN folder; the next phase reads
# them from there. This keeps the build readable top-to-bottom.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE      = os.path.join(PROJECT_ROOT, "input", "childhood_stories_50_input.md")

# Phase 1 - data prep outputs
P1_DIR          = os.path.join(PROJECT_ROOT, "phase1_data_prep")
CORPUS_TXT      = os.path.join(P1_DIR, "corpus.txt")
TRAIN_TXT       = os.path.join(P1_DIR, "train.txt")
VAL_TXT         = os.path.join(P1_DIR, "val.txt")

# Phase 2 - tokenizer output
P2_DIR          = os.path.join(PROJECT_ROOT, "phase2_tokenizer")
TOKENIZER_JSON  = os.path.join(P2_DIR, "tokenizer.json")

# Phase 3 - encoded token shards
P3_DIR          = os.path.join(PROJECT_ROOT, "phase3_encoding")
TRAIN_BIN       = os.path.join(P3_DIR, "train.bin")
VAL_BIN         = os.path.join(P3_DIR, "val.bin")

# Phase 5 - pretrained checkpoint (THE core deliverable)
P5_DIR          = os.path.join(PROJECT_ROOT, "phase5_pretrain")
PRETRAINED_CKPT = os.path.join(P5_DIR, "pretrained.pt")

# Phase 7 - instruction Q&A dataset
P7_DIR          = os.path.join(PROJECT_ROOT, "phase7_qa_dataset")
QA_JSONL        = os.path.join(P7_DIR, "qa_dataset.jsonl")
QA_TRAIN_JSONL  = os.path.join(P7_DIR, "qa_train.jsonl")
QA_VAL_JSONL    = os.path.join(P7_DIR, "qa_val.jsonl")

# Phase 8 - fine-tuned checkpoint
P8_DIR          = os.path.join(PROJECT_ROOT, "phase8_finetune")
FINETUNED_CKPT  = os.path.join(P8_DIR, "finetuned.pt")


# --------------------------------------------------------------------------- #
# Special tokens. `<|endoftext|>` marks document boundaries during pretraining
# (just like GPT-2). The chat tokens structure the instruction format used in
# fine-tuning so the model can learn "after <|assistant|> I produce an answer".
# --------------------------------------------------------------------------- #
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
]


@dataclass
class TokenizerConfig:
    # Target vocabulary size for BPE. The corpus is tiny (~6k tokens) so a small
    # vocab is plenty; a real run uses 32k-256k. Includes the special tokens.
    vocab_size: int = 1024


@dataclass
class ModelConfig:
    # Filled in at load time once we know the tokenizer's real vocab size.
    vocab_size: int = 1024
    block_size: int = 128     # context length (max tokens the model attends over)
    n_layer: int = 4          # transformer blocks
    n_head: int = 4           # attention heads per block
    n_embd: int = 128         # embedding / hidden width
    dropout: float = 0.1
    bias: bool = True         # use bias terms in Linear/LayerNorm


@dataclass
class TrainConfig:
    # Pretraining schedule (Phase 5)
    batch_size: int = 16
    max_iters: int = 3000
    warmup_iters: int = 150
    eval_interval: int = 250
    eval_iters: int = 40          # batches averaged per loss estimate
    learning_rate: float = 3e-4
    min_lr: float = 3e-5          # cosine decays down to this
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Fine-tuning schedule (Phase 8) -- fewer steps, gentler LR.
    ft_batch_size: int = 8
    ft_max_iters: int = 800
    ft_warmup_iters: int = 40
    ft_eval_interval: int = 100
    ft_learning_rate: float = 5e-5
    ft_min_lr: float = 5e-6


@dataclass
class Config:
    seed: int = 1337
    device: str = "cpu"          # forced CPU-only, per project requirements
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# A single shared instance every phase imports.
cfg = Config()


def add_project_root_to_path() -> None:
    """Let `phaseN/run.py` do `from common import ...`.

    Each phase script calls this before importing `common.*` so the project
    root is on sys.path regardless of the directory the script is launched from.
    """
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
