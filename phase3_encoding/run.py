"""Phase 3 -- Encode the corpus into integer token shards + build a DataLoader.

Real-world equivalent: the cleaned corpus is run through the tokenizer and
written out as flat binary shards of token ids (uint16/uint32). Training then
streams random fixed-length windows from those shards. We do exactly that:
encode train/val text with the Phase-2 tokenizer, save train.bin / val.bin as
uint16 arrays, and show the sliding-window Dataset/DataLoader that turns a flat
token stream into (x, y) next-token prediction pairs.

Outputs (this folder):
  train.bin, val.bin -- uint16 token id arrays
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (  # noqa: E402
    TRAIN_TXT, VAL_TXT, TRAIN_BIN, VAL_BIN, TOKENIZER_JSON, cfg,
)
from common.bpe_tokenizer import BPETokenizer  # noqa: E402


class TokenWindowDataset(Dataset):
    """Yields (x, y) where y is x shifted by one -- the next-token target.

    A flat token array of length N with block_size B gives N-B training windows;
    window i is tokens[i:i+B] as input and tokens[i+1:i+B+1] as target.
    """

    def __init__(self, bin_path: str, block_size: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i: int):
        chunk = self.data[i: i + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def encode_file(tok: BPETokenizer, txt_path: str, bin_path: str) -> int:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tok.encode(text)            # special tokens (e.g. <|endoftext|>) preserved
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"  {os.path.basename(txt_path):10s} -> {os.path.basename(bin_path):10s} "
          f"{len(ids):,} tokens")
    return len(ids)


def main() -> None:
    tok = BPETokenizer.load(TOKENIZER_JSON)
    print(f"Loaded tokenizer (vocab {tok.n_vocab}).")

    print("Encoding splits to uint16 binary shards:")
    n_train = encode_file(tok, TRAIN_TXT, TRAIN_BIN)
    n_val = encode_file(tok, VAL_TXT, VAL_BIN)

    # Demonstrate the DataLoader on the train shard.
    ds = TokenWindowDataset(TRAIN_BIN, cfg.model.block_size)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)
    x, y = next(iter(dl))
    print("\n--- DataLoader sanity check ---")
    print(f"  windows in train set : {len(ds):,}")
    print(f"  batch x shape        : {tuple(x.shape)}  (batch, block_size)")
    print(f"  batch y shape        : {tuple(y.shape)}")
    # y should be x shifted by one -> verify on the first row.
    print(f"  y[0,:-1] == x[0,1:]  : {torch.equal(y[0, :-1], x[0, 1:])}")
    print(f"\nTotal tokens: {n_train:,} train + {n_val:,} val")
    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
