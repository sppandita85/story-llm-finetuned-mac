"""A custom Byte-Pair-Encoding (BPE) tokenizer, implemented from scratch.

This is the same algorithm GPT-2/GPT-3/GPT-4 use for tokenization, just written
out plainly so every step is visible:

  1. Start from raw UTF-8 BYTES, so the base vocabulary is the 256 possible byte
     values. This means *any* string is encodable -- there is no "unknown token".
  2. Repeatedly find the most frequent adjacent pair of tokens in the corpus and
     "merge" it into a single new token, growing the vocabulary one entry at a
     time until we hit the target vocab size.
  3. Special tokens (<|endoftext|>, <|user|>, ...) are appended at the end of the
     vocabulary and are matched literally before byte-level encoding, so they can
     never be split apart by the merges.

A real tokenizer (e.g. tiktoken) also applies a regex "pre-tokenizer" to split
text into word-like chunks before merging, which keeps merges from spanning
spaces/punctuation. We do the same with a GPT-2-style regex below.

Training and encoding live in the same class so the merges learned in Phase 2
are exactly the merges applied in Phases 3, 6, 8, and 9.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import regex as re

# GPT-2's pre-tokenization regex: splits on contractions, letter-runs,
# number-runs, punctuation-runs, and whitespace. Keeps a leading space attached
# to a word (" the" stays together), which is how GPT-2 represents word starts.
_GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _get_pair_counts(ids: List[int], counts: Dict[Tuple[int, int], int] | None = None):
    """Count how often each adjacent pair (a, b) appears in a list of token ids."""
    counts = {} if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        # merges: ordered map (pair) -> new token id. Order matters: merges must
        # be applied at encode time in the same order they were learned.
        self.merges: Dict[Tuple[int, int], int] = {}
        # vocab: token id -> raw bytes it expands to.
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # special tokens: string -> id, kept above the BPE id range.
        self.special_tokens: Dict[str, int] = {}
        self.special_inverse: Dict[int, str] = {}
        self._compiled = re.compile(_GPT2_SPLIT_PATTERN)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, special_tokens: List[str], verbose: bool = True) -> None:
        """Learn BPE merges from `text` up to `vocab_size` total tokens."""
        n_special = len(special_tokens)
        assert vocab_size >= 256 + n_special, "vocab_size too small for bytes + specials"
        num_merges = vocab_size - 256 - n_special

        # Pre-tokenize into chunks, then turn each chunk into a list of byte ids.
        # Merges are only ever counted/applied *within* a chunk, never across the
        # boundaries the regex found.
        chunks = self._compiled.findall(text)
        ids_per_chunk: List[List[int]] = [list(ch.encode("utf-8")) for ch in chunks]

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            # Tally pair frequencies across all chunks.
            counts: Dict[Tuple[int, int], int] = {}
            for ids in ids_per_chunk:
                _get_pair_counts(ids, counts)
            if not counts:
                break  # nothing left to merge
            # Most frequent pair wins.
            best = max(counts, key=counts.get)
            new_id = 256 + m
            ids_per_chunk = [_merge(ids, best, new_id) for ids in ids_per_chunk]
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose and (m < 5 or (m + 1) % 100 == 0):
                print(f"  merge {m + 1}/{num_merges}: {best} -> {new_id} "
                      f"({self.vocab[new_id]!r}) count={counts[best]}")

        # Register special tokens at the top of the id range.
        self.special_tokens = {}
        self.special_inverse = {}
        next_id = 256 + len(self.merges)
        for tok in special_tokens:
            self.special_tokens[tok] = next_id
            self.special_inverse[next_id] = tok
            next_id += 1

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """Apply the learned merges to one chunk's raw bytes."""
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            counts = _get_pair_counts(ids)
            # Pick the pair whose merge was learned earliest (lowest new id).
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no remaining pair is mergeable
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode text that contains NO special tokens."""
        out: List[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> List[int]:
        """Encode text, treating any registered special token literally.

        We split the text on the special-token strings first, encode the gaps
        with ordinary BPE, and drop in the special ids at the boundaries.
        """
        if not allowed_special or not self.special_tokens:
            return self.encode_ordinary(text)

        specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
        parts = re.split(pattern, text)
        out: List[int] = []
        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: List[int]) -> str:
        """Turn token ids back into a string."""
        pieces: List[bytes] = []
        for i in ids:
            if i in self.special_inverse:
                pieces.append(self.special_inverse[i].encode("utf-8"))
            elif i in self.vocab:
                pieces.append(self.vocab[i])
            # ids outside both maps are skipped (shouldn't happen normally)
        return b"".join(pieces).decode("utf-8", errors="replace")

    @property
    def n_vocab(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = {
            # store merges as "a,b": new_id  (json keys must be strings)
            "merges": {f"{a},{b}": nid for (a, b), nid in self.merges.items()},
            "special_tokens": self.special_tokens,
            "n_vocab": self.n_vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        # Re-apply merges IN ID ORDER so each new token's bytes can be built from
        # tokens that already exist.
        for key, nid in sorted(data["merges"].items(), key=lambda kv: kv[1]):
            a, b = (int(x) for x in key.split(","))
            tok.merges[(a, b)] = nid
            tok.vocab[nid] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        tok.special_inverse = {v: k for k, v in tok.special_tokens.items()}
        return tok
