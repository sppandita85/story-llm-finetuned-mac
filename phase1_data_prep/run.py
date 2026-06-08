"""Phase 1 -- Data preparation (cleaning, normalization, train/val split).

Real-world equivalent: the data-curation stage of an LLM. A trillion-token run
crawls/filters/dedupes/normalizes petabytes of text and inserts document
separators before anything is tokenized. We do the miniature version: read the
markdown stories, strip the markdown scaffolding into clean running text, insert
an <|endoftext|> document boundary between stories, report corpus statistics,
and hold out a validation slice.

Outputs (in this folder):
  corpus.txt  -- the full cleaned corpus
  train.txt   -- 90% used for pretraining
  val.txt     -- 10% held out to measure generalization
"""

from __future__ import annotations

import os
import re
import sys

# Make `common` importable no matter where this is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (  # noqa: E402
    INPUT_FILE, CORPUS_TXT, TRAIN_TXT, VAL_TXT, cfg,
)

EOT = "<|endoftext|>"


def clean_markdown(raw: str) -> list[str]:
    """Split the markdown into one cleaned document per story.

    We keep the human-readable text (title, moral, body, lesson) but drop the
    markdown noise (#, **, ---, the dataset header). Each story becomes one
    'document' so we can separate them with <|endoftext|>.
    """
    # Split on the story headers, keeping the header text.
    # Stories look like:  ## Story 1: The Boy Who Cried Wolf
    parts = re.split(r"\n##\s+Story\s+\d+:\s*", raw)
    documents: list[str] = []
    for part in parts[1:]:  # parts[0] is the dataset header preamble; skip it
        # The first line of `part` is the title (rest of the header line).
        lines = part.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:])

        # Drop horizontal rules and the trailing dataset separators.
        body = body.replace("---", " ")
        # Convert "**Moral:** X" / "**Lesson:** X" to "Moral: X".
        body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
        # Collapse whitespace.
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r"\n{2,}", "\n", body).strip()

        doc = f"{title}\n{body}".strip()
        documents.append(doc)
    return documents


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    documents = clean_markdown(raw)
    print(f"Parsed {len(documents)} story documents from {os.path.basename(INPUT_FILE)}")

    # Join documents with the end-of-text boundary token. Trailing EOT too, so
    # the model also learns where documents end.
    corpus = (f"\n{EOT}\n".join(documents)) + f"\n{EOT}\n"

    # Corpus statistics (the kind of thing you log for any real dataset).
    n_chars = len(corpus)
    n_words = len(corpus.split())
    print("--- corpus statistics ---")
    print(f"  characters : {n_chars:,}")
    print(f"  words      : {n_words:,}")
    print(f"  documents  : {len(documents)}")
    print(f"  est tokens : ~{int(n_words / 0.75):,}  (rough words/0.75 heuristic)")

    # Document-level 90/10 split so a whole story stays on one side.
    n_val = max(1, len(documents) // 10)
    train_docs = documents[:-n_val]
    val_docs = documents[-n_val:]
    train_text = (f"\n{EOT}\n".join(train_docs)) + f"\n{EOT}\n"
    val_text = (f"\n{EOT}\n".join(val_docs)) + f"\n{EOT}\n"

    for path, text, label in [
        (CORPUS_TXT, corpus, "corpus"),
        (TRAIN_TXT, train_text, "train"),
        (VAL_TXT, val_text, "val"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  wrote {label:7s} -> {os.path.relpath(path)}  ({len(text):,} chars)")

    print(f"\nSplit: {len(train_docs)} train docs / {len(val_docs)} val docs")
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
