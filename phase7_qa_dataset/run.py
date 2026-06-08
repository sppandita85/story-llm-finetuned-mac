"""Phase 7 -- Build an instruction (Q&A) dataset from the 50 stories.

Real-world equivalent: after pretraining, models are aligned with supervised
instruction data -- (prompt, ideal response) pairs that teach the model to
*follow instructions* rather than merely continue text. Production teams write or
collect such pairs at scale. We derive them programmatically from the structured
fields already present in each story (title, moral, body, lesson).

For every story we emit several instruction types:
  - "What is the moral of <title>?"        -> the moral
  - "What lesson does <title> teach?"       -> the lesson
  - "Tell me the story titled <title>."     -> the body
  - "Summarize the story <title>."          -> first sentence of the body
  - "Which story teaches: <lesson>?"        -> the title

Outputs (this folder):
  qa_dataset.jsonl  -- all pairs
  qa_train.jsonl / qa_val.jsonl -- 90/10 split
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (  # noqa: E402
    INPUT_FILE, QA_JSONL, QA_TRAIN_JSONL, QA_VAL_JSONL,
)


def parse_stories(raw: str) -> list[dict]:
    """Extract {title, moral, body, lesson} for each story from the markdown."""
    blocks = re.split(r"\n##\s+Story\s+\d+:\s*", raw)[1:]
    stories = []
    for b in blocks:
        lines = b.splitlines()
        title = lines[0].strip()
        rest = "\n".join(lines[1:])

        moral = _grab(rest, "Moral")
        lesson = _grab(rest, "Lesson")
        # Body = everything that is not the Moral/Lesson/rule lines.
        body_lines = []
        for ln in rest.splitlines():
            s = ln.strip()
            if not s or s == "---":
                continue
            if s.startswith("**Moral:") or s.startswith("**Lesson:"):
                continue
            body_lines.append(s)
        body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
        stories.append({"title": title, "moral": moral, "lesson": lesson, "body": body})
    return stories


def _grab(text: str, label: str) -> str:
    m = re.search(rf"\*\*{label}:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else ""


def first_sentence(text: str) -> str:
    m = re.search(r"^(.+?[.!?])(\s|$)", text)
    return m.group(1).strip() if m else text


def make_pairs(story: dict) -> list[dict]:
    title, moral, lesson, body = (story[k] for k in ("title", "moral", "lesson", "body"))
    pairs = []

    def add(instruction, output):
        if output:
            pairs.append({"instruction": instruction, "input": "", "output": output})

    add(f"What is the moral of the story '{title}'?", moral)
    add(f"What lesson does '{title}' teach?", lesson)
    add(f"Tell me the story titled '{title}'.", body)
    add(f"Summarize the story '{title}' in one sentence.", first_sentence(body))
    if lesson:
        add(f"Which story teaches this lesson: {lesson}", title)
    return pairs


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    stories = parse_stories(raw)
    print(f"Parsed {len(stories)} stories.")

    all_pairs = []
    for s in stories:
        all_pairs.extend(make_pairs(s))
    print(f"Generated {len(all_pairs)} instruction pairs "
          f"(~{len(all_pairs) / len(stories):.1f} per story).")

    # 90/10 split (shuffled deterministically).
    import random
    random.Random(1337).shuffle(all_pairs)
    n_val = max(1, len(all_pairs) // 10)
    val = all_pairs[:n_val]
    train = all_pairs[n_val:]

    for path, rows in [(QA_JSONL, all_pairs), (QA_TRAIN_JSONL, train), (QA_VAL_JSONL, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows):3d} -> {os.path.relpath(path)}")

    print("\n--- example pairs ---")
    for r in all_pairs[:3]:
        print(f"  Q: {r['instruction']}")
        print(f"  A: {r['output'][:90]}{'...' if len(r['output']) > 90 else ''}\n")
    print("Phase 7 complete.")


if __name__ == "__main__":
    main()
