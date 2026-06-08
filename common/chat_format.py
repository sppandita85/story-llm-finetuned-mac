"""The instruction/chat prompt format, shared by fine-tuning and inference.

Keeping this in one place guarantees the template the model is TRAINED on
(Phase 8) is byte-for-byte the template used at INFERENCE (Phase 9). A mismatch
here is one of the most common real-world fine-tuning bugs.

Template:

    <|user|>
    {instruction}{optional input}
    <|assistant|>
    {response}<|endoftext|>

During fine-tuning we only compute loss on the response portion (everything from
just after "<|assistant|>\n" through the final <|endoftext|>), so the model
learns to *produce* answers, not to reproduce the prompt.
"""

from __future__ import annotations

from typing import List, Tuple


def build_prompt(instruction: str, inp: str = "") -> str:
    """The part the model is conditioned on (ends right where the answer starts)."""
    user = instruction if not inp else f"{instruction}\n{inp}"
    return f"<|user|>\n{user}\n<|assistant|>\n"


def build_example(instruction: str, inp: str, output: str) -> str:
    """Full training string: prompt + answer + end-of-text."""
    return build_prompt(instruction, inp) + output + "<|endoftext|>"


def encode_example(tok, instruction: str, inp: str, output: str) -> Tuple[List[int], List[int]]:
    """Return (token_ids, loss_mask) for one fine-tuning example.

    loss_mask is 1 on response tokens (the answer + <|endoftext|>) and 0 on the
    prompt tokens, so the loss ignores the prompt.
    """
    prompt = build_prompt(instruction, inp)
    answer = output + "<|endoftext|>"
    prompt_ids = tok.encode(prompt)
    answer_ids = tok.encode(answer)
    ids = prompt_ids + answer_ids
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
    return ids, mask
