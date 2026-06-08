"""Phase 10b -- Convert the HuggingFace GPT-2 model to a GGUF file for Ollama.

This is a thin, reproducible wrapper around llama.cpp's `convert_hf_to_gguf.py`.

Why a wrapper instead of calling the converter directly: llama.cpp fingerprints
the tokenizer's pre-tokenizer and refuses unknown ones with
"BPE pre-tokenizer was not recognized". Our tokenizer IS a standard GPT-2
byte-level BPE (just with a small custom vocabulary), so the correct answer is
the "gpt-2" pre-tokenizer. We monkey-patch the converter's fallback to return
"gpt-2" for our (otherwise unrecognized) fingerprint, then run it normally.

This keeps the vendored llama.cpp checkout unmodified, so it survives a
re-clone.

Run (after export_hf.py):
    .venv/bin/python phase10_ollama_export/convert_to_gguf.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.join(HERE, "llama.cpp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=os.path.join(HERE, "hf_model"))
    ap.add_argument("--outfile", default=os.path.join(HERE, "story-llm.gguf"))
    ap.add_argument("--outtype", default="f16")
    args = ap.parse_args()

    if not os.path.isdir(LLAMA):
        sys.exit(f"llama.cpp not found at {LLAMA}. Clone it first (see README).")

    # Make the converter and its `conversion` package importable.
    sys.path.insert(0, LLAMA)
    import convert_hf_to_gguf as conv          # the converter script
    from conversion.base import TextModel       # class holding the fingerprint check

    # Patch the pre-tokenizer detection: fall back to GPT-2 for our tokenizer.
    _orig = TextModel.get_vocab_base_pre

    def _patched(self, tokenizer):
        try:
            return _orig(self, tokenizer)
        except NotImplementedError:
            logging.getLogger("convert").warning(
                "Unrecognized BPE pre-tokenizer -> assuming 'gpt-2' "
                "(our tokenizer is GPT-2 byte-level BPE)."
            )
            return "gpt-2"

    TextModel.get_vocab_base_pre = _patched

    # Drive the converter exactly as its CLI would.
    sys.argv = [
        "convert_hf_to_gguf.py",
        args.hf_dir,
        "--outfile", args.outfile,
        "--outtype", args.outtype,
    ]
    conv.main()
    print(f"\nGGUF written to: {os.path.relpath(args.outfile)}")


if __name__ == "__main__":
    main()
