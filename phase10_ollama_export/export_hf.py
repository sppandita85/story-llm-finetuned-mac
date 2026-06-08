"""Phase 10a -- Export our from-scratch model to HuggingFace GPT-2 format.

Ollama cannot load a raw PyTorch .pt; it runs GGUF files. The conversion route
to GGUF goes through the standard HuggingFace GPT-2 format, which llama.cpp's
converter understands. This script is that first hop:

    our pretrained.pt/finetuned.pt + our BPE tokenizer
        -->  HuggingFace GPT2LMHeadModel + GPT2TokenizerFast  (./hf_model/)

Why it works: our model in common/model.py is deliberately a GPT-2 architecture
(learned positional embeddings, pre-LN blocks, causal MHSA, GELU MLP, tied
head). The only real differences from HF's implementation are mechanical:

  - HF GPT-2 stores attention/MLP weights in `Conv1D` layout, which is the
    TRANSPOSE of our `nn.Linear` weights -> we transpose c_attn/c_proj/c_fc.
  - Our byte-level BPE must be re-expressed as GPT-2 `vocab.json` + `merges.txt`
    using GPT-2's byte->unicode table.

Run (after building the model):
    .venv/bin/python phase10_ollama_export/export_hf.py --model finetuned
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import (  # noqa: E402
    TOKENIZER_JSON, PRETRAINED_CKPT, FINETUNED_CKPT, SPECIAL_TOKENS,
)
from common.bpe_tokenizer import BPETokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.join(HERE, "hf_model")


# --------------------------------------------------------------------------- #
# GPT-2 byte<->unicode table. GPT-2 represents raw bytes as printable unicode
# characters so the BPE vocabulary is text-safe. This is the canonical mapping.
# --------------------------------------------------------------------------- #
def bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def export_tokenizer(tok: BPETokenizer, out_dir: str):
    """Write GPT-2 vocab.json + merges.txt from our BPE, then build a fast HF tokenizer."""
    from transformers import GPT2TokenizerFast
    from tokenizers import AddedToken

    byte_encoder = bytes_to_unicode()

    def tok_str(b: bytes) -> str:
        return "".join(byte_encoder[x] for x in b)

    # Non-special tokens occupy ids 0..(N-1); specials come after.
    n_special = len(SPECIAL_TOKENS)
    base_n = tok.n_vocab - n_special  # = 256 + len(merges)

    # vocab.json : token_string -> id  (ids 0..base_n-1)
    vocab = {tok_str(tok.vocab[i]): i for i in range(base_n)}
    vpath = os.path.join(out_dir, "vocab.json")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # merges.txt : the byte-pair merges, in the order they were learned.
    mpath = os.path.join(out_dir, "merges.txt")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for (a, b), _nid in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            f.write(f"{tok_str(tok.vocab[a])} {tok_str(tok.vocab[b])}\n")

    # Build the fast HF tokenizer from vocab+merges, then append specials in the
    # SAME id order our model expects (endoftext, user, assistant, pad).
    hf_tok = GPT2TokenizerFast(vocab_file=vpath, merges_file=mpath)
    hf_tok.add_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])
    hf_tok.eos_token = "<|endoftext|>"
    hf_tok.bos_token = "<|endoftext|>"
    hf_tok.pad_token = "<|pad|>"

    # sanity: ids must line up with our tokenizer
    for t in SPECIAL_TOKENS:
        assert hf_tok.convert_tokens_to_ids(t) == tok.special_tokens[t], \
            f"special id mismatch for {t}"
    assert hf_tok.vocab_size + len(hf_tok.get_added_vocab()) >= tok.n_vocab

    hf_tok.save_pretrained(out_dir)
    print(f"  tokenizer: {tok.n_vocab} tokens "
          f"({base_n} BPE + {n_special} special) -> {out_dir}")
    return hf_tok


def export_model(src_pt: str, out_dir: str, eos_id: int):
    """Map our state_dict into a HF GPT2LMHeadModel and save it."""
    from transformers import GPT2Config, GPT2LMHeadModel

    ckpt = torch.load(src_pt, map_location="cpu")
    mc = ckpt["model_cfg"]
    sd = ckpt["model_state"]

    config = GPT2Config(
        vocab_size=mc["vocab_size"],
        n_positions=mc["block_size"],
        n_ctx=mc["block_size"],
        n_embd=mc["n_embd"],
        n_layer=mc["n_layer"],
        n_head=mc["n_head"],
        activation_function="gelu",        # we use exact GELU (nn.GELU)
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        bos_token_id=eos_id, eos_token_id=eos_id,
    )
    hf = GPT2LMHeadModel(config)

    new_sd = {
        "transformer.wte.weight": sd["transformer.wte.weight"],
        "transformer.wpe.weight": sd["transformer.wpe.weight"],
        "transformer.ln_f.weight": sd["transformer.ln_f.weight"],
        "transformer.ln_f.bias": sd["transformer.ln_f.bias"],
        "lm_head.weight": sd["lm_head.weight"],
    }
    for i in range(mc["n_layer"]):
        p = f"transformer.h.{i}."
        new_sd[p + "ln_1.weight"] = sd[p + "ln_1.weight"]
        new_sd[p + "ln_1.bias"] = sd[p + "ln_1.bias"]
        new_sd[p + "ln_2.weight"] = sd[p + "ln_2.weight"]
        new_sd[p + "ln_2.bias"] = sd[p + "ln_2.bias"]
        # nn.Linear (out,in)  ->  HF Conv1D (in,out): transpose the weight.
        for w in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            new_sd[p + w + ".weight"] = sd[p + w + ".weight"].t().contiguous()
            new_sd[p + w + ".bias"] = sd[p + w + ".bias"]

    missing, unexpected = hf.load_state_dict(new_sd, strict=False)
    # The only acceptable "missing" keys are HF's auto-created causal-mask buffers.
    bad = [m for m in missing if not (m.endswith("attn.bias") or m.endswith("attn.masked_bias"))]
    assert not bad, f"unexpected missing params: {bad}"
    assert not unexpected, f"unexpected extra params: {unexpected}"

    hf.save_pretrained(out_dir, safe_serialization=True)
    print(f"  model: {sum(p.numel() for p in hf.parameters()):,} params "
          f"(GPT2, {mc['n_layer']}L/{mc['n_head']}H/{mc['n_embd']}d) -> {out_dir}")
    return hf, ckpt.get("val_loss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pretrained", "finetuned"], default="finetuned",
                    help="which checkpoint to export (default: finetuned)")
    args = ap.parse_args()

    os.makedirs(HF_DIR, exist_ok=True)
    tok = BPETokenizer.load(TOKENIZER_JSON)
    print(f"Exporting '{args.model}' checkpoint to HuggingFace GPT-2 format:")

    hf_tok = export_tokenizer(tok, HF_DIR)
    src = FINETUNED_CKPT if args.model == "finetuned" else PRETRAINED_CKPT
    _hf, val = export_model(src, HF_DIR, eos_id=tok.special_tokens["<|endoftext|>"])

    print(f"\nDone. HuggingFace model written to: {os.path.relpath(HF_DIR)}")
    print("Next: convert to GGUF, then create the Ollama model (see README.md).")


if __name__ == "__main__":
    main()
