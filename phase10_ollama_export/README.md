# Phase 10 — Host the model on Ollama

**Goal:** serve our from-scratch model through [Ollama](https://ollama.com).

**The catch:** Ollama cannot load a raw PyTorch `.pt` file — it runs **GGUF**
models through its bundled llama.cpp engine. So we convert. Our model was
deliberately built as a **GPT-2 architecture**, which llama.cpp/Ollama support,
so the conversion path is:

```
our finetuned.pt + custom BPE tokenizer
   │  export_hf.py        (transpose Linear→Conv1D weights; rebuild BPE as GPT-2 vocab/merges)
   ▼
HuggingFace GPT-2 format  (hf_model/: config.json, model.safetensors, tokenizer.json)
   │  convert_to_gguf.py  (wraps llama.cpp's convert_hf_to_gguf.py)
   ▼
story-llm.gguf
   │  ollama create -f Modelfile
   ▼
Ollama model "story-llm"
```

> ⚠️ Conversion does **not** improve quality. This 0.94M-param model trained on
> ~6K tokens produces garbled text; Ollama just serves it. The point is a
> working, real serving path.

## Quick start (one command)

From the project root, after the model is built (phases 1–8):

```bash
bash phase10_ollama_export/build_ollama.sh finetuned
ollama run story-llm "What is the moral of 'The Boy Who Cried Wolf'?"
```

That script installs the export-only deps, clones the converter, exports → GGUF,
and registers the Ollama model.

## Manual steps (what the script does)

```bash
# 0. export-only dependencies (kept separate from the core requirements.txt)
.venv/bin/pip install -r phase10_ollama_export/requirements-export.txt

# 1. clone llama.cpp (only the python converter is used)
git clone --depth 1 https://github.com/ggml-org/llama.cpp phase10_ollama_export/llama.cpp

# 2. our checkpoint -> HuggingFace GPT-2 format  (./hf_model/)
.venv/bin/python phase10_ollama_export/export_hf.py --model finetuned

# 3. HuggingFace -> GGUF  (./story-llm.gguf)
.venv/bin/python phase10_ollama_export/convert_to_gguf.py

# 4. register with Ollama
cd phase10_ollama_export && ollama create story-llm -f Modelfile

# 5. use it
ollama run story-llm "Tell me a story."
# or via the HTTP API:
curl http://localhost:11434/api/generate \
  -d '{"model":"story-llm","prompt":"What is the moral?","stream":false}'
```

## Files in this folder

| File | Purpose |
|------|---------|
| `export_hf.py` | converts our `.pt` + BPE tokenizer into HuggingFace GPT-2 format |
| `convert_to_gguf.py` | wraps llama.cpp's converter; registers our tokenizer's fingerprint as `gpt-2` |
| `Modelfile` | Ollama recipe (instruction template + stop tokens + sampling) |
| `build_ollama.sh` | runs the whole phase end-to-end |
| `requirements-export.txt` | export-only Python deps (pinned for torch 2.2) |
| `story-llm.gguf` | the converted model (committed; clone-and-run) |

`llama.cpp/` and `hf_model/` are git-ignored (re-created by the steps above).

## Two implementation details worth knowing

1. **Weight layout.** HuggingFace GPT-2 stores attention/MLP weights in `Conv1D`
   layout, which is the transpose of our `nn.Linear` weights. `export_hf.py`
   transposes `c_attn`, `c_proj`, `c_fc`, and `mlp.c_proj` accordingly.
2. **Tokenizer fingerprint.** llama.cpp refuses unknown BPE pre-tokenizers as a
   safety check. Ours is genuinely GPT-2 byte-level BPE (small custom vocab), so
   `convert_to_gguf.py` monkey-patches the fallback to return `"gpt-2"` instead
   of editing the vendored llama.cpp source.
