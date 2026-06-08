#!/usr/bin/env bash
# Phase 10 -- one-shot: export our model, convert to GGUF, register in Ollama.
# Run from the project root:  bash phase10_ollama_export/build_ollama.sh [pretrained|finetuned]
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$ROOT/phase10_ollama_export"
PY="$ROOT/.venv/bin/python"
MODEL="${1:-finetuned}"

echo ">>> [1/5] install export-only deps into .venv"
"$ROOT/.venv/bin/pip" install -q -r "$HERE/requirements-export.txt"

echo ">>> [2/5] clone llama.cpp converter (if missing)"
if [ ! -d "$HERE/llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HERE/llama.cpp"
fi

echo ">>> [3/5] export $MODEL checkpoint -> HuggingFace GPT-2 format"
"$PY" "$HERE/export_hf.py" --model "$MODEL"

echo ">>> [4/5] convert HuggingFace model -> GGUF"
"$PY" "$HERE/convert_to_gguf.py"

echo ">>> [5/5] create the Ollama model 'story-llm'"
( cd "$HERE" && ollama create story-llm -f Modelfile )

echo
echo "Done. Try it:"
echo "  ollama run story-llm \"What is the moral of 'The Boy Who Cried Wolf'?\""
