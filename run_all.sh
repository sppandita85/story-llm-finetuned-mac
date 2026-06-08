#!/usr/bin/env bash
# Run the full build pipeline, phase 1 -> 8, using the project-local venv.
# Phase 9 (chat) is interactive, so it is not run here -- launch it yourself
# after this finishes:  .venv/bin/python phase9_chat/run.py "your question"
set -e

cd "$(dirname "$0")"
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "No .venv found. Create it first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "=========================================="
echo " Phase 1 - Data preparation"
echo "=========================================="
$PY phase1_data_prep/run.py

echo; echo "=========================================="
echo " Phase 2 - Tokenizer training (BPE)"
echo "=========================================="
$PY phase2_tokenizer/run.py

echo; echo "=========================================="
echo " Phase 3 - Encoding & sharding"
echo "=========================================="
$PY phase3_encoding/run.py

echo; echo "=========================================="
echo " Phase 4 - Model sanity check"
echo "=========================================="
$PY phase4_model/run.py

echo; echo "=========================================="
echo " Phase 5 - Pretraining (this is the long one)"
echo "=========================================="
$PY phase5_pretrain/run.py

echo; echo "=========================================="
echo " Phase 6 - Base-model generation"
echo "=========================================="
$PY phase6_generate/run.py

echo; echo "=========================================="
echo " Phase 7 - Instruction (Q&A) dataset"
echo "=========================================="
$PY phase7_qa_dataset/run.py

echo; echo "=========================================="
echo " Phase 8 - Supervised fine-tuning"
echo "=========================================="
$PY phase8_finetune/run.py

echo; echo "All phases complete."
echo "Try the fine-tuned model:"
echo "  $PY phase9_chat/run.py \"What is the moral of 'The Boy Who Cried Wolf'?\""
