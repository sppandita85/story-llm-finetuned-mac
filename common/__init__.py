"""Shared, importable modules for the from-scratch LLM build.

This package holds the three pieces that more than one phase needs:
  - config.py        : every hyperparameter and file path in one place
  - bpe_tokenizer.py : the custom Byte-Pair-Encoding tokenizer
  - model.py         : the decoder-only GPT transformer

Keeping them here means each phase's run.py imports the *same* code, so the
tokenizer used to encode the data is identical to the one used at inference,
and the model architecture used for pre-training is identical to the one used
for fine-tuning. This mirrors a real LLM codebase where the model/tokenizer
definitions are a library shared by every pipeline stage.
"""
