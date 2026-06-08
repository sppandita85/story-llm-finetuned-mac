"""A decoder-only GPT transformer, hand-coded in PyTorch.

This is the same architecture family as GPT-2/3/4 and Llama: a stack of
transformer blocks, each doing (1) masked multi-head self-attention so every
token can look at the tokens *before* it, and (2) a position-wise MLP. The model
is trained to predict the next token, which is the entire objective behind LLM
pretraining.

Nothing here is toy-grade in *kind* -- only in *size*. The exact same module,
with bigger n_layer/n_embd/block_size and a real vocab, is what a trillion-token
run uses. Notable production-grade details included:
  - causal (look-ahead) masking
  - pre-LayerNorm blocks + residual connections (stable to train)
  - weight tying between the input embedding and the output projection
  - GPT-2-style scaled init for residual projections
  - top-k / temperature sampling for generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    All heads are computed in one batched matmul. The causal mask zeroes out
    attention to future positions so position t can only use positions <= t.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # One projection produces query, key, value for all heads at once.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular matrix used as the causal mask; not a parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        # reshape into (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                              # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # recombine heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, both residual."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # position embeddings
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: input embedding and output projection share weights.
        # Saves parameters and is standard in GPT-2.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 scaled init for residual projection layers.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Parameter count (excludes the tied position-embedding double count)."""
        n = sum(p.numel() for p in self.parameters())
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits, loss).

        If `targets` is given, cross-entropy next-token loss is computed. An
        optional `loss_mask` (B, T) of 0/1 lets fine-tuning ignore prompt tokens
        and only learn the answer tokens.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)      # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.reshape(-1),
                    ignore_index=-1,
                )
            else:
                # Per-token loss, then average only over unmasked (answer) tokens.
                per_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.reshape(-1),
                    ignore_index=-1, reduction="none",
                )
                m = loss_mask.reshape(-1).float()
                loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 stop_id: int | None = None) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` continuation tokens."""
        self.eval()
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx


def build_model(model_cfg, device: str = "cpu") -> GPT:
    """Construct a GPT from a common.config.ModelConfig-like object."""
    gcfg = GPTConfig(
        vocab_size=model_cfg.vocab_size,
        block_size=model_cfg.block_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        dropout=model_cfg.dropout,
        bias=model_cfg.bias,
    )
    return GPT(gcfg).to(device)
