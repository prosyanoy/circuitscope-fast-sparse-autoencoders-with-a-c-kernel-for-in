"""A small pre-LN causal transformer with explicit residual-stream access points."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    """Residual points: 0 = embeddings, i = output of block i (n_layers = last, pre-unembed)."""

    def __init__(self, vocab: int = 4, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, max_len: int = 64):
        super().__init__()
        self.cfg = dict(vocab=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=max_len)
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab)
        nn.init.normal_(self.tok.weight, std=0.5)
        nn.init.normal_(self.pos.weight, std=0.5)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def resid(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual stream after `layer` blocks, [B, T, d]."""
        x = self.tok(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        for blk in self.blocks[:layer]:
            x = blk(x)
        return x

    def forward_from(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Logits obtained by running blocks `layer..` on a (possibly edited) residual stream."""
        for blk in self.blocks[layer:]:
            x = blk(x)
        return self.unembed(self.ln_f(x))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_from(self.resid(tokens, 0), 0)


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, -1)).sum(-1).mean()


def kl_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean KL(target || softmax(logits)) per position."""
    logp = F.log_softmax(logits, -1)
    return (target * (torch.log(target.clamp_min(1e-12)) - logp)).sum(-1).mean()


def valid_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of positions where the argmax prediction is a legal continuation."""
    pred = logits.argmax(-1, keepdim=True)
    return (target.gather(-1, pred) > 0).float().mean().item()


def train_transformer(cfg, log=print) -> tuple[TinyTransformer, dict]:
    from .data import sample_dyck

    torch.manual_seed(cfg.seed)
    model = TinyTransformer(d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=cfg.model_steps, pct_start=0.1)
    rng = __import__("numpy").random.default_rng(cfg.seed + 1)
    history = []
    for step in range(cfg.model_steps):
        b = sample_dyck(cfg.model_batch, cfg.seq_len, rng)
        loss = soft_ce(model(b.tokens), b.target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == cfg.model_steps - 1:
            history.append((step, loss.item()))
        if step % 500 == 0:
            log(f"[model] step {step:5d} loss {loss.item():.4f}")
    val = sample_dyck(2000, cfg.seq_len, 10_000 + cfg.seed)
    with torch.no_grad():
        logits = model(val.tokens)
    metrics = {
        "val_kl": kl_to_target(logits, val.target).item(),
        "val_valid_accuracy": valid_accuracy(logits, val.target),
        "history": history,
    }
    log(f"[model] val KL {metrics['val_kl']:.5f}, valid-argmax accuracy {metrics['val_valid_accuracy']:.4f}")
    return model.eval(), metrics
