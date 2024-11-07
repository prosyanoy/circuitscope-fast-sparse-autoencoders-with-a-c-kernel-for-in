"""Dyck-2 task with exact ground truth.

Sequences are balanced bracket strings of fixed length L over  ( [ ) ]  (tokens 0..3). They are
sampled from a known generative policy, so for every prefix the *exact* next-token distribution
is available (it depends only on stack depth, remaining length and the top-of-stack bracket
type). That gives us:
  * soft training targets and an exact KL metric for the transformer, and
  * ground-truth latent variables (depth, top-of-stack type, position) to score SAE features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

VOCAB = ["(", "[", ")", "]"]
N_TOK = 4
EMPTY = 2  # "top of stack" label when the stack is empty


@dataclass
class DyckBatch:
    tokens: torch.Tensor  # [n, L-1] model inputs
    target: torch.Tensor  # [n, L-1, 4] exact next-token distributions
    depth: torch.Tensor  # [n, L-1] stack depth after consuming the token
    tos: torch.Tensor  # [n, L-1] top-of-stack type after the token: 0 '(' 1 '[' 2 empty
    full: torch.Tensor  # [n, L] the complete sequences


def sample_dyck(n: int, seq_len: int, seed: int | torch.Generator | np.random.Generator = 0) -> DyckBatch:
    if seq_len < 4 or seq_len % 2:
        raise ValueError("seq_len must be an even number >= 4")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    L = seq_len
    toks = np.zeros((n, L), dtype=np.int64)
    depth = np.zeros((n, L), dtype=np.int64)
    tos = np.full((n, L), EMPTY, dtype=np.int64)
    dist = np.zeros((n, L, N_TOK), dtype=np.float32)
    stack = np.zeros((n, L + 1), dtype=np.int64)
    d = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for s in range(L):
        remaining = L - s  # positions left to fill, including this one
        can_open = d + 1 <= remaining - 1
        can_close = d > 0
        top = stack[rows, np.maximum(d - 1, 0)]
        p = np.zeros((n, N_TOK), dtype=np.float32)
        both = can_open & can_close
        only_open = can_open & ~can_close
        only_close = ~can_open & can_close
        p[both, 0] = p[both, 1] = 0.25
        p[only_open, 0] = p[only_open, 1] = 0.5
        p[both, 2 + top[both]] = 0.5
        p[only_close, 2 + top[only_close]] = 1.0
        dist[:, s] = p
        cum = p.cumsum(1)
        tok = (rng.random(n)[:, None] > cum).sum(1).clip(0, N_TOK - 1)
        toks[:, s] = tok
        opened = tok < 2
        stack[rows[opened], d[opened]] = tok[opened]
        d = d + np.where(opened, 1, -1)
        depth[:, s] = d
        tos[:, s] = np.where(d > 0, stack[rows, np.maximum(d - 1, 0)], EMPTY)
    t = torch.from_numpy
    return DyckBatch(t(toks[:, :-1]), t(dist[:, 1:]), t(depth[:, :-1]), t(tos[:, :-1]), t(toks))


def is_balanced(seq) -> bool:
    stack = []
    for tok in map(int, seq):
        if tok < 2:
            stack.append(tok)
        elif not stack or stack.pop() != tok - 2:
            return False
    return not stack


def render(seq) -> str:
    return "".join(VOCAB[int(t)] for t in seq)
