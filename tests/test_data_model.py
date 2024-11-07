import numpy as np
import torch

from circuitscope.data import EMPTY, is_balanced, sample_dyck
from circuitscope.model import TinyTransformer, kl_to_target, valid_accuracy


def test_sequences_balanced_and_targets_exact():
    b = sample_dyck(300, 16, 0)
    assert all(is_balanced(s) for s in b.full)
    assert torch.allclose(b.target.sum(-1), torch.ones(300, 15))
    # the sampled next token always has non-zero probability under the exact target
    nxt = b.full[:, 1:]
    assert (b.target.gather(-1, nxt.unsqueeze(-1)) > 0).all()


def test_labels_consistent_with_sequence():
    b = sample_dyck(50, 12, 3)
    for s, dep, tos in zip(b.full, b.depth, b.tos):
        stack = []
        for t, token in enumerate(s[:-1].tolist()):
            stack.append(token) if token < 2 else stack.pop()
            assert dep[t] == len(stack)
            assert tos[t] == (stack[-1] if stack else EMPTY)


def test_forced_moves():
    b = sample_dyck(200, 10, 1)
    empty = b.depth == 0
    assert (b.target[empty][:, 2:] == 0).all()  # cannot close an empty stack
    assert (b.target[:, -1, :2] == 0).all()  # last predicted token must close


def test_residual_split_is_exact():
    torch.manual_seed(0)
    m = TinyTransformer(d_model=32, n_layers=3, n_heads=2).eval()
    toks = torch.randint(0, 4, (4, 9))
    full = m(toks)
    for layer in range(4):
        assert torch.allclose(m.forward_from(m.resid(toks, layer), layer), full, atol=1e-5)


def test_metrics():
    target = torch.tensor([[[0.5, 0.5, 0, 0]]])
    good = torch.tensor([[[5.0, 5.0, -9, -9]]])
    bad = torch.tensor([[[-9.0, -9, 5, 5]]])
    assert kl_to_target(good, target) < 1e-3 < kl_to_target(bad, target)
    assert valid_accuracy(good, target) == 1.0 and valid_accuracy(bad, target) == 0.0
