"""Top-k and sparse-decode ops with two interchangeable backends.

* ``torch``: the reference path (``torch.topk`` + dense scatter + dense matmul).
* ``cpp``  : the OpenMP extension in ``csrc/kernels.cpp`` (heap top-k, sparse decode).

Both are exposed as autograd Functions so the SAE trainer can swap backends with a flag.
"""
from __future__ import annotations

import torch

try:  # the extension is optional: everything works (slower) without it
    from . import _kernels
except ImportError:  # pragma: no cover - depends on local build
    _kernels = None


def available() -> bool:
    return _kernels is not None


def max_threads() -> int:
    return _kernels.max_threads() if _kernels is not None else torch.get_num_threads()


def _require_cpp():
    if _kernels is None:
        raise RuntimeError("C++ kernels not built; run `make build` (needs cmake, a C++17 compiler, OpenMP)")
    return _kernels


def _f32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32 or x.device.type != "cpu":
        raise TypeError("C++ kernels need float32 CPU tensors")
    return x.contiguous()


# ----------------------------------------------------------------------------- top-k
def topk_cpp_raw(pre: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pre = _f32(pre)
    vals = torch.empty(pre.shape[0], k, dtype=torch.float32)
    idx = torch.empty(pre.shape[0], k, dtype=torch.int64)
    _require_cpp().topk_rows(pre.numpy(), k, vals.numpy(), idx.numpy())
    return vals, idx


class _TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, k, backend):
        if backend == "cpp":
            vals, idx = topk_cpp_raw(pre, k)
        else:
            vals, idx = torch.topk(pre, k, dim=1, sorted=True)
        ctx.save_for_backward(idx)
        ctx.n_features = pre.shape[1]
        ctx.mark_non_differentiable(idx)
        return vals, idx

    @staticmethod
    def backward(ctx, gvals, _gidx):
        (idx,) = ctx.saved_tensors
        g = torch.zeros(idx.shape[0], ctx.n_features, dtype=gvals.dtype)
        return g.scatter_(1, idx, gvals), None, None


def topk(pre: torch.Tensor, k: int, backend: str = "torch") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row top-k values (descending) and their feature indices."""
    return _TopK.apply(pre, k, backend)


# ----------------------------------------------------------------------------- decode
class _SparseDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vals, idx, W, bias):
        vals, W, bias = _f32(vals), _f32(W), _f32(bias)
        idx = idx.contiguous()
        out = torch.empty(vals.shape[0], W.shape[1], dtype=torch.float32)
        _require_cpp().sparse_decode(vals.numpy(), idx.numpy(), W.numpy(), bias.numpy(), out.numpy())
        ctx.save_for_backward(vals, idx, W)
        return out

    @staticmethod
    def backward(ctx, g):
        vals, idx, W = ctx.saved_tensors
        g = _f32(g)
        gvals = torch.empty_like(vals)
        gW = torch.zeros_like(W)
        _require_cpp().sparse_decode_backward(g.numpy(), vals.numpy(), idx.numpy(), W.numpy(), gvals.numpy(), gW.numpy())
        return gvals, None, gW, g.sum(0)


def sparse_decode(vals, idx, W, bias, backend: str = "torch") -> torch.Tensor:
    """bias + sum_j vals[:, j] * W[idx[:, j]]."""
    if backend == "cpp":
        return _SparseDecode.apply(vals, idx, W, bias)
    dense = torch.zeros(vals.shape[0], W.shape[0], dtype=vals.dtype).scatter(1, idx, vals)
    return dense @ W + bias


def resolve_backend(requested: str) -> str:
    """'cpp' degrades gracefully to 'torch' when the extension is missing."""
    if requested not in ("cpp", "torch"):
        raise ValueError(f"unknown backend {requested!r}")
    return "cpp" if requested == "cpp" and available() else "torch"
