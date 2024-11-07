"""Kernel benchmark: speed and peak memory of the C++ path vs. the pure PyTorch path.

Each (backend, shape) is measured in a fresh subprocess so that peak RSS (ru_maxrss) is not
polluted by earlier measurements. The measured unit is one SAE training step's hot path:
    pre -> top-k -> decode -> loss -> backward(through decode + top-k)
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

import torch

from . import kernels

DEFAULT_SHAPES = [  # (batch, n_features, k, d_model)
    (4096, 1024, 8, 64),
    (4096, 4096, 16, 128),
    (4096, 16384, 32, 256),
    (2048, 32768, 64, 512),
]


def _child(backend: str, B: int, F: int, k: int, d: int, iters: int) -> dict:
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(F, d) / d ** 0.5)
    bias = torch.nn.Parameter(torch.zeros(d))
    pre0 = torch.randn(B, F)
    target = torch.randn(B, d)
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB, includes inputs + imports

    def step():
        pre = pre0.clone().requires_grad_(True)
        vals, idx = kernels.topk(pre, k, backend)
        out = kernels.sparse_decode(torch.relu(vals), idx, W, bias, backend)
        loss = (out - target).pow(2).sum(-1).mean()
        loss.backward()
        return vals, idx, out

    def t_topk():
        kernels.topk(pre0, k, backend)

    vals, idx = kernels.topk(pre0, k, "torch")
    vals = torch.relu(vals)

    def t_decode():
        kernels.sparse_decode(vals, idx, W.detach(), bias.detach(), backend)

    def time_it(fn):
        fn()
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2] * 1e3

    res = {"topk_ms": time_it(t_topk), "decode_ms": time_it(t_decode), "step_ms": time_it(step)}
    res["peak_extra_mb"] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base_rss) / 1024
    res["checksum"] = float(step()[2].sum())
    return res


def run_benchmark(shapes=None, iters: int = 7) -> list[dict]:
    rows = []
    for (B, F, k, d) in shapes or DEFAULT_SHAPES:
        row = {"B": B, "F": F, "k": k, "d": d}
        for backend in ("torch", "cpp"):
            out = subprocess.run([sys.executable, "-m", "circuitscope.benchmark", backend, str(B), str(F), str(k), str(d), str(iters)],
                                 capture_output=True, text=True, check=True)
            row[backend] = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    head = "| B | F | k | d | topk torch/cpp (ms) | decode torch/cpp (ms) | step torch/cpp (ms) | step speedup | peak extra MB torch/cpp |\n|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for r in rows:
        t, c = r["torch"], r["cpp"]
        lines.append(f"| {r['B']} | {r['F']} | {r['k']} | {r['d']} | {t['topk_ms']:.2f} / {c['topk_ms']:.2f} | "
                     f"{t['decode_ms']:.2f} / {c['decode_ms']:.2f} | {t['step_ms']:.1f} / {c['step_ms']:.1f} | "
                     f"{t['step_ms'] / c['step_ms']:.2f}x | {t['peak_extra_mb']:.0f} / {c['peak_extra_mb']:.0f} |")
    return "\n".join(lines)


def plot_benchmark(rows: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"F={r['F']}\nk={r['k']}" for r in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ("topk_ms", "decode_ms", "step_ms"), ("top-k", "sparse decode", "fwd+bwd step")):
        ax.bar([i - 0.2 for i in x], [r["torch"][key] for r in rows], 0.4, label="PyTorch", color="#8a93a5")
        ax.bar([i + 0.2 for i in x], [r["cpp"][key] for r in rows], 0.4, label="C++/OpenMP", color="#3a6df0")
        ax.set_xticks(list(x), labels, fontsize=8)
        ax.set_title(f"{title} (ms, lower is better)", fontsize=10)
        ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Kernel benchmark ({kernels.max_threads()} OpenMP thread(s))", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    b, *nums = sys.argv[1:]
    print(json.dumps(_child(b, *map(int, nums))))
