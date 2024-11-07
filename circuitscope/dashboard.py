"""Static HTML feature dashboard (no JS, no external assets) and markdown report."""
from __future__ import annotations

import html
from pathlib import Path

import torch

from .data import VOCAB
from .analysis import Harvest

CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;color:#1d2433;background:#fafbfc}
h1{margin:0 0 4px} .sub{color:#5b6475;margin-bottom:18px}
table{border-collapse:collapse;margin:8px 0 20px} td,th{padding:3px 12px 3px 0;text-align:left;font-variant-numeric:tabular-nums}
.card{background:#fff;border:1px solid #d9dee7;border-radius:8px;padding:12px 16px;margin:12px 0}
.card h3{margin:0 0 4px;font-size:15px} .meta{color:#5b6475;font-size:12.5px}
.ex{font-family:ui-monospace,Menlo,monospace;font-size:13px;white-space:pre;margin:2px 0}
.ex b{background:#ffd54a;color:#000;border-radius:2px} .ex i{color:#8a93a5;font-style:normal;font-size:11.5px}
.bars{display:inline-block;vertical-align:top;margin-left:24px} nav a{margin-right:10px}
"""


def _bars(values: list[float], labels: list[str], w=220, h=46) -> str:
    top = max(values) if max(values) > 0 else 1.0
    bw = w / len(values)
    rects = "".join(
        f'<rect x="{i * bw:.1f}" y="{h - 34 * v / top:.1f}" width="{bw - 1:.1f}" height="{34 * v / top:.1f}" fill="#3a6df0"><title>{html.escape(l)}: {v:.3f}</title></rect>'
        for i, (v, l) in enumerate(zip(values, labels)))
    return f'<svg width="{w}" height="{h}">{rects}<text x="0" y="{h - 1}" font-size="9" fill="#8a93a5">mean activation by depth 0..{len(values) - 1}</text></svg>'


def _example(h: Harvest, full: torch.Tensor, flat_i: int, value: float, ctx: int = 12) -> str:
    s, p = int(h.seq[flat_i]), int(h.pos[flat_i])
    seq = full[s]
    lo = max(0, p - ctx)
    left = "".join(VOCAB[int(t)] for t in seq[lo:p])
    cur = VOCAB[int(seq[p])]
    right = "".join(VOCAB[int(t)] for t in seq[p + 1:min(len(seq), p + 4)])
    pre = "…" if lo else ""
    return (f'<div class="ex">{pre}{html.escape(left)}<b>{html.escape(cur)}</b>{html.escape(right)}'
            f'   <i>act {value:.2f} · pos {p} · depth {int(h.depth[flat_i])} · tos {["(", "[", "empty"][int(h.tos[flat_i])]}</i></div>')


def write_dashboard(path: Path, sae_k: int, stats: dict, report: dict, h: Harvest, full: torch.Tensor,
                    top_vals: torch.Tensor, top_idx: torch.Tensor, max_features: int = 120) -> None:
    names = stats["label_names"]
    alive = (stats["count"] > 0).nonzero().flatten()
    # structural (non-positional) matches first: position-locked features are the least surprising
    positional = torch.tensor([names[int(stats["best_label"][f])].startswith("pos==") for f in alive.tolist()]).float()
    order = alive[(stats["best_f1"][alive] - positional).argsort(descending=True)][:max_features]
    depth_mean = _mean_by_depth(stats, h)
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.4g}</td></tr>" if isinstance(v, float)
                   else f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in report.items())
    cards = []
    for f in order.tolist():
        ex = "".join(_example(h, full, int(i), float(v)) for v, i in zip(top_vals[f], top_idx[f]) if i >= 0)
        label = names[int(stats["best_label"][f])]
        cards.append(
            f'<div class="card" id="f{f}"><h3>Feature {f} — best ground-truth match: <code>{html.escape(label)}</code> '
            f'(F1 {stats["best_f1"][f]:.2f})</h3><div class="meta">density {stats["density"][f]:.4f} · '
            f'max act {stats["max_act"][f]:.2f} · precision {stats["best_precision"][f]:.2f} · recall {stats["best_recall"][f]:.2f}</div>'
            f'{ex}<div class="bars">{_bars(depth_mean[f].tolist(), [f"depth {d}" for d in range(depth_mean.shape[1])])}</div></div>')
    dead = (stats["count"] == 0).nonzero().flatten().tolist()
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><title>CircuitScope dashboard</title><style>{CSS}</style></head><body>'
           f'<h1>CircuitScope feature dashboard</h1><div class="sub">Top-{sae_k} SAE on the residual stream of a Dyck-2 transformer. '
           f'Features sorted by how well they match a ground-truth variable (F1), depth/stack-type matches before position matches. Showing {len(order)} of {len(alive)} alive features.</div>'
           f'<h2>Summary</h2><table>{rows}</table>'
           f'<h2>Dead features ({len(dead)})</h2><div class="meta">{", ".join(map(str, dead[:200])) or "none"}</div>'
           f'<h2>Features</h2>{"".join(cards)}</body></html>')
    path.write_text(doc, encoding="utf-8")


def _mean_by_depth(stats: dict, h: Harvest) -> torch.Tensor:
    F = stats["density"].shape[0]
    D = int(h.depth.max()) + 1
    k = stats["idx"].shape[1]
    tot = torch.zeros(F * D)
    cnt = torch.zeros(D).index_add_(0, h.depth, torch.ones_like(h.depth, dtype=torch.float))
    key = stats["idx"] * D + h.depth.unsqueeze(1)
    tot.index_add_(0, key.flatten(), stats["vals"].flatten())
    return tot.view(F, D) / cnt.clamp_min(1)


def write_report_md(path: Path, model_metrics: dict, report: dict, fit, ablation_summary: dict, patching: dict) -> None:
    lines = ["# CircuitScope run report", "", "## Transformer", "",
             f"- validation KL to exact next-token distribution: {model_metrics['val_kl']:.5f}",
             f"- argmax is a legal continuation: {model_metrics['val_valid_accuracy']:.4f}", "",
             "## Sparse autoencoder", ""]
    lines += [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in report.items()]
    lines += [f"- training time: {fit.seconds:.1f}s, features resampled: {fit.resampled}", "",
              "## Causal tests", ""]
    lines += [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in ablation_summary.items()]
    lines += ["", "Patching (fraction of closing-bracket logit-difference flipped):", "",
              "| #features | selected | random (mean ± std) |", "|---|---|---|"]
    for m, s, r, sd in zip(patching["sizes"], patching["selected"], patching["random_mean"], patching["random_std"]):
        lines.append(f"| {m} | {s:.3f} | {r:.3f} ± {sd:.3f} |")
    lines.append(f"| all | {patching['all_features']:.3f} | – |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_preview(path: Path, stats: dict, h: Harvest, full: torch.Tensor, top_vals: torch.Tensor,
                   top_idx: torch.Tensor, n_cards: int = 3) -> None:
    """PNG rendering of the first dashboard cards (same data and ordering as dashboard.html), for docs
    and for environments with no browser to screenshot with."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = stats["label_names"]
    alive = (stats["count"] > 0).nonzero().flatten()
    positional = torch.tensor([names[int(stats["best_label"][f])].startswith("pos==") for f in alive.tolist()]).float()
    order = alive[(stats["best_f1"][alive] - positional).argsort(descending=True)][:n_cards].tolist()
    depth_mean = _mean_by_depth(stats, h)
    fig = plt.figure(figsize=(11, 2.9 * n_cards + 0.8), facecolor="#fafbfc")
    fig.text(0.02, 1 - 0.45 / fig.get_figheight(), "CircuitScope feature dashboard (rendered preview)", fontsize=15, weight="bold", va="center")
    for r, f in enumerate(order):
        top = 1 - (0.9 + 2.9 * r) / fig.get_figheight()
        card = fig.add_axes([0.02, top - 2.65 / fig.get_figheight(), 0.96, 2.65 / fig.get_figheight()])
        card.set_xticks([]); card.set_yticks([])
        card.set_facecolor("white")
        card.text(0.015, 0.93, f"Feature {f} - best ground-truth match: {names[int(stats['best_label'][f])]}  (F1 {stats['best_f1'][f]:.2f})",
                  fontsize=11, weight="bold", va="top", transform=card.transAxes)
        card.text(0.015, 0.80, f"density {stats['density'][f]:.4f}   max act {stats['max_act'][f]:.2f}   "
                  f"precision {stats['best_precision'][f]:.2f}   recall {stats['best_recall'][f]:.2f}", fontsize=9, color="#5b6475", va="top", transform=card.transAxes)
        y = 0.66
        for v, i in list(zip(top_vals[f], top_idx[f]))[:5]:
            if i < 0:
                continue
            s_, p_ = int(h.seq[i]), int(h.pos[i])
            seq = "".join(VOCAB[int(t)] for t in full[s_][:p_ + 4])
            lo = max(0, p_ - 12)
            left, cur, right = seq[lo:p_], seq[p_], seq[p_ + 1:p_ + 4]
            x = 0.015
            for txt, hl in ((("..." if lo else "") + left, False), (cur, True), (right, False)):
                t = card.text(x, y, txt, family="monospace", fontsize=10, va="top", transform=card.transAxes,
                              bbox=dict(facecolor="#ffd54a", edgecolor="none", pad=1) if hl else None)
                x += 0.0069 * len(txt) + (0.002 if hl else 0)
            card.text(0.31, y, f"act {v:.2f} | pos {p_} | depth {int(h.depth[i])} | tos {['(', '[', 'empty'][int(h.tos[i])]}",
                      fontsize=8.5, color="#8a93a5", va="top", transform=card.transAxes)
            y -= 0.125
        bars = fig.add_axes([0.66, top - 2.0 / fig.get_figheight(), 0.30, 1.1 / fig.get_figheight()])
        bars.bar(range(depth_mean.shape[1]), depth_mean[f].numpy(), color="#3a6df0")
        bars.set_title("mean activation by stack depth", fontsize=8, color="#5b6475")
        bars.tick_params(labelsize=7)
        for sp in ("top", "right"):
            bars.spines[sp].set_visible(False)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
