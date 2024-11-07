"""One-minute live demo: open the dashboard, then ablate a feature live and watch the model change.

Uses the artifacts of a previous run (`make demo` / `python -m circuitscope all`).
"""
from __future__ import annotations

import json
import os
import time
import webbrowser

import torch

from . import pipeline as P
from .analysis import feature_stats, harvest
from .data import VOCAB, render, sample_dyck
from .interventions import ablation_sweep, single_ablation


def _pr(msg: str, pace: float) -> None:
    print(msg, flush=True)
    time.sleep(pace)


def run_demo(cfg, feature: int | None = None, pace: float = 4.0, open_browser: bool = True) -> None:
    out = cfg.out
    if not (out / "sae.pt").exists():
        raise SystemExit("No artifacts found - run `make demo` (or `python -m circuitscope all`) first.")
    layer = cfg.hook_layer
    model, sae = P.load_model(cfg), P.load_sae(cfg)

    _pr("== 1/5  Opening the static feature dashboard", pace)
    url = (out / "dashboard.html").resolve().as_uri()
    opened = open_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.name != "posix") and webbrowser.open(url)
    _pr(f"   {'opened in your browser: ' if opened else 'open this file in a browser: '}{url}", pace)

    batch = sample_dyck(cfg.n_eval_seqs, cfg.seq_len, 30_000 + cfg.seed)
    h = harvest(model, batch, layer)
    stats = feature_stats(sae, h)
    names = stats["label_names"]

    _pr("== 2/5  Picking a feature (structural ground-truth match with the largest causal effect)", pace)
    if feature is None:
        cand = [f for f in range(sae.n_features) if stats["count"][f] > 0 and stats["best_f1"][f] > 0.4
                and not names[int(stats["best_label"][f])].startswith("pos==")]
        sweep = ablation_sweep(model, sae, batch.tokens[:400], layer, cand)
        kl = torch.tensor(sweep["kl_active"]).nan_to_num(0)
        feature = cand[int(kl.argmax())]
    label = names[int(stats["best_label"][feature])]
    _pr(f"   feature {feature}: matches ground truth '{label}' with F1 {stats['best_f1'][feature]:.2f}, "
        f"fires on {100 * stats['density'][feature]:.1f}% of tokens", pace)

    _pr("== 3/5  Choosing a top-activating example from the dashboard", pace)
    acts = torch.stack([single_ablation(model, sae, batch.tokens[i], layer, feature)[0] for i in range(300)])
    i, p = divmod(int(acts.argmax()), acts.shape[1])
    seq = batch.tokens[i]
    _pr(f"   sequence {render(seq)}  (feature fires most at position {p}, activation {acts[i, p]:.2f})", pace)

    _pr("== 4/5  Ablating the feature LIVE (removing a * decoder direction from the residual stream)", pace)
    act, clean, abl = single_ablation(model, sae, seq, layer, feature)
    _, _, ctl = single_ablation(model, sae, seq, layer, feature, random_control=True)
    exact = batch.target[i]
    print(f"   position {p}, after prefix '{render(seq[:p + 1])}':", flush=True)
    print("   next token    exact   model   ablated   random-dir control", flush=True)
    for t, tok in enumerate(VOCAB):
        print(f"      {tok}          {exact[p, t]:.2f}    {clean[p, t]:.2f}    {abl[p, t]:.2f}      {ctl[p, t]:.2f}", flush=True)
    kl = lambda q: (clean[p] * (clean[p].clamp_min(1e-9).log() - q[p].clamp_min(1e-9).log())).sum().item()
    _pr(f"   KL(clean || ablated) = {kl(abl):.4f}   vs   random-direction control = {kl(ctl):.4f}", pace)
    moved = (abl[p] - clean[p]).abs().argmax().item()
    _pr(f"   largest shift: P('{VOCAB[moved]}') {clean[p, moved]:.2f} -> {abl[p, moved]:.2f}", pace)

    _pr("== 5/5  Whole-dataset check and patching result from the last run", pace)
    sweep = ablation_sweep(model, sae, batch.tokens[:600], layer, [feature])
    print(f"   over 600 sequences: mean KL when the feature fires = {sweep['kl_active'][0]:.4f}, "
          f"elsewhere (via attention) = {sweep['kl_inactive'][0]:.6f}, random control = {sweep['kl_control'][0]:.4f}", flush=True)
    res = json.loads((out / "results.json").read_text())["patching"]
    print(f"   patching 8 selected features '(' -> '[' flips {100 * res['selected'][3]:.0f}% of the closing-bracket logit gap "
          f"(8 random features: {100 * res['random_mean'][3]:.0f}%)", flush=True)
