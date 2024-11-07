"""`python -m circuitscope <command>`"""
from __future__ import annotations

import argparse
import json

from .config import Config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="circuitscope", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("all", "train model + SAE, analyse, run causal tests"), ("model", "train the transformer"),
                        ("sae", "train the SAE (needs model.pt)"), ("analyze", "dashboard, report, causal tests (needs model.pt, sae.pt)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--backend", choices=["cpp", "torch"], help="SAE kernel backend (default: $CS_SAE_BACKEND or cpp)")
        p.add_argument("--out", help="artifact directory")
        p.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    d = sub.add_parser("demo", help="1-minute live demo: open the dashboard and ablate a feature (needs a previous `all` run)")
    d.add_argument("--out")
    d.add_argument("--feature", type=int, help="feature id (default: best structural feature by causal effect)")
    d.add_argument("--pace", type=float, default=4.0, help="seconds to pause between demo steps")
    d.add_argument("--no-open", action="store_true", help="do not try to open a browser")
    b = sub.add_parser("bench", help="benchmark C++ kernels vs PyTorch")
    b.add_argument("--out")
    b.add_argument("--iters", type=int, default=7)
    b.add_argument("--small", action="store_true", help="only the two smallest shapes")
    a = ap.parse_args(argv)

    cfg = Config.from_env(sae_backend=getattr(a, "backend", None), out_dir=getattr(a, "out", None))
    if getattr(a, "quick", False):
        cfg.model_steps, cfg.sae_steps, cfg.n_sae_seqs, cfg.n_eval_seqs = 150, 200, 1500, 600

    from . import pipeline as P

    if a.cmd == "demo":
        from .demo import run_demo
        run_demo(cfg, a.feature, a.pace, not a.no_open)
        return

    if a.cmd == "bench":
        from .benchmark import DEFAULT_SHAPES, format_table, plot_benchmark, run_benchmark
        rows = run_benchmark(DEFAULT_SHAPES[:2] if a.small else None, a.iters)
        table = format_table(rows)
        (cfg.out / "benchmark.json").write_text(json.dumps(rows, indent=1))
        (cfg.out / "benchmark.md").write_text(table + "\n")
        plot_benchmark(rows, cfg.out / "benchmark.png")
        print(table)
        return

    if a.cmd in ("all", "model"):
        model, mm = P.stage_model(cfg)
    else:
        model, mm = P.load_model(cfg), json.loads((cfg.out / "model_metrics.json").read_text())
    if a.cmd == "model":
        return
    if a.cmd in ("all", "sae"):
        sae, fit = P.stage_sae(cfg, model)
    else:
        sae = P.load_sae(cfg)
        d = json.loads((cfg.out / "sae_fit.json").read_text())
        from .sae import SAEFitResult
        fit = SAEFitResult([tuple(x) for x in d["history"]], d["resampled"], d["seconds"])
    if a.cmd == "sae":
        return
    P.stage_analyze(cfg, model, sae, fit, mm)
    print(f"\nDone. Open {cfg.out / 'dashboard.html'}; summary in {cfg.out / 'report.md'}; plots in {cfg.out / 'plots'}")
