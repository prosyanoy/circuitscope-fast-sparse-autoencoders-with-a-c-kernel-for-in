"""End-to-end stages; each reads/writes artifacts in cfg.out so stages can be run separately."""
from __future__ import annotations

import json

import torch

from . import kernels
from .analysis import feature_stats, harvest, reconstruction_report, recovery_rate, top_examples
from .config import Config
from .dashboard import render_preview, write_dashboard, write_report_md
from .data import sample_dyck
from .interventions import ablation_sweep, patching_experiment
from .model import TinyTransformer, train_transformer
from .sae import TopKSAE, fit_sae


def stage_model(cfg: Config, log=print):
    model, metrics = train_transformer(cfg, log)
    torch.save({"state": model.state_dict(), "cfg": model.cfg}, cfg.out / "model.pt")
    (cfg.out / "model_metrics.json").write_text(json.dumps(metrics))
    return model, metrics


def load_model(cfg: Config) -> TinyTransformer:
    ck = torch.load(cfg.out / "model.pt")
    model = TinyTransformer(**ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.eval()


def stage_sae(cfg: Config, model, log=print):
    torch.manual_seed(cfg.seed)
    batch = sample_dyck(cfg.n_sae_seqs, cfg.seq_len, 20_000 + cfg.seed)
    acts = harvest(model, batch, cfg.hook_layer).acts
    sae = TopKSAE(acts.shape[1], cfg.n_features, cfg.k, cfg.sae_backend, cfg.seed)
    log(f"[sae] {acts.shape[0]} activation vectors, backend={sae.backend}, threads={kernels.max_threads()}")
    fit = fit_sae(sae, acts, cfg.sae_steps, cfg.sae_batch, cfg.sae_lr, seed=cfg.seed, log=log)
    torch.save({"state": sae.state_dict(), "shape": (acts.shape[1], cfg.n_features, cfg.k)}, cfg.out / "sae.pt")
    (cfg.out / "sae_fit.json").write_text(json.dumps({"history": fit.history, "seconds": fit.seconds, "resampled": fit.resampled}))
    return sae, fit


def load_sae(cfg: Config) -> TopKSAE:
    ck = torch.load(cfg.out / "sae.pt")
    sae = TopKSAE(*ck["shape"], backend=cfg.sae_backend)
    sae.load_state_dict(ck["state"])
    return sae


def stage_analyze(cfg: Config, model, sae, fit, model_metrics, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer = cfg.hook_layer
    (cfg.out / "plots").mkdir(exist_ok=True)
    batch = sample_dyck(cfg.n_eval_seqs, cfg.seq_len, 30_000 + cfg.seed)
    h = harvest(model, batch, layer)
    stats = feature_stats(sae, h)
    report = reconstruction_report(model, sae, batch, h, layer, stats)
    report.update(recovery_rate(stats))
    top_v, top_i = top_examples(stats)
    render_preview(cfg.out / "plots" / "dashboard_preview.png", stats, h, batch.full, top_v, top_i)
    write_dashboard(cfg.out / "dashboard.html", sae.k, stats, report, h, batch.full, top_v, top_i)
    log(f"[analyze] dead={report['dead_features']} F1>0.9: {report['features_f1_gt_0.9']} VE={report['variance_explained']:.3f}")

    # --- causal tests -------------------------------------------------------------------
    alive = (stats["count"] > 0).nonzero().flatten()
    log(f"[causal] ablation sweep over {len(alive)} alive features")
    sweep_tokens = batch.tokens[:1000]
    sweep = ablation_sweep(model, sae, sweep_tokens, layer, alive.tolist())
    kl_a = torch.tensor(sweep["kl_active"]); kl_c = torch.tensor(sweep["kl_control"])
    ok = ~(kl_a.isnan() | kl_c.isnan())
    matched = stats["best_f1"][alive] > 0.7
    dlogp = sweep["dlogp"]  # [m, vocab]
    # ground-truth check: a feature labelled tos=='(' / tos=='[' should mainly move the matching ')' / ']' logit
    names = stats["label_names"]
    tos_hits, tos_total = 0, 0
    for j, f in enumerate(alive.tolist()):
        lab = names[int(stats["best_label"][f])]
        if stats["best_f1"][f] > 0.4 and lab in ("tos=='('", "tos=='['"):
            tos_total += 1
            expected = 2 if lab == "tos=='('" else 3
            tos_hits += int(dlogp[j].argmin().item() == expected)
    ablation_summary = {
        "ablation_features_tested": int(len(alive)),
        "mean_kl_ablate_feature": kl_a[ok].mean().item(),
        "mean_kl_ablate_random_direction": kl_c[ok].mean().item(),
        "features_where_ablation_beats_control": int((kl_a[ok] > 2 * kl_c[ok]).sum()),
        "ground_truth_matched_features(F1>0.7)": int(matched.sum()),
        "mean_kl_matched_features": kl_a[matched & ok].mean().item() if (matched & ok).any() else float("nan"),
        "tos_features(F1>0.4)_whose_ablation_lowers_matching_close_most": f"{tos_hits}/{tos_total}",
    }
    patching = patching_experiment(model, sae, batch, layer)
    log(f"[causal] patching 8 selected features recovers {patching['selected'][3]:.2f} of the logit gap; all features {patching['all_features']:.2f}")
    write_report_md(cfg.out / "report.md", model_metrics, report, fit, ablation_summary, patching)
    (cfg.out / "results.json").write_text(json.dumps(
        {"sae": report, "ablation": ablation_summary, "patching": patching,
         "sweep": {"features": sweep["features"], "kl_active": sweep["kl_active"], "kl_control": sweep["kl_control"]}}, indent=1))

    # --- plots -----------------------------------------------------------------------------
    plots = cfg.out / "plots"
    plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    hs = model_metrics["history"]; ax[0].plot([a for a, _ in hs], [b for _, b in hs]); ax[0].set_yscale("log")
    ax[0].set(title="Transformer training loss", xlabel="step")
    hist = fit.history; ax[1].plot([a for a, _, _ in hist], [b for _, b, _ in hist]); ax[1].set_yscale("log")
    ax[1].set(title="SAE normalised MSE", xlabel="step")
    d = stats["density"][stats["density"] > 0].log10()
    ax[2].hist(d.numpy(), bins=30, color="#3a6df0"); ax[2].set(title=f"Feature density (log10), {report['dead_features']} dead", xlabel="log10 firing rate")
    ax[3].hist(stats["best_f1"][alive].numpy(), bins=20, color="#2a9d8f"); ax[3].set(title="Best ground-truth F1 per feature", xlabel="F1")
    fig.tight_layout(); fig.savefig(plots / "training_and_features.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(kl_c[ok].numpy(), kl_a[ok].numpy(), s=10, c=matched[ok].numpy(), cmap="coolwarm")
    mx = float(max(kl_a[ok].max(), kl_c[ok].max())); ax[0].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[0].set(xscale="symlog", yscale="symlog", xlabel="KL, random-direction control", ylabel="KL, feature ablation", title="Ablation vs control (red: F1>0.7)")
    ax[1].errorbar(patching["sizes"], patching["random_mean"], patching["random_std"], label="random features", color="#8a93a5", marker="o")
    ax[1].plot(patching["sizes"], patching["selected"], label="selected features", color="#3a6df0", marker="o")
    ax[1].axhline(1, color="k", lw=0.6, ls=":"); ax[1].set(xscale="log", xlabel="# patched features", ylabel="logit-diff flipped", title="Activation patching '(' → '['")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plots / "causal_tests.png", dpi=130); plt.close(fig)
    return report, ablation_summary, patching
