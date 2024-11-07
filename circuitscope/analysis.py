"""Feature analysis: density, dead features, reconstruction quality, ground-truth matching,
top-activating examples."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import EMPTY, DyckBatch
from .model import TinyTransformer, kl_to_target
from .sae import TopKSAE, encode_all


@dataclass
class Harvest:
    """Residual-stream activations flattened over (sequence, position), with labels."""
    acts: torch.Tensor  # [N, d]
    seq: torch.Tensor  # [N] sequence index
    pos: torch.Tensor  # [N]
    depth: torch.Tensor
    tos: torch.Tensor
    T: int


@torch.no_grad()
def harvest(model: TinyTransformer, batch: DyckBatch, layer: int) -> Harvest:
    n, T = batch.tokens.shape
    acts = torch.cat([model.resid(batch.tokens[i:i + 1024], layer) for i in range(0, n, 1024)]).reshape(n * T, -1)
    seq = torch.arange(n).repeat_interleave(T)
    pos = torch.arange(T).repeat(n)
    return Harvest(acts, seq, pos, batch.depth.reshape(-1), batch.tos.reshape(-1), T)


def label_candidates(h: Harvest) -> tuple[list[str], torch.Tensor]:
    """Boolean ground-truth predicates over tokens: [N, C]."""
    names, cols = [], []
    for d in range(0, int(h.depth.max()) + 1):
        names.append(f"depth=={d}")
        cols.append(h.depth == d)
    for t, name in ((0, "tos=='('"), (1, "tos=='['"), (EMPTY, "stack empty")):
        names.append(name)
        cols.append(h.tos == t)
    for p in range(h.T):
        names.append(f"pos=={p}")
        cols.append(h.pos == p)
    return names, torch.stack(cols, 1)


@torch.no_grad()
def feature_stats(sae: TopKSAE, h: Harvest) -> dict:
    """Per-feature firing statistics and best-matching ground-truth predicate (by F1)."""
    vals, idx = encode_all(sae, h.acts)
    N, k = vals.shape
    F = sae.n_features
    active = vals > 0
    flat_idx = idx.flatten()
    count = torch.zeros(F).index_add_(0, flat_idx, active.flatten().float())
    total = torch.zeros(F).index_add_(0, flat_idx, vals.flatten())
    maxact = torch.zeros(F).scatter_reduce_(0, flat_idx, vals.flatten(), "amax", include_self=True)
    names, cand = label_candidates(h)
    C = cand.shape[1]
    co = torch.zeros(F, C)
    for s in range(0, N, 4096):  # co-occurrence of "feature active" with each predicate
        a = active[s:s + 4096].float()
        ci = cand[s:s + 4096].float()
        rows = idx[s:s + 4096].flatten()
        co.index_add_(0, rows, (a.unsqueeze(-1) * ci.unsqueeze(1)).reshape(-1, C))
    prior = cand.float().sum(0)
    precision = co / count.clamp_min(1).unsqueeze(1)
    recall = co / prior.clamp_min(1).unsqueeze(0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    best_f1, best_c = f1.max(1)
    return dict(vals=vals, idx=idx, density=count / N, count=count, mean_act=total / count.clamp_min(1),
                max_act=maxact, label_names=names, best_label=best_c, best_f1=best_f1,
                best_precision=precision.gather(1, best_c[:, None])[:, 0],
                best_recall=recall.gather(1, best_c[:, None])[:, 0], cand=cand, f1=f1)


def recovery_rate(stats: dict, thresholds=(0.9, 0.7)) -> dict:
    """How many ground-truth variables does the SAE recover as a single feature?

    The ground-truth set is every predicate in `label_candidates` that actually occurs in the data:
    depth==d, top-of-stack '(' / '[' / empty, and pos==p. A predicate counts as *recovered at t* if some
    feature's activation matches it with F1 >= t. Reported overall and for the structural subset
    (depth/stack predicates, i.e. everything that is not a position).
    """
    names = stats["label_names"]
    present = stats["cand"].any(0)
    best = stats["f1"].max(0).values  # best feature per predicate
    structural = torch.tensor([not n.startswith("pos==") for n in names]) & present
    out = {"ground_truth_predicates": int(present.sum()), "structural_predicates": int(structural.sum())}
    for t in thresholds:
        out[f"recovery_rate_F1>={t}"] = (best[present] >= t).float().mean().item()
        out[f"structural_recovery_rate_F1>={t}"] = (best[structural] >= t).float().mean().item()
    out["unrecovered_structural(F1<0.7)"] = ", ".join(n for n, b, m in zip(names, best.tolist(), structural.tolist()) if m and b < 0.7) or "none"
    return out


@torch.no_grad()
def reconstruction_report(model: TinyTransformer, sae: TopKSAE, batch: DyckBatch, h: Harvest, layer: int,
                          stats: dict) -> dict:
    """Reconstruction error plus downstream (spliced-in) loss versus ablation baselines."""
    recon = sae.decode(stats["vals"], stats["idx"])
    err = (recon - h.acts).pow(2).sum(-1)
    var = (h.acts - h.acts.mean(0)).pow(2).sum(-1).mean()
    n, T = batch.tokens.shape

    def downstream(resid_flat):
        return kl_to_target(model.forward_from(resid_flat.reshape(n, T, -1), layer), batch.target).item()

    clean = downstream(h.acts)
    spliced = downstream(recon)
    zero = downstream(torch.zeros_like(h.acts))
    mean = downstream(h.acts.mean(0, keepdim=True).expand_as(h.acts))
    dens = stats["density"]
    return {
        "mse": err.mean().item() / h.acts.shape[1],
        "normalized_mse": (err.mean() / var).item(),
        "variance_explained": 1 - (err.mean() / var).item(),
        "mean_l0": (stats["vals"] > 0).sum(1).float().mean().item(),
        "kl_clean": clean, "kl_spliced": spliced, "kl_zero_ablation": zero, "kl_mean_ablation": mean,
        "loss_recovered": 1 - (spliced - clean) / max(mean - clean, 1e-9),
        "n_features": sae.n_features,
        "dead_features": int((stats["count"] == 0).sum()),
        "rare_features_lt_1e-4": int(((dens > 0) & (dens < 1e-4)).sum()),
        "dense_features_gt_10pct": int((dens > 0.1).sum()),
        "features_f1_gt_0.9": int((stats["best_f1"] > 0.9).sum()),
        "features_f1_gt_0.7": int((stats["best_f1"] > 0.7).sum()),
        "median_alive_f1": stats["best_f1"][stats["count"] > 0].median().item() if (stats["count"] > 0).any() else 0.0,
    }


@torch.no_grad()
def top_examples(stats: dict, n_per: int = 8, chunk: int = 8192):
    """Top-activating token positions per feature: (values [F, n], flat token indices [F, n])."""
    vals, idx = stats["vals"], stats["idx"]
    F = int(stats["density"].shape[0])
    best_v = torch.zeros(F, n_per)
    best_i = torch.full((F, n_per), -1, dtype=torch.long)
    for s in range(0, vals.shape[0], chunk):
        dense = torch.zeros(vals[s:s + chunk].shape[0], F).scatter(1, idx[s:s + chunk], vals[s:s + chunk])
        v, i = dense.topk(min(n_per, dense.shape[0]), dim=0)
        v, i = v.t(), i.t() + s
        allv = torch.cat([best_v, v], 1)
        alli = torch.cat([best_i, i], 1)
        top = allv.topk(n_per, dim=1)
        best_v, best_i = top.values, alli.gather(1, top.indices)
    best_i[best_v <= 0] = -1
    return best_v, best_i
