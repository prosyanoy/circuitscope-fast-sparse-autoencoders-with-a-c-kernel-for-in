"""Causal tests on SAE features: ablation sweeps and activation patching.

All edits are applied *additively in the residual stream* at the SAE hook point:
    resid' = resid + sum_i (a_i' - a_i) * d_i
where d_i is the feature's decoder direction (model units). Unlike swapping in the SAE
reconstruction, this leaves the SAE's reconstruction error untouched, so any change in the
model's output is attributable to the edited features alone.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import DyckBatch
from .model import TinyTransformer
from .sae import TopKSAE


@torch.no_grad()
def feature_acts(sae: TopKSAE, resid: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """Activations [N, m] of the chosen features for flattened residuals [N, d] (0 if not in top-k)."""
    vals, idx = sae.encode(resid)
    hit = idx.unsqueeze(-1) == feats.view(1, 1, -1)  # [N, k, m]
    return (hit * vals.unsqueeze(-1)).sum(1)


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(clean || edited) per position."""
    lp, lq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


@torch.no_grad()
def ablation_sweep(model: TinyTransformer, sae: TopKSAE, tokens: torch.Tensor, layer: int,
                   features: list[int], generator: torch.Generator | None = None) -> dict:
    """Zero-ablate each feature (and a norm-matched random-direction control) and measure the change.

    Returns per-feature tensors:
      kl_active / kl_inactive : mean KL(clean||ablated) at positions where the feature fires / does not
      kl_control              : same as kl_active but removing a_i * (random unit direction)
      dlogp                   : mean change in log-probs at firing positions, [m, vocab]
    """
    n, T = tokens.shape
    resid = model.resid(tokens, layer)
    flat = resid.reshape(n * T, -1)
    clean_logits = model.forward_from(resid, layer)
    feats = torch.tensor(features)
    acts = feature_acts(sae, flat, feats)  # [N, m]
    g = generator or torch.Generator().manual_seed(0)
    out = {k: [] for k in ("kl_active", "kl_inactive", "kl_control", "dlogp")}
    clean_lp = F.log_softmax(clean_logits, -1).reshape(n * T, -1)
    for j, f in enumerate(features):
        a = acts[:, j]
        act_mask = a > 0
        d = sae.direction(f)
        rand = torch.randn(d.shape[0], generator=g)
        rand = rand / rand.norm() * d.norm()
        res = {}
        for name, direction in (("abl", d), ("ctl", rand)):
            edited = (flat - a.unsqueeze(1) * direction).reshape(n, T, -1)
            res[name] = model.forward_from(edited, layer).reshape(n * T, -1)
        kl = _kl(clean_logits.reshape(n * T, -1), res["abl"])
        klc = _kl(clean_logits.reshape(n * T, -1), res["ctl"])
        nz = act_mask.sum().item()
        out["kl_active"].append(kl[act_mask].mean().item() if nz else float("nan"))
        out["kl_inactive"].append(kl[~act_mask].mean().item() if (~act_mask).any() else float("nan"))
        out["kl_control"].append(klc[act_mask].mean().item() if nz else float("nan"))
        dl = (F.log_softmax(res["abl"], -1) - clean_lp)[act_mask]
        out["dlogp"].append(dl.mean(0) if nz else torch.zeros(clean_lp.shape[1]))
    out["dlogp"] = torch.stack(out["dlogp"])
    out["features"] = features
    return out


@torch.no_grad()
def build_tos_pairs(batch: DyckBatch, max_pairs: int = 2000, seed: int = 0):
    """Pairs of (clean, corrupt) token positions with equal position and depth (>0) but different
    top-of-stack bracket type. Returns (seq_a, seq_b, pos) index tensors."""
    g = torch.Generator().manual_seed(seed)
    n, T = batch.tokens.shape
    sa, sb, ps = [], [], []
    for p in range(1, T):
        for d in range(1, int(batch.depth[:, p].max()) + 1):
            m = batch.depth[:, p] == d
            a = ((batch.tos[:, p] == 0) & m).nonzero().flatten()
            b = ((batch.tos[:, p] == 1) & m).nonzero().flatten()
            c = min(len(a), len(b), 4)
            if c == 0:
                continue
            a = a[torch.randperm(len(a), generator=g)[:c]]
            b = b[torch.randperm(len(b), generator=g)[:c]]
            sa += a.tolist(); sb += b.tolist(); ps += [p] * c
    keep = torch.randperm(len(sa), generator=g)[:max_pairs]
    return (torch.tensor(sa)[keep], torch.tensor(sb)[keep], torch.tensor(ps)[keep])


@torch.no_grad()
def patching_experiment(model: TinyTransformer, sae: TopKSAE, batch: DyckBatch, layer: int,
                        sizes=(1, 2, 4, 8, 16, 32), n_random: int = 20, seed: int = 0) -> dict:
    """Patch feature activations from a 'corrupt' run (top of stack is '[') into a 'clean' run
    (top of stack is '(') at one position and measure how much of the closing-bracket logit
    difference LD = logit(')') - logit(']') moves toward the corrupt run.

    recovery = (LD_patched - LD_clean) / (LD_corrupt - LD_clean)     (1.0 = fully flipped)
    Feature sets: top-m features ranked by |mean activation difference| between the two groups
    ('selected') versus random alive-feature sets of equal size ('random').
    """
    sa, sb, ps = build_tos_pairs(batch, seed=seed)
    resid = model.resid(batch.tokens, layer)
    ra, rb = resid[sa, ps], resid[sb, ps]  # [P, d]
    la = model.forward_from(resid[sa], layer)[torch.arange(len(sa)), ps]
    lb = model.forward_from(resid[sb], layer)[torch.arange(len(sa)), ps]
    ld = lambda lg: lg[:, 2] - lg[:, 3]
    ld_clean, ld_corrupt = ld(la), ld(lb)
    base_gap = (ld_corrupt - ld_clean).mean()

    va, ia = sae.encode(ra)
    vb, ib = sae.encode(rb)
    da, db = sae.dense(va, ia), sae.dense(vb, ib)  # [P, F]
    ranking = (db.mean(0) - da.mean(0)).abs().argsort(descending=True)
    alive = (torch.cat([da, db]).sum(0) > 0).nonzero().flatten()
    g = torch.Generator().manual_seed(seed)

    def run_full(feats: torch.Tensor) -> float:
        delta = (db[:, feats] - da[:, feats]) @ (sae.W_dec[feats] / sae.scale)
        edited = resid[sa].clone()
        edited[torch.arange(len(sa)), ps] += delta
        lg = model.forward_from(edited, layer)[torch.arange(len(sa)), ps]
        return ((ld(lg) - ld_clean).mean() / base_gap).item()

    out = {"sizes": list(sizes), "selected": [], "random_mean": [], "random_std": [],
           "n_pairs": len(sa), "logit_diff_gap": base_gap.item()}
    for m in sizes:
        out["selected"].append(run_full(ranking[:m]))
        rs = [run_full(alive[torch.randperm(len(alive), generator=g)[:m]]) for _ in range(n_random)]
        rs = torch.tensor(rs)
        out["random_mean"].append(rs.mean().item())
        out["random_std"].append(rs.std().item())
    out["all_features"] = run_full(torch.arange(sae.n_features))
    out["top_features"] = ranking[:max(sizes)].tolist()
    return out


@torch.no_grad()
def single_ablation(model: TinyTransformer, sae: TopKSAE, tokens: torch.Tensor, layer: int, feature: int,
                    random_control: bool = False, seed: int = 0):
    """Ablate one feature on one sequence [T]. Returns (activation [T], probs_clean [T,V], probs_ablated [T,V]).
    With random_control=True a norm-matched random direction is removed instead (same activations)."""
    resid = model.resid(tokens.unsqueeze(0), layer)
    flat = resid[0]
    act = feature_acts(sae, flat, torch.tensor([feature]))[:, 0]
    d = sae.direction(feature)
    if random_control:
        r = torch.randn(d.shape[0], generator=torch.Generator().manual_seed(seed))
        d = r / r.norm() * d.norm()
    edited = (flat - act.unsqueeze(1) * d).unsqueeze(0)
    return (act, F.softmax(model.forward_from(resid, layer), -1)[0],
            F.softmax(model.forward_from(edited, layer), -1)[0])
