"""Top-k sparse autoencoder (Gao et al. style) and its trainer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import kernels


class TopKSAE(nn.Module):
    """x -> (x - b_dec) W_enc + b_enc -> keep k largest (ReLU) -> sparse decode with unit-norm W_dec."""

    def __init__(self, d_model: int, n_features: int, k: int, backend: str = "torch", seed: int = 0):
        super().__init__()
        if not 1 <= k <= n_features:
            raise ValueError("need 1 <= k <= n_features")
        g = torch.Generator().manual_seed(seed)
        dec = torch.randn(n_features, d_model, generator=g)
        dec = dec / dec.norm(dim=1, keepdim=True)
        self.d_model, self.n_features, self.k = d_model, n_features, k
        self.backend = kernels.resolve_backend(backend)
        self.W_dec = nn.Parameter(dec)
        self.W_enc = nn.Parameter(dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("scale", torch.tensor(1.0))  # input normalisation (set by the trainer)

    def encode(self, x: torch.Tensor):
        """Sparse code in normalised space: (values [B,k] >= 0, feature indices [B,k])."""
        xs = x * self.scale
        pre = (xs - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = kernels.topk(pre, self.k, self.backend)
        return torch.relu(vals), idx

    def decode(self, vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return kernels.sparse_decode(vals, idx, self.W_dec, self.b_dec, self.backend) / self.scale

    def forward(self, x: torch.Tensor):
        vals, idx = self.encode(x)
        return self.decode(vals, idx), vals, idx

    @torch.no_grad()
    def normalize_decoder(self):
        n = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(n)

    def direction(self, feature: int) -> torch.Tensor:
        """Decoder direction of a feature expressed in the model's residual-stream units."""
        return self.W_dec[feature].detach() / self.scale

    def dense(self, vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.zeros(vals.shape[0], self.n_features).scatter(1, idx, vals)


@dataclass
class SAEFitResult:
    history: list = field(default_factory=list)  # (step, nmse, n_dead)
    resampled: int = 0
    seconds: float = 0.0


def fit_sae(sae: TopKSAE, acts: torch.Tensor, steps: int, batch: int = 2048, lr: float = 2e-3,
            resample_every: int = 500, dead_window: int = 300, seed: int = 0, log=print) -> SAEFitResult:
    """Adam training with unit-norm decoder rows and periodic resampling of dead features."""
    import time

    g = torch.Generator().manual_seed(seed)
    n = acts.shape[0]
    with torch.no_grad():
        sae.scale.fill_(math.sqrt(acts.shape[1]) / acts.norm(dim=1).mean().item())
        sae.b_dec.copy_(acts.mean(0) * sae.scale)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.1, 1 - s / steps))
    last_fired = torch.zeros(sae.n_features, dtype=torch.long)
    var = (acts * sae.scale).var(0).sum().item()
    res = SAEFitResult()
    t0 = time.perf_counter()
    for step in range(steps):
        x = acts[torch.randint(0, n, (batch,), generator=g)]
        xs = x * sae.scale
        recon, vals, idx = sae(x)
        err = recon * sae.scale - xs
        loss = err.pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():  # keep Adam from moving decoder rows along their own direction
            sae.W_dec.grad -= (sae.W_dec.grad * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step()
        sched.step()
        sae.normalize_decoder()
        with torch.no_grad():
            fired = torch.zeros(sae.n_features, dtype=torch.bool)
            fired[idx[vals > 0].flatten()] = True
            last_fired[fired] = step
            dead = (step - last_fired) >= dead_window
            if resample_every and step > 0 and step % resample_every == 0 and step < 0.7 * steps and dead.any():
                _resample(sae, opt, dead, xs, err.pow(2).sum(-1), g)
                last_fired[dead] = step
                res.resampled += int(dead.sum())
        if step % 50 == 0 or step == steps - 1:
            res.history.append((step, loss.item() / var, int(dead.sum())))
        if step % 500 == 0:
            log(f"[sae/{sae.backend}] step {step:5d} nmse {loss.item() / var:.4f} dead {int(dead.sum())}")
    res.seconds = time.perf_counter() - t0
    return res


@torch.no_grad()
def _resample(sae: TopKSAE, opt, dead, xs, sq_err, g):
    ids = dead.nonzero().flatten()
    probs = sq_err / sq_err.sum() if sq_err.sum() > 0 else torch.full_like(sq_err, 1 / len(sq_err))
    pick = torch.multinomial(probs, len(ids), replacement=len(ids) > len(probs), generator=g)
    direction = xs[pick] - sae.b_dec
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    alive_scale = sae.W_enc[:, ~dead].norm(dim=0).mean() if (~dead).any() else torch.tensor(1.0)
    sae.W_dec[ids] = direction
    sae.W_enc[:, ids] = (direction * 0.2 * alive_scale).t()
    sae.b_enc[ids] = 0.0
    for p, sl in ((sae.W_dec, (ids,)), (sae.W_enc, (slice(None), ids)), (sae.b_enc, (ids,))):
        for key in ("exp_avg", "exp_avg_sq"):
            if p in opt.state and key in opt.state[p]:
                opt.state[p][key][sl] = 0.0


@torch.no_grad()
def encode_all(sae: TopKSAE, acts: torch.Tensor, chunk: int = 16384):
    vals, idx = zip(*(sae.encode(acts[i:i + chunk]) for i in range(0, acts.shape[0], chunk)))
    return torch.cat(vals), torch.cat(idx)
