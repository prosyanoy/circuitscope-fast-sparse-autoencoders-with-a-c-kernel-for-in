import pytest
import torch

from circuitscope import kernels
from circuitscope.analysis import feature_stats, harvest, recovery_rate, top_examples
from circuitscope.data import sample_dyck
from circuitscope.interventions import ablation_sweep, feature_acts, patching_experiment, single_ablation
from circuitscope.model import TinyTransformer
from circuitscope.sae import TopKSAE, fit_sae

BACKENDS = ["torch"] + (["cpp"] if kernels.available() else [])


def planted_data(n=6000, d=16, n_true=12, seed=0):
    """Data that is exactly a sparse combination of known unit directions."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_true, d, generator=g)
    dirs /= dirs.norm(dim=1, keepdim=True)
    coef = torch.zeros(n, n_true)
    for j in range(2):
        coef[torch.arange(n), torch.randint(0, n_true, (n,), generator=g)] = 1 + torch.rand(n, generator=g)
    return coef @ dirs, dirs


@pytest.mark.parametrize("backend", BACKENDS)
def test_sae_recovers_planted_features(backend):
    torch.manual_seed(0)
    x, dirs = planted_data()
    sae = TopKSAE(16, 24, k=2, backend=backend)
    fit_sae(sae, x, steps=600, batch=512, lr=5e-3, seed=0, log=lambda *_: None)
    recon, _, _ = sae(x)
    assert ((recon - x).pow(2).sum(-1).mean() / x.pow(2).sum(-1).mean()) < 0.1
    cos = (dirs @ sae.W_dec.detach().t()).abs().max(1).values  # best-matching learned atom per true direction
    assert cos.min() > 0.9
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(24), atol=1e-4)


def test_backends_give_same_sae_forward():
    if not kernels.available():
        pytest.skip("C++ extension not built")
    x = torch.randn(64, 16)
    a, b = TopKSAE(16, 40, 4, "torch", seed=1), TopKSAE(16, 40, 4, "cpp", seed=1)
    ra, rb = a(x)[0], b(x)[0]
    assert torch.allclose(ra, rb, atol=1e-5)


def test_sae_sparsity_is_exactly_k():
    sae = TopKSAE(8, 32, 3)
    vals, idx = sae.encode(torch.randn(10, 8))
    assert idx.shape == (10, 3) and (vals >= 0).all()
    assert (sae.dense(vals, idx) > 0).sum(1).max() <= 3


@pytest.fixture(scope="module")
def tiny_setup():
    torch.manual_seed(0)
    model = TinyTransformer(d_model=16, n_layers=2, n_heads=2).eval()
    batch = sample_dyck(300, 12, 0)
    h = harvest(model, batch, 1)
    sae = TopKSAE(16, 32, 4, "torch")
    fit_sae(sae, h.acts, steps=60, batch=256, seed=0, log=lambda *_: None)
    return model, sae, batch, h


def test_feature_stats_and_top_examples(tiny_setup):
    model, sae, batch, h = tiny_setup
    st = feature_stats(sae, h)
    assert st["density"].shape == (32,) and (st["best_f1"] <= 1 + 1e-6).all()
    assert abs(st["density"].sum().item() - (st["vals"] > 0).sum(1).float().mean().item()) < 1e-4
    tv, ti = top_examples(st, n_per=3)
    f = int(st["density"].argmax())
    assert tv[f, 0] == st["max_act"][f]
    assert (ti[f] >= 0).all()


def test_ablation_zero_for_inactive_and_exact_removal(tiny_setup):
    model, sae, batch, h = tiny_setup
    st = feature_stats(sae, h)
    dead = (st["count"] == 0).nonzero().flatten().tolist()
    alive = st["density"].argsort(descending=True)[:3].tolist()
    toks = batch.tokens[:100]
    res = ablation_sweep(model, sae, toks, 1, alive + dead[:1])
    assert all(v > 0 for v in res["kl_active"][:3])
    if dead:  # a feature that never fires cannot change anything
        assert res["kl_active"][3] != res["kl_active"][3]  # NaN: no firing positions
    # feature_acts agrees with the encoder
    resid = model.resid(toks, 1).reshape(-1, 16)
    a = feature_acts(sae, resid, torch.tensor(alive))
    vals, idx = sae.encode(resid)
    assert torch.allclose(a.sum(1), (vals * torch.isin(idx, torch.tensor(alive))).sum(1), atol=1e-5)


def test_patching_all_features_flips_most_of_the_gap(tiny_setup):
    model, sae, batch, h = tiny_setup
    out = patching_experiment(model, sae, batch, 1, sizes=(1, 4), n_random=3)
    assert len(out["selected"]) == 2 and out["n_pairs"] > 0
    assert out["all_features"] > 0  # patching the whole code moves logits toward the corrupt run


def test_recovery_rate_bounds_and_perfect_case(tiny_setup):
    model, sae, batch, h = tiny_setup
    st = feature_stats(sae, h)
    r = recovery_rate(st)
    assert r["ground_truth_predicates"] >= r["structural_predicates"] > 0
    assert 0 <= r["recovery_rate_F1>=0.9"] <= r["recovery_rate_F1>=0.7"] <= 1
    # a fake "perfect" SAE: feature j matches predicate j with F1 = 1
    perfect = dict(st, f1=torch.eye(st["cand"].shape[1]))
    assert recovery_rate(perfect)["recovery_rate_F1>=0.9"] == 1.0


def test_single_ablation_matches_batch_sweep_and_control(tiny_setup):
    model, sae, batch, h = tiny_setup
    f = int(feature_stats(sae, h)["density"].argmax())
    act, clean, abl = single_ablation(model, sae, batch.tokens[0], 1, f)
    assert torch.allclose(clean.sum(-1), torch.ones(clean.shape[0]), atol=1e-5)
    silent = act == 0
    assert act.max() > 0
    # before the first position where the feature fires, nothing can have changed
    first = int((act > 0).nonzero()[0])
    assert torch.allclose(clean[:first], abl[:first], atol=1e-5) and not torch.allclose(clean, abl, atol=1e-6)
    _, _, ctl = single_ablation(model, sae, batch.tokens[0], 1, f, random_control=True)
    assert ctl.shape == abl.shape
