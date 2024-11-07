import pytest
import torch

from circuitscope import kernels

cpp = pytest.mark.skipif(not kernels.available(), reason="C++ extension not built (make build)")


@cpp
@pytest.mark.parametrize("B,F,k", [(1, 5, 5), (7, 64, 1), (33, 300, 8), (16, 1000, 64)])
def test_topk_matches_torch(B, F, k):
    torch.manual_seed(B + F)
    pre = torch.randn(B, F)
    v_ref, i_ref = torch.topk(pre, k, dim=1)
    v, i = kernels.topk(pre, k, "cpp")
    assert torch.equal(v, v_ref) and torch.equal(i, i_ref)


@cpp
def test_topk_ties_prefer_lower_index_and_negatives():
    pre = torch.tensor([[1.0, 3.0, 3.0, -2.0, 3.0, -5.0]])
    v, i = kernels.topk(pre, 4, "cpp")
    assert i.tolist() == [[1, 2, 4, 0]] and v.tolist() == [[3.0, 3.0, 3.0, 1.0]]


@cpp
def test_topk_rejects_bad_k():
    with pytest.raises(Exception):
        kernels.topk(torch.randn(2, 4), 5, "cpp")


@cpp
def test_sparse_decode_matches_dense_and_gradients():
    torch.manual_seed(0)
    B, F, k, d = 40, 200, 6, 24
    pre = torch.randn(B, F)
    vals, idx = torch.topk(pre, k, dim=1)
    grads = {}
    outs = {}
    for backend in ("torch", "cpp"):
        v = vals.clone().requires_grad_(True)
        W = torch.randn(F, d, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
        b = torch.randn(d, generator=torch.Generator().manual_seed(2)).requires_grad_(True)
        out = kernels.sparse_decode(v, idx, W, b, backend)
        (out.pow(2).sum()).backward()
        outs[backend] = out.detach()
        grads[backend] = (v.grad, W.grad, b.grad)
    assert torch.allclose(outs["torch"], outs["cpp"], atol=1e-4, rtol=1e-4)
    for gt, gc in zip(grads["torch"], grads["cpp"]):
        assert torch.allclose(gt, gc, atol=1e-3, rtol=1e-4)


@cpp
def test_topk_backward_scatters_to_selected_features():
    pre = torch.randn(5, 30, requires_grad=True)
    v, i = kernels.topk(pre, 3, "cpp")
    v.sum().backward()
    assert pre.grad.sum().item() == 15
    assert torch.equal(pre.grad.gather(1, i), torch.ones(5, 3))


@cpp
def test_decode_rejects_out_of_range_index():
    idx = torch.tensor([[0, 99]])
    with pytest.raises(Exception):
        kernels.sparse_decode(torch.ones(1, 2), idx, torch.randn(10, 4), torch.zeros(4), "cpp")


def test_backend_fallback():
    assert kernels.resolve_backend("torch") == "torch"
    assert kernels.resolve_backend("cpp") in ("cpp", "torch")
    with pytest.raises(ValueError):
        kernels.resolve_backend("cuda")
