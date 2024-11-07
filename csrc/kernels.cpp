// CircuitScope kernels: row-parallel top-k selection and sparse decode (fwd + bwd).
// Operates on raw float32/int64 buffers handed over zero-copy from torch via numpy.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

struct Entry {
    float v;
    int64_t i;
};

// "a ranks before b": larger value first, ties broken by smaller index.
inline bool better(const Entry& a, const Entry& b) {
    return a.v > b.v || (a.v == b.v && a.i < b.i);
}

inline int num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void check_2d(const py::buffer_info& b, const char* name) {
    if (b.ndim != 2) throw std::invalid_argument(std::string(name) + " must be 2-D");
}

}  // namespace

// For each row keep the k largest entries (sorted descending) using a size-k min-heap.
// Expected cost O(F + k log k log(F/k)) per row because most elements fail the root test.
void topk_rows(F32 pre, int64_t k, py::array_t<float> vals_out, py::array_t<int64_t> idx_out) {
    auto p = pre.request();
    auto v = vals_out.request(true);
    auto ix = idx_out.request(true);
    check_2d(p, "pre");
    const int64_t B = p.shape[0], F = p.shape[1];
    if (k < 1 || k > F) throw std::invalid_argument("k must satisfy 1 <= k <= F");
    if (v.ndim != 2 || v.shape[0] != B || v.shape[1] != k || ix.ndim != 2 || ix.shape[0] != B ||
        ix.shape[1] != k)
        throw std::invalid_argument("output buffers must have shape [B, k]");
    const float* x = static_cast<const float*>(p.ptr);
    float* vo = static_cast<float*>(v.ptr);
    int64_t* io = static_cast<int64_t*>(ix.ptr);
    {
        py::gil_scoped_release release;
#pragma omp parallel
        {
            std::vector<Entry> heap(static_cast<size_t>(k));
#pragma omp for schedule(static)
            for (int64_t b = 0; b < B; ++b) {
                const float* row = x + b * F;
                for (int64_t j = 0; j < k; ++j) heap[j] = {row[j], j};
                std::make_heap(heap.begin(), heap.end(), better);  // root = worst kept
                for (int64_t j = k; j < F; ++j) {
                    const float val = row[j];
                    if (val > heap.front().v) {  // strict: earlier index wins ties
                        std::pop_heap(heap.begin(), heap.end(), better);
                        heap.back() = {val, j};
                        std::push_heap(heap.begin(), heap.end(), better);
                    }
                }
                std::sort_heap(heap.begin(), heap.end(), better);  // best first
                for (int64_t j = 0; j < k; ++j) {
                    vo[b * k + j] = heap[j].v;
                    io[b * k + j] = heap[j].i;
                }
            }
        }
    }
}

// out[b] = bias + sum_j vals[b,j] * W[idx[b,j]]     (touches B*k*d instead of B*F*d)
void sparse_decode(F32 vals, I64 idx, F32 W, F32 bias, py::array_t<float> out) {
    auto v = vals.request();
    auto ix = idx.request();
    auto w = W.request();
    auto bb = bias.request();
    auto o = out.request(true);
    check_2d(v, "vals");
    check_2d(ix, "idx");
    check_2d(w, "W");
    const int64_t B = v.shape[0], k = v.shape[1], F = w.shape[0], d = w.shape[1];
    if (ix.shape[0] != B || ix.shape[1] != k || bb.size != d || o.ndim != 2 || o.shape[0] != B ||
        o.shape[1] != d)
        throw std::invalid_argument("shape mismatch in sparse_decode");
    const float* pv = static_cast<const float*>(v.ptr);
    const int64_t* pi = static_cast<const int64_t*>(ix.ptr);
    const float* pw = static_cast<const float*>(w.ptr);
    const float* pb = static_cast<const float*>(bb.ptr);
    float* po = static_cast<float*>(o.ptr);
    bool bad = false;
    {
        py::gil_scoped_release release;
#pragma omp parallel for schedule(static) reduction(|| : bad)
        for (int64_t b = 0; b < B; ++b) {
            float* orow = po + b * d;
            for (int64_t c = 0; c < d; ++c) orow[c] = pb[c];
            for (int64_t j = 0; j < k; ++j) {
                const int64_t f = pi[b * k + j];
                if (f < 0 || f >= F) {
                    bad = true;
                    continue;
                }
                const float a = pv[b * k + j];
                const float* wrow = pw + f * d;
#pragma omp simd
                for (int64_t c = 0; c < d; ++c) orow[c] += a * wrow[c];
            }
        }
    }
    if (bad) throw std::out_of_range("feature index out of range in sparse_decode");
}

// Given g = dL/d(out): gvals[b,j] = <g[b], W[idx[b,j]]> and gW[f] += vals[b,j] * g[b].
// gW is accumulated without atomics: thread t owns features with f % T == t, so the result
// is deterministic for a fixed thread count.
void sparse_decode_backward(F32 g, F32 vals, I64 idx, F32 W, py::array_t<float> gvals_out,
                            py::array_t<float> gW_inout) {
    auto gb = g.request();
    auto v = vals.request();
    auto ix = idx.request();
    auto w = W.request();
    auto gv = gvals_out.request(true);
    auto gw = gW_inout.request(true);
    const int64_t B = v.shape[0], k = v.shape[1], F = w.shape[0], d = w.shape[1];
    if (gb.ndim != 2 || gb.shape[0] != B || gb.shape[1] != d || gv.size != B * k || gw.size != F * d)
        throw std::invalid_argument("shape mismatch in sparse_decode_backward");
    const float* pg = static_cast<const float*>(gb.ptr);
    const float* pv = static_cast<const float*>(v.ptr);
    const int64_t* pi = static_cast<const int64_t*>(ix.ptr);
    const float* pw = static_cast<const float*>(w.ptr);
    float* pgv = static_cast<float*>(gv.ptr);
    float* pgw = static_cast<float*>(gw.ptr);
    bool bad = false;
    {
        py::gil_scoped_release release;
#pragma omp parallel reduction(|| : bad)
        {
#pragma omp for schedule(static) nowait
            for (int64_t b = 0; b < B; ++b) {
                const float* grow = pg + b * d;
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t f = pi[b * k + j];
                    if (f < 0 || f >= F) {
                        bad = true;
                        pgv[b * k + j] = 0.f;
                        continue;
                    }
                    const float* wrow = pw + f * d;
                    float acc = 0.f;
#pragma omp simd reduction(+ : acc)
                    for (int64_t c = 0; c < d; ++c) acc += grow[c] * wrow[c];
                    pgv[b * k + j] = acc;
                }
            }
#ifdef _OPENMP
            const int64_t T = omp_get_num_threads(), tid = omp_get_thread_num();
#else
            const int64_t T = 1, tid = 0;
#endif
            for (int64_t n = 0; n < B * k; ++n) {
                const int64_t f = pi[n];
                if (f < 0 || f >= F || f % T != tid) continue;
                const float a = pv[n];
                const float* grow = pg + (n / k) * d;
                float* dst = pgw + f * d;
#pragma omp simd
                for (int64_t c = 0; c < d; ++c) dst[c] += a * grow[c];
            }
        }
    }
    if (bad) throw std::out_of_range("feature index out of range in sparse_decode_backward");
}

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "OpenMP top-k and sparse-decode kernels for CircuitScope";
    m.def("topk_rows", &topk_rows, py::arg("pre"), py::arg("k"), py::arg("vals_out"),
          py::arg("idx_out"));
    m.def("sparse_decode", &sparse_decode, py::arg("vals"), py::arg("idx"), py::arg("W"),
          py::arg("bias"), py::arg("out"));
    m.def("sparse_decode_backward", &sparse_decode_backward, py::arg("g"), py::arg("vals"),
          py::arg("idx"), py::arg("W"), py::arg("gvals_out"), py::arg("gW_inout"));
    m.def("max_threads", &num_threads);
}
