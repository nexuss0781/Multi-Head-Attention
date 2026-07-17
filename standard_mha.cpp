#include "standard_mha.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

// ============================================================================
// Utility helpers (same style as TMHA-CRUS)
// ============================================================================

namespace {
    void matmul(double* C, const double* A, const double* B,
                size_t m, size_t k, size_t n) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] = 0.0;
                for (size_t l = 0; l < k; ++l)
                    C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
    }

    void vec_mat_mul(double* y, const double* x, const double* A,
                     size_t m, size_t n) {
        // y_j = sum_i x[i] * A[i][j]   (x: m, A: m×n, y: n)
        for (size_t j = 0; j < n; ++j) {
            y[j] = 0.0;
            for (size_t i = 0; i < m; ++i)
                y[j] += x[i] * A[i * n + j];
        }
    }
}

// ============================================================================
// StdMHA
// ============================================================================

StdMHA::StdMHA(const SMHAConfig& config) : cfg(config) {
    size_t d   = cfg.d;
    size_t h   = cfg.h;
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();

    W_Q.resize(d * h * d_k, 0.0);
    W_K.resize(d * h * d_k, 0.0);
    W_V.resize(d * h * d_v, 0.0);
    W_O.resize(h * d_v * d, 0.0);

    // C2.2 per-head feature maps
    size_t r_eff = cfg.r_feat_eff();
    W_22.reserve(h);
    for (size_t i = 0; i < h; ++i)
        W_22.emplace_back(d_k, r_eff / 2);
    w_g.resize(h * r_eff, 0.0);
    b_g.resize(h, 0.0);
}

void StdMHA::init(std::mt19937& rng) {
    size_t d = cfg.d, h = cfg.h, d_k = cfg.d_k(), d_v = cfg.d_v();
    size_t r_eff = cfg.r_feat_eff();

    auto xavier = [](size_t d_in, size_t d_out) {
        return std::sqrt(2.0 / (d_in + d_out));
    };

    auto fill_norm = [&](double* data, size_t n, double std) {
        std::normal_distribution<double> dist(0.0, std);
        for (size_t i = 0; i < n; ++i) data[i] = dist(rng);
    };

    fill_norm(W_Q.data(), d * h * d_k, xavier(d, d_k));
    fill_norm(W_K.data(), d * h * d_k, xavier(d, d_k));
    fill_norm(W_V.data(), d * h * d_v, xavier(d, d_v));
    fill_norm(W_O.data(), h * d_v * d, xavier(h * d_v, d));

    for (size_t i = 0; i < h; ++i)
        W_22[i].randomize(static_cast<unsigned>(rng()));
    fill_norm(w_g.data(), h * r_eff, xavier(d_k, r_eff));
    fill_norm(b_g.data(), h, 0.01);
}

// ---- Bidirectional Forward ----

std::vector<double> StdMHA::forward_bidirectional(
    const double* X, size_t n) const
{
    size_t d = cfg.d, h = cfg.h, d_k = cfg.d_k(), d_v = cfg.d_v();
    size_t r_eff = cfg.r_feat_eff();

    // Per-head QKV
    std::vector<double> Q(n * h * d_k);
    std::vector<double> K(n * h * d_k);
    std::vector<double> V(n * h * d_v);

    // Q = X · W_Q, etc.
    for (size_t i = 0; i < h; ++i) {
        matmul(Q.data() + i * n * d_k, X,
               W_Q.data() + i * d_k * d, n, d, d_k);
        matmul(K.data() + i * n * d_k, X,
               W_K.data() + i * d_k * d, n, d, d_k);
        matmul(V.data() + i * n * d_v, X,
               W_V.data() + i * d_v * d, n, d, d_v);
    }

    // Per-head C2.2 attention, then concat
    std::vector<double> heads(n * h * d_v, 0.0);
    for (size_t i = 0; i < h; ++i) {
        const double* Qi = Q.data() + i * n * d_k;
        const double* Ki = K.data() + i * n * d_k;
        const double* Vi = V.data() + i * n * d_v;
        double* Oi = heads.data() + i * n * d_v;

        ::sgkd_sdpa_bidirectional(
            Oi, Qi, Ki, Vi,
            W_22[i],
            w_g.data() + i * r_eff, b_g[i],
            n, d_k, r_eff, d_v, cfg.w_local);
    }

    // Output projection: O · W_O
    std::vector<double> result(n * d, 0.0);
    matmul(result.data(), heads.data(), W_O.data(), n, h * d_v, d);

    return result;
}

// ---- Causal Forward Step ----

StdMHA::StepOutput StdMHA::forward_causal_step(
    const double* x, SMHACausalState& state) const
{
    size_t d = cfg.d, h = cfg.h, d_k = cfg.d_k(), d_v = cfg.d_v();
    size_t r_eff = cfg.r_feat_eff();

    // Auto-initialize SGKD states
    if (state.sgkd_states.size() != h ||
        (state.sgkd_states.size() > 0 && state.sgkd_states[0].S_i.size() == 0)) {
        state.sgkd_states.resize(h);
        for (size_t i = 0; i < h; ++i)
            state.sgkd_states[i] = SGKDState<double>(r_eff, d_v, cfg.w_local, d_k);
    }

    // Per-head QKV for this token
    std::vector<double> q_i(h * d_k), k_i(h * d_k), v_i(h * d_v);
    for (size_t i = 0; i < h; ++i) {
        vec_mat_mul(q_i.data() + i * d_k, x,
                    W_Q.data() + i * d_k * d, d, d_k);
        vec_mat_mul(k_i.data() + i * d_k, x,
                    W_K.data() + i * d_k * d, d, d_k);
        vec_mat_mul(v_i.data() + i * d_v, x,
                    W_V.data() + i * d_v * d, d, d_v);
    }

    // Per-head C2.2 step, store outputs concatenated
    std::vector<double> head_out(h * d_v);
    for (size_t i = 0; i < h; ++i) {
        ::sgkd_sdpa_step(
            head_out.data() + i * d_v,
            state.sgkd_states[i],
            q_i.data() + i * d_k,
            k_i.data() + i * d_k,
            v_i.data() + i * d_v,
            W_22[i],
            w_g.data() + i * r_eff, b_g[i],
            d_k, r_eff, d_v);
    }

    // Output projection
    StepOutput out;
    out.o.resize(d);
    vec_mat_mul(out.o.data(), head_out.data(), W_O.data(), h * d_v, d);
    return out;
}
