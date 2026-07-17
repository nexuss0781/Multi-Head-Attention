#include "core.hpp"

namespace tmha {

// ============================================================================
// Matrix utility implementations
// ============================================================================

void mat_vec_mul(double* y, const double* A, const double* x,
                  size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        y[i] = 0.0;
        for (size_t j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void mat_mat_mul(double* C, const double* A, const double* B,
                  size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] = 0.0;
            for (size_t l = 0; l < k; ++l) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void hadamard_rows(double* C, const double* A, const double* s,
                    size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] = A[i * n + j] * s[j];
        }
    }
}

double dot(const double* a, const double* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void axpy(double* y, double alpha, const double* x, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void softmax(double* p, const double* z, size_t n) {
    double max_val = *std::max_element(z, z + n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        p[i] = std::exp(z[i] - max_val);
        sum += p[i];
    }
    double inv_sum = 1.0 / (sum + 1e-30);
    for (size_t i = 0; i < n; ++i) p[i] *= inv_sum;
}

void layer_norm(double* y, const double* x, size_t n, double eps) {
    double mean = 0.0, var = 0.0;
    for (size_t i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    for (size_t i = 0; i < n; ++i) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    double inv_std = 1.0 / std::sqrt(var + eps);
    for (size_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * inv_std;
    }
}

std::vector<size_t> topk_indices(const double* values, size_t n, size_t k) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    size_t kk = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [values](size_t a, size_t b) { return values[a] > values[b]; });
    return std::vector<size_t>(idx.begin(), idx.begin() + kk);
}

// ============================================================================
// TMHACore Implementation
// ============================================================================

TMHACore::TMHACore(const Config& config) : cfg(config), T(config.T) {
    if (!cfg.valid()) {
        throw std::invalid_argument("TMHA Config validation failed");
    }
    // Clamp r_feat to 2*d_k (Householder constraint m ≤ d_k)
    cfg.r_feat = cfg.r_feat_eff();
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();

    Q_factors = CPFactors(cfg.d, cfg.c, cfg.r, d_k);
    K_factors = CPFactors(cfg.d, cfg.c, cfg.r, d_k);
    V_factors = CPFactors(cfg.d, cfg.c, cfg.r, d_v);

    W_cross.resize(cfg.r_h * 2 * cfg.r_h, 0.0);
    W_psi.resize(cfg.r_h * d_v, 0.0);
    W_read.resize(d_k * cfg.r_h, 0.0);

    W_router.resize(cfg.c * (cfg.d + 1), 0.0);
    b_router.resize(cfg.c, 0.0);

    A_g.resize(cfg.c * d_v * cfg.r_o, 0.0);
    B.resize(cfg.d * cfg.r_o, 0.0);

    // Component 2.2 parameters
    size_t r_feat = cfg.r_feat;
    size_t w_local = cfg.w_local;
    W_22.reserve(cfg.c);
    for (size_t g = 0; g < cfg.c; ++g) {
        W_22.emplace_back(d_k, r_feat / 2);
    }
    w_g_22.resize(cfg.c * r_feat, 0.0);
    b_g_22.resize(cfg.c, 0.0);
    if (cfg.T_max_22 > 1) {
        W_chi_22.resize(cfg.c * r_feat * d_v, 0.0);
        W_r_22.resize(cfg.c * r_feat * 2 * r_feat, 0.0);
        w_rho_22.resize(cfg.c * r_feat, 0.0);
        b_rho_22.resize(cfg.c, 0.0);
        tau_22.resize(cfg.c, 0.0);
        w_u_22.resize(cfg.c, 0.0);
    }
}

void TMHACore::init(std::mt19937& rng) {
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();

    // Xavier/Glorot scaling per tensor dimension
    auto xavier_std = [](size_t d_in, size_t d_out) {
        return std::sqrt(2.0 / (d_in + d_out));
    };

    auto fill_norm = [&](double* data, size_t n, double std) {
        std::normal_distribution<double> dist(0.0, std);
        for (size_t i = 0; i < n; ++i) data[i] = dist(rng);
    };

    // B: CP factors — Xavier init per factor
    double u_std = xavier_std(cfg.d, cfg.r);
    double v_std = xavier_std(cfg.c, cfg.r);
    double w_std_q = xavier_std(cfg.d_k(), cfg.r);
    double w_std_v = xavier_std(cfg.d_v(), cfg.r);

    fill_norm(Q_factors.U.data(), Q_factors.d * Q_factors.r, u_std);
    fill_norm(Q_factors.V.data(), Q_factors.rows * Q_factors.r, v_std);
    fill_norm(Q_factors.W.data(), Q_factors.d_k * Q_factors.r, w_std_q);
    for (auto& l : Q_factors.lambda) l = 1.0;

    fill_norm(K_factors.U.data(), K_factors.d * K_factors.r, u_std);
    fill_norm(K_factors.V.data(), K_factors.rows * K_factors.r, v_std);
    fill_norm(K_factors.W.data(), K_factors.d_k * K_factors.r, w_std_q);
    for (auto& l : K_factors.lambda) l = 1.0;

    fill_norm(V_factors.U.data(), V_factors.d * V_factors.r, u_std);
    fill_norm(V_factors.V.data(), V_factors.rows * V_factors.r, v_std);
    fill_norm(V_factors.W.data(), V_factors.d_k * V_factors.r, w_std_v);
    for (auto& l : V_factors.lambda) l = 1.0;

    // C: Cross-head recurrence
    fill_norm(W_cross.data(), cfg.r_h * 2 * cfg.r_h, xavier_std(cfg.r_h, 2 * cfg.r_h));
    fill_norm(W_psi.data(), cfg.r_h * d_v, xavier_std(cfg.r_h, d_v));
    fill_norm(W_read.data(), d_k * cfg.r_h, xavier_std(d_k, cfg.r_h));

    // D: Router
    fill_norm(W_router.data(), cfg.c * (cfg.d + 1), xavier_std(cfg.c, cfg.d + 1));
    fill_norm(b_router.data(), cfg.c, 0.01);

    // E: Output fusion
    fill_norm(A_g.data(), cfg.c * d_v * cfg.r_o, xavier_std(d_v, cfg.r_o));
    fill_norm(B.data(), cfg.d * cfg.r_o, xavier_std(cfg.d, cfg.r_o));

    // F: Component 2.2 init
    for (size_t g = 0; g < cfg.c; ++g) {
        W_22[g].randomize(static_cast<unsigned>(rng()));
    }
    double g_std = xavier_std(d_k, cfg.r_feat);
    fill_norm(w_g_22.data(), cfg.c * cfg.r_feat, g_std);
    fill_norm(b_g_22.data(), cfg.c, 0.01);
    if (cfg.T_max_22 > 1) {
        fill_norm(W_chi_22.data(), cfg.c * cfg.r_feat * d_v, xavier_std(cfg.r_feat, d_v));
        fill_norm(W_r_22.data(), cfg.c * cfg.r_feat * 2 * cfg.r_feat, xavier_std(cfg.r_feat, 2 * cfg.r_feat));
        fill_norm(w_rho_22.data(), cfg.c * cfg.r_feat, g_std);
        fill_norm(b_rho_22.data(), cfg.c, 0.01);
        fill_norm(tau_22.data(), cfg.c, 0.5);
        fill_norm(w_u_22.data(), cfg.c, 0.1);
    }
}

// ---- Phase B: Shared Latent Projection (§1.3 Construction 1.5) ----

void TMHACore::project_shared_latent(
    const double* X, size_t n,
    double* Z_Q, double* Z_K, double* Z_V) const
{
    mat_mat_mul(Z_Q, X, Q_factors.U.data(), n, cfg.d, cfg.r);
    mat_mat_mul(Z_K, X, K_factors.U.data(), n, cfg.d, cfg.r);
    mat_mat_mul(Z_V, X, V_factors.U.data(), n, cfg.d, cfg.r);
}

void TMHACore::project_shared_latent_single(
    const double* x,
    double* z_Q, double* z_K, double* z_V) const
{
    // z = x · U,  x: d, U: d×r, z: r
    // z[j] = sum_i x[i] * U[i][j]
    for (size_t j = 0; j < cfg.r; ++j) {
        double sq = 0.0, sk = 0.0, sv = 0.0;
        for (size_t i = 0; i < cfg.d; ++i) {
            sq += x[i] * Q_factors.u(i, j);
            sk += x[i] * K_factors.u(i, j);
            sv += x[i] * V_factors.u(i, j);
        }
        z_Q[j] = sq;
        z_K[j] = sk;
        z_V[j] = sv;
    }
}

// ---- Phase B: Per-Cluster QKV Reconstruction (§1.3) ----

void TMHACore::cluster_scaling_vector(const CPFactors& factors, size_t g,
    double* s_out) const
{
    for (size_t j = 0; j < factors.r; ++j) {
        s_out[j] = factors.lambda[j] * factors.v(g, j);
    }
}

void TMHACore::reconstruct_qkv_single(
    const double* z_Q, const double* z_K, const double* z_V,
    size_t g, double* q, double* k, double* v) const
{
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();
    size_t r   = cfg.r;

    // s_Q = λ_Q ⊙ v_{Q,g}
    std::vector<double> s_Q(r);
    cluster_scaling_vector(Q_factors, g, s_Q.data());

    // z_scaled = z ⊙ s
    std::vector<double> zs_Q(r);
    for (size_t j = 0; j < r; ++j) zs_Q[j] = z_Q[j] * s_Q[j];

    // q = zs_Q · W_Q^T   (W_Q: d_k × r, stored row-major)
    // q[d] = sum_j zs_Q[j] * W_Q[d][j]
    for (size_t d = 0; d < d_k; ++d) {
        q[d] = 0.0;
        for (size_t j = 0; j < r; ++j) {
            q[d] += zs_Q[j] * Q_factors.W[d * r + j];
        }
    }

    if (k) {
        std::vector<double> s_K(r);
        cluster_scaling_vector(K_factors, g, s_K.data());
        std::vector<double> zs_K(r);
        for (size_t j = 0; j < r; ++j) zs_K[j] = z_K[j] * s_K[j];
        for (size_t d = 0; d < d_k; ++d) {
            k[d] = 0.0;
            for (size_t j = 0; j < r; ++j) {
                k[d] += zs_K[j] * K_factors.W[d * r + j];
            }
        }
    }

    if (v) {
        std::vector<double> s_V(r);
        cluster_scaling_vector(V_factors, g, s_V.data());
        std::vector<double> zs_V(r);
        for (size_t j = 0; j < r; ++j) zs_V[j] = z_V[j] * s_V[j];
        for (size_t d = 0; d < d_v; ++d) {
            v[d] = 0.0;
            for (size_t j = 0; j < r; ++j) {
                v[d] += zs_V[j] * V_factors.W[d * r + j];
            }
        }
    }
}

void TMHACore::reconstruct_qkv_batch(
    const double* Z_Q, const double* Z_K, const double* Z_V,
    size_t n, size_t g, double* Q, double* K, double* V) const
{
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();
    size_t r   = cfg.r;

    // Q = (Z_Q ⊙ (1⊗s_Q^⊤)) · W_Q^⊤
    // Z_scaled: n×r, W_Q: d_k×r stored row-major. Need Q = Z_scaled · W_Q^T
    // Q[i][j] = sum_l Z_scaled[i][l] * W_Q[j][l] = sum_l Z_scaled[i][l] * W_data[j*r + l]

    auto compute = [&](double* out, const double* Z, const CPFactors& F,
                       size_t d_out) {
        std::vector<double> s(r);
        cluster_scaling_vector(F, g, s.data());
        std::vector<double> Zs(n * r);
        hadamard_rows(Zs.data(), Z, s.data(), n, r);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d_out; ++j) {
                double val = 0.0;
                for (size_t l = 0; l < r; ++l) {
                    val += Zs[i * r + l] * F.W[j * r + l];
                }
                out[i * d_out + j] = val;
            }
        }
    };

    if (Q) compute(Q, Z_Q, Q_factors, d_k);
    if (K) compute(K, Z_K, K_factors, d_k);
    if (V) compute(V, Z_V, V_factors, d_v);
}

// ---- Phase C: Cross-Head Recurrence (§1.4 Definition 1.6) ----

void TMHACore::condition_query(double* q_tilde, const double* q,
    const double* H) const
{
    // q̃ = q + W_read · H   (Spec eq. §1.4)
    std::vector<double> bias(cfg.d_k(), 0.0);
    mat_vec_mul(bias.data(), W_read.data(), H, cfg.d_k(), cfg.r_h);
    for (size_t i = 0; i < cfg.d_k(); ++i) {
        q_tilde[i] = q[i] + bias[i];
    }
}

void TMHACore::update_cross_head_state(
    double* H_new, const double* H_prev, const double* o) const
{
    // ψ(o) = W_psi · o  (compression ℝ^{d_v} → ℝ^{r_h})
    std::vector<double> psi(cfg.r_h, 0.0);
    mat_vec_mul(psi.data(), W_psi.data(), o, cfg.r_h, cfg.d_v());

    // [H_prev; ψ(o)]  (concatenation)
    std::vector<double> concat(2 * cfg.r_h);
    std::memcpy(concat.data(), H_prev, cfg.r_h * sizeof(double));
    std::memcpy(concat.data() + cfg.r_h, psi.data(), cfg.r_h * sizeof(double));

    // W_cross · [H_prev; ψ(o)]
    std::vector<double> update(cfg.r_h, 0.0);
    mat_vec_mul(update.data(), W_cross.data(), concat.data(), cfg.r_h, 2 * cfg.r_h);

    // pre-LN: H_prev + update
    std::vector<double> pre_ln(cfg.r_h);
    for (size_t i = 0; i < cfg.r_h; ++i) {
        pre_ln[i] = H_prev[i] + update[i];
    }

    // LayerNorm
    layer_norm(H_new, pre_ln.data(), cfg.r_h);
}

// ---- Phase D: Uncertainty-Adaptive Routing (§1.5) ----

void TMHACore::compute_router_logits(
    const double* x, double uncertainty, double* logits) const
{
    // r_i = W_router · [x_i; LN(Σ_i)] + b_router
    std::vector<double> input(cfg.d + 1);
    std::memcpy(input.data(), x, cfg.d * sizeof(double));
    input[cfg.d] = uncertainty;

    mat_vec_mul(logits, W_router.data(), input.data(), cfg.c, cfg.d + 1);
    for (size_t g = 0; g < cfg.c; ++g) {
        logits[g] += b_router[g];
    }
}

void TMHACore::compute_routing(double* pi, size_t* active_set,
    size_t& n_active, const double* x, double uncertainty) const
{
    std::vector<double> logits(cfg.c);
    compute_router_logits(x, uncertainty, logits.data());

    // Apply temperature
    for (size_t g = 0; g < cfg.c; ++g) logits[g] /= T;

    // Softmax
    softmax(pi, logits.data(), cfg.c);

    // Hard sparsification: A_i = TopK({g | π_{i,g} ≥ τ}, k_max)
    // with guarantee 1 ≤ |A_i| ≤ k_max
    std::vector<size_t> candidates;
    for (size_t g = 0; g < cfg.c; ++g) {
        if (pi[g] >= cfg.tau) {
            candidates.push_back(g);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
        [pi](size_t a, size_t b) { return pi[a] > pi[b]; });

    n_active = std::min(candidates.size(), cfg.k_max);

    // Ensure at least 1 cluster is always active
    if (n_active == 0 && cfg.c > 0) {
        active_set[0] = std::max_element(pi, pi + cfg.c) - pi;
        n_active = 1;
        return;
    }

    for (size_t i = 0; i < n_active; ++i) {
        active_set[i] = candidates[i];
    }
}

// ---- Phase E: Output Fusion (§1.6 Definition 1.9) ----

void TMHACore::accumulate_output(double* p, const double* o,
    size_t g, double pi_g) const
{
    // p += π_{i,g} · (o · A_g)
    size_t d_v = cfg.d_v();
    const double* A_g_ptr = A_g.data() + g * d_v * cfg.r_o;

    for (size_t j = 0; j < cfg.r_o; ++j) {
        double compressed = 0.0;
        for (size_t i = 0; i < d_v; ++i) {
            compressed += o[i] * A_g_ptr[i * cfg.r_o + j];
        }
        p[j] += pi_g * compressed;
    }
}

void TMHACore::reconstruct_output(double* o_i, const double* p) const
{
    // o_i = p · B^T
    for (size_t i = 0; i < cfg.d; ++i) {
        o_i[i] = 0.0;
        for (size_t j = 0; j < cfg.r_o; ++j) {
            o_i[i] += p[j] * B[i * cfg.r_o + j];
        }
    }
}

// ---- Phase F: Bidirectional Forward Pass (§2.1) ----
// Uses C2.2 feature-map linear attention with precomputed M_g, z_g per cluster

std::vector<double> TMHACore::forward_bidirectional(
    const double* X, size_t n, const double* Sigma) const
{
    size_t d   = cfg.d;
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();
    size_t r_feat = cfg.r_feat;

    // Pass A: Shared latent projection (computed once)
    std::vector<double> Z_Q(n * cfg.r), Z_K(n * cfg.r), Z_V(n *cfg.r);
    project_shared_latent(X, n, Z_Q.data(), Z_K.data(), Z_V.data());

    // Precompute all clusters' K, V (Pass B)
    std::vector<std::vector<double>> all_K(cfg.c);
    std::vector<std::vector<double>> all_V(cfg.c);
    for (size_t g = 0; g < cfg.c; ++g) {
        all_K[g].resize(n * d_k);
        all_V[g].resize(n * d_v);
        reconstruct_qkv_batch(Z_Q.data(), Z_K.data(), Z_V.data(),
            n, g, nullptr, all_K[g].data(), all_V[g].data());
        // Note: reconstruct_qkv_batch reads Z_K for K and Z_V for V.
        // We pass Z_V for both Q and V slots to avoid re-reading.
    }

    // Precompute feature maps and aggregates per cluster
    // M_g = Φ(K_g)^T · V_g  ∈ ℝ^{r_feat × d_v}
    // z_g = sum_j Φ(k_{g,j})  ∈ ℝ^{r_feat}
    std::vector<std::vector<double>> M(cfg.c);
    std::vector<std::vector<double>> z(cfg.c);
    for (size_t g = 0; g < cfg.c; ++g) {
        M[g].assign(r_feat * d_v, 0.0);
        z[g].assign(r_feat, 0.0);
        std::vector<double> phi_k(r_feat);
        for (size_t j = 0; j < n; ++j) {
            ::feature_map(W_22[g], all_K[g].data() + j * d_k,
                          phi_k.data(), d_k, r_feat);
            for (size_t a = 0; a < r_feat; ++a) {
                z[g][a] += phi_k[a];
                for (size_t b = 0; b < d_v; ++b)
                    M[g][a * d_v + b] += phi_k[a] * all_V[g][j * d_v + b];
            }
        }
    }

    // Pass C: Routing for each token
    std::vector<std::vector<double>> pi_all(n);
    std::vector<std::vector<size_t>> active_all(n);
    for (size_t i = 0; i < n; ++i) {
        double uncertainty = Sigma ? Sigma[i] : 0.0;
        pi_all[i].resize(cfg.c);
        active_all[i].resize(cfg.k_max);
        size_t n_active = 0;
        compute_routing(pi_all[i].data(), active_all[i].data(), n_active,
            X + i * d, uncertainty);
        active_all[i].resize(n_active);
    }

    // Pass D + E: Cross-head evaluation and output accumulation
    std::vector<double> O(n * d, 0.0);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> H(cfg.r_h, 0.0);
        std::vector<double> p(cfg.r_o, 0.0);

        auto& active = active_all[i];
        std::sort(active.begin(), active.end(),
            [&pi = pi_all[i]](size_t a, size_t b) { return pi[a] > pi[b]; });

        for (size_t g : active) {
            // q_{i,g}
            std::vector<double> q_g(d_k);
            reconstruct_qkv_single(Z_Q.data() + i * cfg.r,
                Z_K.data() + i * cfg.r, Z_V.data() + i * cfg.r,
                g, q_g.data(), nullptr, nullptr);

            // Condition query with cross-head state
            std::vector<double> q_tilde(d_k);
            condition_query(q_tilde.data(), q_g.data(), H.data());

            // Feature-map linear attention
            std::vector<double> phi_qt(r_feat);
            ::feature_map(W_22[g], q_tilde.data(), phi_qt.data(), d_k, r_feat);

            // o_g = M_g^T · φ(q̃) / (z_g · φ(q̃) + δ)
            std::vector<double> attn_out(d_v, 0.0);
            double denom = ::vec_dot(phi_qt.data(), z[g].data(), r_feat) + delta_22;
            for (size_t a = 0; a < r_feat; ++a) {
                double pqa = phi_qt[a];
                if (pqa == 0.0) continue;
                for (size_t b = 0; b < d_v; ++b)
                    attn_out[b] += M[g][a * d_v + b] * pqa;
            }
            for (size_t b = 0; b < d_v; ++b)
                attn_out[b] /= denom;

            // Update cross-head state
            std::vector<double> H_new(cfg.r_h);
            update_cross_head_state(H_new.data(), H.data(), attn_out.data());
            H.swap(H_new);

            // Accumulate into low-rank output
            accumulate_output(p.data(), attn_out.data(), g, pi_all[i][g]);
        }

        reconstruct_output(O.data() + i * d, p.data());
    }

    return O;
}

// ---- Phase G: Causal Forward Step (§2.2, Algorithm 2.1) ----
// Uses Component 2.2's sgkd_sdpa_step (T_max=1) or rsauq_sdpa_step (T_max>1)

AttentionOutput TMHACore::forward_causal_step(
    const double* x, double uncertainty,
    CausalState& state) const
{
    size_t d   = cfg.d;
    size_t d_k = cfg.d_k();
    size_t d_v = cfg.d_v();

    // Step 1: Shared latent projection for single token
    std::vector<double> z_Q(cfg.r), z_K(cfg.r), z_V(cfg.r);
    project_shared_latent_single(x, z_Q.data(), z_K.data(), z_V.data());

    // Auto-initialize SGKD states if needed
    if (state.sgkd_states.size() != cfg.c ||
        (state.sgkd_states.size() > 0 && state.sgkd_states[0].S_i.size() == 0)) {
        state.sgkd_states.resize(cfg.c);
        for (size_t g = 0; g < cfg.c; ++g)
            state.sgkd_states[g] = SGKDState<double>(cfg.r_feat, d_v, cfg.w_local, d_k);
    }

    // Step 2: Uncertainty-adaptive routing
    std::vector<double> pi(cfg.c);
    std::vector<size_t> active_set(cfg.k_max);
    size_t n_active = 0;
    compute_routing(pi.data(), active_set.data(), n_active, x, uncertainty);

    // Sort active clusters by π descending
    std::sort(active_set.data(), active_set.data() + n_active,
        [&pi](size_t a, size_t b) { return pi[a] > pi[b]; });

    // Step 3: Cross-head recurrent evaluation
    std::vector<double> H = state.H;  // carry from previous token step
    std::vector<double> p(cfg.r_o, 0.0);
    double total_uncertainty = 0.0;

    for (size_t ai = 0; ai < n_active; ++ai) {
        size_t g = active_set[ai];

        // Reconstruct head-specific Q, K, V
        std::vector<double> q_g(d_k), k_g(d_k), v_g(d_v);
        reconstruct_qkv_single(z_Q.data(), z_K.data(), z_V.data(),
            g, q_g.data(), k_g.data(), v_g.data());

        // Cross-head conditioning: q̃ = q + W_read · H
        std::vector<double> q_tilde(d_k);
        condition_query(q_tilde.data(), q_g.data(), H.data());

        // --- Component 2.2 integration ---
        std::vector<double> o_g(d_v);

        if (cfg.T_max_22 <= 1) {
            // SGKD-SDPA mode: O(1) linear attention
            ::sgkd_sdpa_step(o_g.data(), state.sgkd_states[g],
                q_tilde.data(), k_g.data(), v_g.data(),
                W_22[g],
                w_g_22.data() + g * cfg.r_feat, b_g_22[g],
                d_k, cfg.r_feat, d_v, delta_22);
        } else {
            // RSA-UQ-ACD mode: recurrent refinement with uncertainty
            std::vector<double> sigma_g(d_v);
            size_t T_i = 0;
            ::rsauq_sdpa_step(o_g.data(), sigma_g.data(), T_i,
                q_tilde.data(), k_g.data(), v_g.data(),
                state.sgkd_states[g], W_22[g],
                W_chi_22.data() + g * cfg.r_feat * d_v,
                W_r_22.data() + g * cfg.r_feat * 2 * cfg.r_feat,
                w_rho_22.data() + g * cfg.r_feat, b_rho_22[g],
                tau_22[g],
                w_g_22.data() + g * cfg.r_feat, b_g_22[g],
                w_u_22[g],
                cfg.T_max_22, d_k, cfg.r_feat, d_v,
                delta_22, sigma_sq_min_22);

            // Mean-pool uncertainty for router
            double mean_sigma = 0.0;
            for (size_t j = 0; j < d_v; ++j) mean_sigma += sigma_g[j];
            total_uncertainty += mean_sigma / (double)d_v;
        }

        // Update cross-head state: H ← LN(H + W_cross · [H; ψ(o)])
        std::vector<double> H_new(cfg.r_h);
        update_cross_head_state(H_new.data(), H.data(), o_g.data());
        H.swap(H_new);

        // Accumulate low-rank output: p += π_g · (o_g · A_g)
        accumulate_output(p.data(), o_g.data(), g, pi[g]);
    }

    // Step 4: Output reconstruction: o_i = p · B^⊤
    AttentionOutput result;
    result.o.resize(d, 0.0);
    reconstruct_output(result.o.data(), p.data());

    // Uncertainty: from RSA-UQ-ACD if available, else default
    if (cfg.T_max_22 <= 1) {
        result.uncertainty = 0.5;
    } else {
        result.uncertainty = total_uncertainty / std::max(n_active, size_t(1));
    }

    // Step 5: State update: H_i ← H
    state.H = H;

    return result;
}

// ---- Phase H: GQA Constraint (§1.7, Theorem 1.10) ----
// Constrain V_K and V_V to have only g < h unique rows

void TMHACore::apply_gqa_constraint(size_t g) {
    if (g == 0 || g > cfg.h) return;
    size_t groups = std::min(g, cfg.c);
    for (size_t group = 0; group < groups; ++group) {
        size_t start = (group * cfg.c) / groups;
        size_t end   = ((group + 1) * cfg.c) / groups;
        // Copy prototype row to all rows in the group
        for (size_t row = start + 1; row < end; ++row) {
            for (size_t j = 0; j < cfg.r; ++j) {
                K_factors.v(row, j) = K_factors.v(start, j);
                V_factors.v(row, j) = V_factors.v(start, j);
            }
        }
    }
}

} // namespace tmha
