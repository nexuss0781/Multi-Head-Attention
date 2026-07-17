#pragma once

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>
#include <cstring>

// Include Component 2.2 (all template implementations inline)
#include "../Component-2.2_Scaled-Dot-Product-Attention/Core.hpp"

namespace tmha {

// ============================================================================
// Phase A — Configuration & Core Data Structures (§1.1, §1.2)
// ============================================================================

struct Config {
    size_t d      = 4096;
    size_t h      = 64;
    size_t c      = 8;
    size_t r      = 64;
    size_t r_o    = 64;
    size_t r_h    = 64;
    size_t k_max  = 4;
    double tau    = 0.05;
    double T      = 1.0;

    // Component 2.2 parameters
    size_t r_feat   = 32;   // feature map rank (must be even, ≤ 2*d_k)
    size_t w_local  = 64;   // local exact window size
    size_t T_max_22 = 1;    // refinement depth (1 = SGKD-SDPA, >1 = RSA-UQ-ACD)

    size_t d_k() const { return d / h; }
    size_t d_v() const { return d / h; }

    // Effective r_feat capped to 2*d_k (Householder constraint)
    size_t r_feat_eff() const { return std::min(r_feat, 2 * d_k()); }

    bool valid() const {
        return d > 0 && h > 0 && c > 0 && r > 0 && r_o > 0 && r_h > 0
            && k_max > 0 && k_max <= c
            && d % h == 0 && tau >= 0.0 && tau <= 1.0 && T > 0.0
            && r_feat > 0 && r_feat % 2 == 0
            && w_local > 0 && T_max_22 > 0;
    }
};

struct CPFactors {
    size_t d    = 0;
    size_t rows = 0;
    size_t r    = 0;
    size_t d_k  = 0;

    std::vector<double> U;
    std::vector<double> V;
    std::vector<double> W;
    std::vector<double> lambda;

    size_t idx_U(size_t i, size_t j) const { return i * r + j; }
    size_t idx_V(size_t i, size_t j) const { return i * r + j; }
    size_t idx_W(size_t i, size_t j) const { return i * r + j; }

    double& u(size_t i, size_t j) { return U[idx_U(i, j)]; }
    double& v(size_t i, size_t j) { return V[idx_V(i, j)]; }
    double& w(size_t i, size_t j) { return W[idx_W(i, j)]; }

    const double& u(size_t i, size_t j) const { return U[idx_U(i, j)]; }
    const double& v(size_t i, size_t j) const { return V[idx_V(i, j)]; }
    const double& w(size_t i, size_t j) const { return W[idx_W(i, j)]; }

    CPFactors() = default;
    CPFactors(size_t d_, size_t rows_, size_t r_, size_t d_k_)
        : d(d_), rows(rows_), r(r_), d_k(d_k_)
        , U(d_ * r_, 0.0), V(rows_ * r_, 0.0), W(d_k_ * r_, 0.0), lambda(r_, 1.0) {}
};

struct CrossHeadState {
    std::vector<double> h;
    CrossHeadState() = default;
    explicit CrossHeadState(size_t r_h) : h(r_h, 0.0) {}
};

struct CausalState {
    std::vector<double> H;
    std::vector<SGKDState<double>> sgkd_states;
};

struct AttentionOutput {
    std::vector<double> o;
    double uncertainty = 0.0;
};

// ============================================================================
// Matrix utility functions
// ============================================================================

inline size_t mat_idx(size_t i, size_t j, size_t cols) {
    return i * cols + j;
}

void mat_vec_mul(double* y, const double* A, const double* x,
                         size_t m, size_t n);
void mat_mat_mul(double* C, const double* A, const double* B,
                         size_t m, size_t k, size_t n);
void hadamard_rows(double* C, const double* A, const double* s,
                           size_t m, size_t n);
double dot(const double* a, const double* b, size_t n);
void axpy(double* y, double alpha, const double* x, size_t n);
void softmax(double* p, const double* z, size_t n);
void layer_norm(double* y, const double* x, size_t n, double eps = 1e-6);
std::vector<size_t> topk_indices(const double* values, size_t n, size_t k);

// ============================================================================
// TMHACore — all 15 phases (§1–§7) + full Component 2.2 integration
// ============================================================================

class TMHACore {
public:
    Config cfg;

    // B: CP factors for Q, K, V
    CPFactors Q_factors;
    CPFactors K_factors;
    CPFactors V_factors;

    // C: Cross-head recurrence parameters (§1.4)
    std::vector<double> W_cross;  // r_h × 2r_h
    std::vector<double> W_psi;    // r_h × d_v
    std::vector<double> W_read;   // d_k × r_h

    // D: Router parameters (§1.5)
    std::vector<double> W_router; // c × (d + 1)
    std::vector<double> b_router; // c
    double T;

    // E: Output fusion parameters (§1.6)
    std::vector<double> A_g;      // c × d_v × r_o
    std::vector<double> B;        // d × r_o

    // Component 2.2 parameters (per cluster)
    std::vector<StiefelHouseholder<double>> W_22; // c × Stiefel(d_k, r_feat/2)
    std::vector<double> w_g_22;                   // c × r_feat
    std::vector<double> b_g_22;                   // c (scalar bias)
    // RSA-UQ-ACD parameters (only used when cfg.T_max_22 > 1)
    std::vector<double> W_chi_22;  // c × r_feat × d_v
    std::vector<double> W_r_22;    // c × r_feat × 2*r_feat
    std::vector<double> w_rho_22;  // c × r_feat
    std::vector<double> b_rho_22;  // c (scalar)
    std::vector<double> tau_22;    // c (scalar halting threshold)
    std::vector<double> w_u_22;    // c (scalar uncertainty gate weight)

    double delta_22 = 1e-6;
    double sigma_sq_min_22 = 1e-10;

    explicit TMHACore(const Config& config);
    void init(std::mt19937& rng);

    // ---- Phase B: Shared Latent Projection (§1.3) ----
    void project_shared_latent(const double* X, size_t n,
        double* Z_Q, double* Z_K, double* Z_V) const;
    void project_shared_latent_single(const double* x,
        double* z_Q, double* z_K, double* z_V) const;

    // ---- Phase B: Per-Cluster QKV (§1.3 Construction 1.5) ----
    // s_M^{(g)} = λ_M ⊙ v_{M,g}
    void cluster_scaling_vector(const CPFactors& factors, size_t g,
        double* s_out) const;
    // M_g = (Z_M ⊙ (1⊗s^⊤)) · W_M^⊤   (single token)
    void reconstruct_qkv_single(const double* z_Q, const double* z_K, const double* z_V,
        size_t g, double* q, double* k, double* v) const;
    // Batched version
    void reconstruct_qkv_batch(const double* Z_Q, const double* Z_K, const double* Z_V,
        size_t n, size_t g, double* Q, double* K, double* V) const;

    // ---- Phase C: Cross-Head Recurrence (§1.4) ----
    void condition_query(double* q_tilde, const double* q, const double* H) const;
    void update_cross_head_state(double* H_new, const double* H_prev,
        const double* o) const;

    // ---- Phase D: Uncertainty-Adaptive Routing (§1.5) ----
    void compute_router_logits(const double* x, double uncertainty,
        double* logits) const;
    void compute_routing(double* pi, size_t* active_set, size_t& n_active,
        const double* x, double uncertainty) const;

    // ---- Phase E: Output Fusion (§1.6) ----
    void accumulate_output(double* p, const double* o, size_t g, double pi_g) const;
    void reconstruct_output(double* o_i, const double* p) const;

    // ---- Phase F: Bidirectional Forward Pass (§2.1) ----
    std::vector<double> forward_bidirectional(
        const double* X, size_t n, const double* Sigma) const;

    // ---- Phase G: Causal Forward Step (§2.2, Algorithm 2.1) ----
    AttentionOutput forward_causal_step(
        const double* x, double uncertainty, CausalState& state) const;

    // ---- Phase H: GQA Constraint (§1.7) ----
    void apply_gqa_constraint(size_t g);

    // ---- Parameter Access for Testing ----
    const double* get_W_router() const { return W_router.data(); }
    const double* get_b_router() const { return b_router.data(); }
    const double* get_W_cross() const { return W_cross.data(); }
    const double* get_W_psi() const { return W_psi.data(); }
    const double* get_W_read() const { return W_read.data(); }
    const CPFactors& get_Q() const { return Q_factors; }
    const CPFactors& get_K() const { return K_factors; }
    const CPFactors& get_V() const { return V_factors; }
    const std::vector<double>& get_A_g() const { return A_g; }
    const std::vector<double>& get_B() const { return B; }
    double get_T() const { return T; }
    const StiefelHouseholder<double>& get_W_22(size_t g) const { return W_22[g]; }
    size_t get_r_feat() const { return cfg.r_feat; }
    size_t get_w_local() const { return cfg.w_local; }
    size_t get_T_max_22() const { return cfg.T_max_22; }
};

// ============================================================================
// Test utility
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string detail;
};

inline void print_results(const std::vector<TestResult>& results) {
    size_t passed = 0, total = results.size();
    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS]" : "[FAIL]") << " "
                  << r.name << "\n";
        if (!r.passed) std::cout << "  └─ " << r.detail << "\n";
        if (r.passed) ++passed;
    }
    std::cout << "\n========================================\n"
              << "Results: " << passed << "/" << total << " passed ("
              << std::fixed << std::setprecision(1)
              << 100.0 * passed / total << "%)\n"
              << "========================================\n";
}

} // namespace tmha
