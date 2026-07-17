#pragma once

#include "../Component-2.2_Scaled-Dot-Product-Attention/Core.hpp"
#include <vector>
#include <cstddef>
#include <random>

// ============================================================================
// Standard Multi-Head Attention using C2.2 SGRK-UQAD as inner primitive
//
// Each head is an independent C2.2 attention with per-head QKV projections,
// exactly as standard MHA but replacing softmax(QK^T)V with SGRK-UQAD.
// ============================================================================

struct SMHAConfig {
    size_t d      = 4096;  // model dimension
    size_t h      = 64;    // number of heads
    size_t r_feat = 32;    // C2.2 feature rank (even, ≤ 2*d_k)
    size_t w_local = 64;   // C2.2 local window
    size_t T_max   = 1;    // C2.2 refinement depth

    size_t d_k() const { return d / h; }
    size_t d_v() const { return d / h; }
    size_t r_feat_eff() const { return std::min(r_feat, 2 * d_k()); }
    bool valid() const {
        return d > 0 && h > 0 && d % h == 0
            && r_feat > 0 && r_feat % 2 == 0
            && w_local > 0 && T_max > 0;
    }
};

struct SMHACausalState {
    std::vector<SGKDState<double>> sgkd_states;
};

class StdMHA {
public:
    SMHAConfig cfg;

    // Per-head projections: stacked [h × d × d_k] row-major
    std::vector<double> W_Q;  // d × (h * d_k)
    std::vector<double> W_K;  // d × (h * d_k)
    std::vector<double> W_V;  // d × (h * d_v)
    std::vector<double> W_O;  // (h * d_v) × d

    // C2.2 parameters (per head)
    std::vector<StiefelHouseholder<double>> W_22;  // h × Stiefel(d_k, r_feat/2)
    std::vector<double> w_g;                       // h × r_feat
    std::vector<double> b_g;                       // h (scalar)

    StdMHA(const SMHAConfig& config);
    void init(std::mt19937& rng);

    // Bidirectional: X ∈ ℝ^{n × d}  →  ℝ^{n × d}
    std::vector<double> forward_bidirectional(
        const double* X, size_t n) const;

    // Causal step: single token  →  AttentionOutput
    // State auto-initializes on first call.
    struct StepOutput {
        std::vector<double> o; // d
    };
    StepOutput forward_causal_step(
        const double* x, SMHACausalState& state) const;
};
