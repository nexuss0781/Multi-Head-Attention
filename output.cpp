// ==========================================================================
// TMHA-CRUS — Tensorized Multi-Head Attention with Cross-Head Recurrence
//              and Uncertainty-Adaptive Routing
//
// Single-header C/C++ API — include once, call directly.
//
//   #include "output.cpp"
//
// Provides:
//   - Full TMHA-CRUS forward pass (bidirectional & causal)
//   - Lifecycle management
//   - Causal state management for autoregressive streaming
//   - Parameter save/load
//   - GQA constraint
//   - Standard MHA reference (StdMHA) for comparison
// ==========================================================================

#ifndef NEX_TMHA_OUTPUT_CPP
#define NEX_TMHA_OUTPUT_CPP

#include "core.hpp"
#include "standard_mha.hpp"

#include <cstring>
#include <new>

// ===========================================================================
// C-compatible configuration structs
// ===========================================================================

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TMHA_Config {
    size_t d;
    size_t h;
    size_t c;
    size_t r;
    size_t r_o;
    size_t r_h;
    size_t k_max;
    double tau;
    double T;
    size_t r_feat;
    size_t w_local;
    size_t T_max_22;
} TMHA_Config;

typedef struct TMHA_Handle TMHA_Handle;

typedef struct TMHA_CausalState TMHA_CausalState;

#ifdef __cplusplus
} // extern "C"
#endif

// ===========================================================================
// C++ implementation
// ===========================================================================

#ifdef __cplusplus

// --- TMHA Handle ---

struct TMHA_Handle {
    tmha::TMHACore core;
    TMHA_Handle(const TMHA_Config& cfg)
        : core(make_config(cfg)) {}
    static tmha::Config make_config(const TMHA_Config& cfg) {
        tmha::Config c;
        c.d = cfg.d; c.h = cfg.h; c.c = cfg.c;
        c.r = cfg.r; c.r_o = cfg.r_o; c.r_h = cfg.r_h;
        c.k_max = cfg.k_max; c.tau = cfg.tau; c.T = cfg.T;
        c.r_feat = cfg.r_feat; c.w_local = cfg.w_local;
        c.T_max_22 = cfg.T_max_22;
        return c;
    }
};

struct TMHA_CausalState {
    tmha::CausalState state;
};

// ===========================================================================
// Lifecycle
// ===========================================================================

inline TMHA_Handle* tmha_create(const TMHA_Config* config) {
    if (!config) return nullptr;
    tmha::Config c = TMHA_Handle::make_config(*config);
    if (!c.valid()) return nullptr;
    void* mem = ::operator new(sizeof(TMHA_Handle));
    return ::new (mem) TMHA_Handle(*config);
}

inline void tmha_destroy(TMHA_Handle* handle) {
    if (handle) {
        handle->~TMHA_Handle();
        ::operator delete(handle);
    }
}

inline void tmha_init(TMHA_Handle* handle, unsigned seed) {
    if (!handle) return;
    std::mt19937 rng(seed);
    handle->core.init(rng);
}

// ===========================================================================
// Forward — Bidirectional
// ===========================================================================

inline void tmha_forward_bidirectional(TMHA_Handle* handle,
    const double* X, size_t n, const double* Sigma, double* O)
{
    if (!handle || !X || !O) return;
    auto result = handle->core.forward_bidirectional(X, n, Sigma);
    std::memcpy(O, result.data(), result.size() * sizeof(double));
}

// ===========================================================================
// Forward — Causal (autoregressive step)
// ===========================================================================

inline TMHA_CausalState* tmha_causal_state_create(const TMHA_Handle* handle) {
    if (!handle) return nullptr;
    void* mem = ::operator new(sizeof(TMHA_CausalState));
    auto* s = ::new (mem) TMHA_CausalState();
    s->state.H.resize(handle->core.cfg.r_h, 0.0);
    return s;
}

inline void tmha_causal_state_destroy(TMHA_CausalState* state) {
    if (state) {
        state->~TMHA_CausalState();
        ::operator delete(state);
    }
}

inline void tmha_causal_state_reset(TMHA_CausalState* state) {
    if (!state) return;
    std::fill(state->state.H.begin(), state->state.H.end(), 0.0);
    state->state.sgkd_states.clear();
}

inline void tmha_forward_causal_step(TMHA_Handle* handle,
    const double* x, double uncertainty,
    TMHA_CausalState* state,
    double* o, double* out_uncertainty)
{
    if (!handle || !x || !state || !o) return;
    auto result = handle->core.forward_causal_step(x, uncertainty, state->state);
    std::memcpy(o, result.o.data(), result.o.size() * sizeof(double));
    if (out_uncertainty) *out_uncertainty = result.uncertainty;
}

// ===========================================================================
// Parameter save/load
// ===========================================================================

inline size_t tmha_total_params(const TMHA_Handle* handle) {
    if (!handle) return 0;
    const auto& c = handle->core;
    size_t n = 0;
    auto add = [&](const std::vector<double>& v) { n += v.size(); };
    add(c.Q_factors.U); add(c.Q_factors.V);
    add(c.Q_factors.W); add(c.Q_factors.lambda);
    add(c.K_factors.U); add(c.K_factors.V);
    add(c.K_factors.W); add(c.K_factors.lambda);
    add(c.V_factors.U); add(c.V_factors.V);
    add(c.V_factors.W); add(c.V_factors.lambda);
    add(c.W_cross); add(c.W_psi); add(c.W_read);
    add(c.W_router); add(c.b_router);
    add(c.A_g); add(c.B);
    add(c.w_g_22); add(c.b_g_22);
    add(c.W_chi_22); add(c.W_r_22);
    add(c.w_rho_22); add(c.b_rho_22);
    add(c.tau_22); add(c.w_u_22);
    for (const auto& w : c.W_22) n += w.v.size();
    return n;
}

inline size_t tmha_total_param_bytes(const TMHA_Handle* handle) {
    return tmha_total_params(handle) * sizeof(double);
}

inline void tmha_save_params(const TMHA_Handle* handle, double* buffer) {
    if (!handle || !buffer) return;
    const auto& c = handle->core;
    double* ptr = buffer;
    auto save = [&](const std::vector<double>& v) {
        std::memcpy(ptr, v.data(), v.size() * sizeof(double));
        ptr += v.size();
    };
    save(c.Q_factors.U); save(c.Q_factors.V);
    save(c.Q_factors.W); save(c.Q_factors.lambda);
    save(c.K_factors.U); save(c.K_factors.V);
    save(c.K_factors.W); save(c.K_factors.lambda);
    save(c.V_factors.U); save(c.V_factors.V);
    save(c.V_factors.W); save(c.V_factors.lambda);
    save(c.W_cross); save(c.W_psi); save(c.W_read);
    save(c.W_router); save(c.b_router);
    save(c.A_g); save(c.B);
    save(c.w_g_22); save(c.b_g_22);
    save(c.W_chi_22); save(c.W_r_22);
    save(c.w_rho_22); save(c.b_rho_22);
    save(c.tau_22); save(c.w_u_22);
    for (const auto& w : c.W_22) {
        std::memcpy(ptr, w.v.data(), w.v.size() * sizeof(double));
        ptr += w.v.size();
    }
}

inline void tmha_load_params(TMHA_Handle* handle, const double* buffer) {
    if (!handle || !buffer) return;
    auto& c = handle->core;
    const double* ptr = buffer;
    auto load = [&](std::vector<double>& v) {
        std::memcpy(v.data(), ptr, v.size() * sizeof(double));
        ptr += v.size();
    };
    load(c.Q_factors.U); load(c.Q_factors.V);
    load(c.Q_factors.W); load(c.Q_factors.lambda);
    load(c.K_factors.U); load(c.K_factors.V);
    load(c.K_factors.W); load(c.K_factors.lambda);
    load(c.V_factors.U); load(c.V_factors.V);
    load(c.V_factors.W); load(c.V_factors.lambda);
    load(c.W_cross); load(c.W_psi); load(c.W_read);
    load(c.W_router); load(c.b_router);
    load(c.A_g); load(c.B);
    load(c.w_g_22); load(c.b_g_22);
    load(c.W_chi_22); load(c.W_r_22);
    load(c.w_rho_22); load(c.b_rho_22);
    load(c.tau_22); load(c.w_u_22);
    for (auto& w : c.W_22) {
        std::memcpy(w.v.data(), ptr, w.v.size() * sizeof(double));
        ptr += w.v.size();
    }
}

// ===========================================================================
// GQA
// ===========================================================================

inline void tmha_apply_gqa_constraint(TMHA_Handle* handle, size_t g) {
    if (!handle) return;
    handle->core.apply_gqa_constraint(g);
}

#endif // __cplusplus
#endif // NEX_TMHA_OUTPUT_CPP
