# Component Pipeline Report — Handoff to KV Cache (2.4)

## Overview

Data flows through five stages before reaching KV Cache. Each stage transforms the representation and exposes a contract for downstream consumers.

```
Input Text
    │
    ▼
Tokenizer (Phase 1) → token IDs
    │
    ▼
Embedder (Phase 1) → X ∈ ℝ^{n × d}
    │
    ▼
Positional Encoding (Phase 1) → X_pe ∈ ℝ^{n × d}
    │
    ▼
Component 2.1 — QKV Projection → Q, K, V ∈ ℝ^{n × h × d_k}
    │
    ▼
Component 2.2 — SGRK-UQAD → attention output per head/cluster
    │
    ▼
Component 2.3 — TMHA-CRUS → O ∈ ℝ^{n × d}
    │
    ▼
Component 2.4 — KV Cache (you are here)
```

---

## Component 2.1 — QKV Projection (SQFP)

**Input:** `X ∈ ℝ^{n × d}` — position-encoded token embeddings.

**Output:** `Q ∈ ℝ^{n × h × d_k}`, `K ∈ ℝ^{n × g × d_k}`, `V ∈ ℝ^{n × g × d_v}`.

**Key details:**
- GQA support: `g` KV groups, `h` query heads (g ≤ h).
- `d_k = d_v = d / h`.
- Projections are Diagonal-Plus-Low-Rank (DPLR) with Butterfly residual — but the contract is identical to dense projections (same shapes, same semantics).
- **KV cache stores `g` groups**, not `h` heads. The expand to `h` heads happens after cache lookup (repeat_interleave).
- RoPE is applied after projection (Q and K both rotated).

**Downstream contract for 2.4:**
```
K_cache[t, g, :]  ← K[t, g * d_k : (g+1) * d_k]   // flat layout
V_cache[t, g, :]  ← V[t, g * d_v : (g+1) * d_v]
```

---

## Component 2.2 — SGRK-UQAD (Scaled Dot-Product Attention)

**Input:** Per-cluster/head `q ∈ ℝ^{d_k}`, `k ∈ ℝ^{d_k}`, `v ∈ ℝ^{d_v}`.

**Output:** `o ∈ ℝ^{d_v}`, plus `σ ∈ ℝ^{d_v}` epistemic uncertainty (RSA-UQ-ACD mode).

**Key details:**
- Two modes: SGKD-SDPA (T_max=1, linear O(1) per step) and RSA-UQ-ACD (T_max>1, adaptive depth with recurrent refinement).
- Feature map `φ: ℝ^{d_k} → ℝ^{r_feat}` via Householder orthogonal parameterization.
- Maintains `SGKDState`: `S_i ∈ ℝ^{r_feat × d_v}`, `Z_i ∈ ℝ^{r_feat}`, local FIFO buffer of size `w_local`.
- Each cluster/head has its own `StiefelHouseholder` + gate params.
- **Causal streaming**: `sgkd_sdpa_step` updates `SGKDState` in-place — each step appends (k, v) to aggregates and FIFO.

**Downstream contract for 2.4:**
- 2.4 does NOT interact directly with 2.2. Attention state (`SGKDState`) is managed inside 2.3 per cluster.
- 2.4 only needs to store and retrieve raw K, V tensors for cache-and-reuse.

---

## Component 2.3 — TMHA-CRUS (Multi-Head Attention)

**Input:** `X ∈ ℝ^{n × d}` (position-encoded tokens), optional per-token `Σ ∈ ℝ^n`.

**Output:** `O ∈ ℝ^{n × d}`.

**Key details:**
- Replaces standard MHA entirely. No per-head W_Q/W_K/W_V matrices.
- CP-factorized shared latent: `z = x · U` (U ∈ ℝ^{d × r}), then per-cluster reconstruction.
- `c` clusters (analogous to heads), each with its own C2.2 attention.
- Cross-head recurrence: `H ∈ ℝ^{r_h}` propagates state across clusters within a token.
- Uncertainty-adaptive routing: top-k clusters per token, with sparsity threshold τ.
- Two forward modes:
  - **Bidirectional** (`forward_bidirectional`): precomputes feature aggregates `M_g, z_g` per cluster for all n tokens, then processes queries with linear attention. O(n) in sequence length.
  - **Causal** (`forward_causal_step`): single-token O(1) step. Maintains `CausalState`:
    - `H ∈ ℝ^{r_h}` — cross-head recurrence carry
    - `sgkd_states[c]` — per-cluster SGKDState (S_i, Z_i, FIFO buffer)

**Parameters (d=4096, h=64, c=8, r=64):** ~1.15M total.

**Downstream contract for 2.4:**
- **CausalState contains the KV cache implicitly** through `sgkd_states[g].S_i` and `sgkd_states[g].Z_i` (global aggregates) plus `sgkd_states[g].buffer` (FIFO of recent (k, v) pairs).
- The raw K, V per cluster/token are **not stored explicitly** after reconstruction — they are consumed by `sgkd_sdpa_step` and folded into aggregates.
- **If 2.4 needs explicit K, V tensors** (e.g., for prefix caching, speculative decoding, or cross-attention), it must either:
  - (a) Capture K, V at the 2.1 output (before CP reconstruction), or
  - (b) Reconstruct per-cluster K_g, V_g from Z_K, Z_V at the desired positions.
- The CP factor reconstruction is: `K_g[t] = (z_K[t] ⊙ λ_K ⊙ v_{K,g}) · W_K^T`.

---

## Data Flow Summary for KV Cache

```
                   2.1                        2.3
Input ──→ QKV Projection ──→ Q, K, V ──→ TMHA-CRUS ──→ O
Stream           │                       │
                 │ K, V per group         │ CausalState (SGKDState per cluster)
                 │ (g groups, n tokens)    │ grows O(1) in n (aggregates + FIFO)
                 ▼                       ▼
          ┌─────────────┐       ┌──────────────────┐
          │  2.4 Cache  │       │ Internal to 2.3  │
          │  raw K/V    │       │ not exposed       │
          └─────────────┘       └──────────────────┘
```

**Recommendation for 2.4:**
- Interface with **Component 2.1 output** for raw K, V caching. Shapes are clean: `(n, g, d_k)` for K, `(n, g, d_v)` for V.
- GQA expand (`repeat_interleave` from g to h) after cache lookup.
- TMHA-CRUS causal mode currently manages its own attention state internally. If 2.4 needs to inject cached tokens into TMHA-CRUS (e.g., prefix caching), the interface point is CausalState (specifically the `sgkd_states` vector and `H` vector).
