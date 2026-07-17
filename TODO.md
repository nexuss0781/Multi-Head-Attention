# TODO.md — Component 2.3 TMHA-CRUS Implementation

## Phase A: Core Data Structures & Configuration

- [ ] **A.1** Define hyperparameter struct/class: `d, h, c, d_k, d_v, r, r_o, r_h, k_max, tau, T`
- [ ] **A.2** Define state container for causal mode: `(H_i ∈ ℝ^{r_h}, S_i^{(2.2)} for all c clusters)`
- [ ] **A.3** Define symbol table constants mapping (per SPEC §1.1)

## Phase B: CP-Decomposed Tensorized Head Subspaces (§1.3)

- [ ] **B.1** Initialize CP factors: `U_M ∈ ℝ^{d×r}, V_M ∈ ℝ^{h×r}, W_M ∈ ℝ^{d_k×r}, λ_M ∈ ℝ^r` for M∈{Q,K,V}
- [ ] **B.2** Implement shared latent projection: `Z_M = X · U_M` (SPEC eq. §1.3)
- [ ] **B.3** Implement per-cluster scaling vector: `s_M^{(g)} = λ_M ⊙ v_{M,g}` (SPEC eq. §1.3)
- [ ] **B.4** Implement per-cluster QKV reconstruction: `M_g = Z_M ⊙ (1_n ⊗ s_g^⊤) · W_M^⊤` (SPEC eq. §1.3)
- [ ] **B.5** Verify parameter reduction: Θ(r·d) vs Θ(d²) (SPEC §1.3 Property 1)
- [ ] **B.6** Verify cache efficiency: U_M, W_M loaded once, v_{M,g} register-resident (SPEC §1.3 Property 2)

## Phase C: Cross-Head Recurrent State — CHRS (§1.4)

- [ ] **C.1** Define state update eq: `h_{i,j} = LN(h_{i,j-1} + W_cross · [h_{i,j-1}; ψ(o_{i,j})])` (SPEC eq. §1.4)
- [ ] **C.2** Implement compression projection: `ψ(o) = W_ψ · o`, ℝ^{d_v} → ℝ^{r_h}
- [ ] **C.3** Implement read projection: `W_read ∈ ℝ^{d_k × r_h}` for query conditioning
- [ ] **C.4** Implement query conditioning: `q̃_{i,g} = q_{i,g} + W_read · h_{i,g-1}` (SPEC eq. §1.4)
- [ ] **C.5** Verify state causality and boundedness (SPEC §1.4, per test CH-1, CH-3)

## Phase D: Uncertainty-Adaptive Spectral Routing — UASR (§1.5)

- [ ] **D.1** Implement router logit computation: `z_{i,g} = w_router^⊤ · [x_i; LN(Σ_i); μ_g] + b_g` (SPEC eq. §1.5)
- [ ] **D.2** Implement softmax routing distribution: `π_i = softmax(z_i / T)` with learned T
- [ ] **D.3** Implement hard sparsification (inference): `A_i = TopK({g | π_{i,g} ≥ τ}, k_max)` (SPEC eq. §1.5)
- [ ] **D.4** Implement uncertainty gating: low Σ → 1–2 clusters, high Σ → k_max clusters (SPEC §1.5)
- [ ] **D.5** Verify router boundedness and differentiability (SPEC §4.4 tests UR-1 through UR-5)

## Phase E: Hierarchical Spectral Output Fusion — HSOF (§1.6)

- [ ] **E.1** Define per-cluster compressors: `A_g ∈ ℝ^{d_v × r_o}` and shared basis `B ∈ ℝ^{d × r_o}`
- [ ] **E.2** Implement streaming output accumulation: `p_i += π_{i,g} · o_{i,g} · A_g` (SPEC eq. §1.6)
- [ ] **E.3** Implement output reconstruction: `o_i = p_i · B^⊤` (SPEC eq. §1.6)
- [ ] **E.4** Verify convexity preservation and streaming equivalence (SPEC §4.5 tests OF-1 through OF-4)

## Phase F: Bidirectional Forward Pass (§2.1)

- [ ] **F.1** Pass A — Shared latent projection (Z_Q, Z_K, Z_V) (SPEC §2.1 eq.)
- [ ] **F.2** Pass B — Cluster-wise QKV feature preparation (Q_g, K_g, V_g) (SPEC §2.1 eq.)
- [ ] **F.3** Pass C — Uncertainty-adaptive routing (π_i, A_i) (SPEC §2.1 eq.)
- [ ] **F.4** Pass D — Cross-head recurrent evaluation loop over active clusters (SPEC §2.1 eq.)
- [ ] **F.5** Pass E — Output reconstruction (o_i = p_i · B^⊤ → O ∈ ℝ^{n×d}) (SPEC §2.1 eq.)
- [ ] **F.6** End-to-end bidirectional TMHA output shape guarantee: (n, d) (SPEC test SC-1)

## Phase G: Causal (Autoregressive) Forward Pass (§2.2)

- [ ] **G.1** Implement step-by-step state machine with Θ(1) auxiliary memory
- [ ] **G.2** Step 1 — Shared latent projection for single token (z_Q, z_K, z_V) (SPEC Alg 2.1)
- [ ] **G.3** Step 2 — Uncertainty-adaptive routing for single token (SPEC Alg 2.1)
- [ ] **G.4** Step 3 — Cross-head recurrent evaluation loop per token (SPEC Alg 2.1)
- [ ] **G.5** Step 4 — Output reconstruction per token (SPEC Alg 2.1)
- [ ] **G.6** Step 5 — State update: H_i and S_i^{(2.2)} (SPEC Alg 2.1)
- [ ] **G.7** Verify causal convexity: o_i is convex combination of {v_j}_{j≤i} (SPEC Theorem 2.2)

## Phase H: GQA Degenerate Case (§1.7)

- [ ] **H.1** Constrain V_K, V_V to g < h unique rows
- [ ] **H.2** Verify KV-cache reduces to g groups
- [ ] **H.3** Verify functional identity with explicit GQA (SPEC test IC-3)

## Phase I: Structural Correctness Tests (§4.1)

- [ ] **I.1** SC-1: Output shape (n, d) for all n and arbitrary Σ
- [ ] **I.2** SC-2: Causal enforcement via gradient masking
- [ ] **I.3** SC-3: Single-token identity — convex combination weight sum = 1
- [ ] **I.4** SC-4: State boundedness — ‖H_i‖₂ ≤ C after 10⁶ steps
- [ ] **I.5** SC-5: Router boundedness — π_{i,g} ∈ [0,1], Σ π = 1
- [ ] **I.6** SC-6: Active set boundedness — 1 ≤ |A_i| ≤ k_max

## Phase J: Tensorized Projection Invariant Tests (§4.2)

- [ ] **J.1** TP-1: Full-rank equivalence — r = h, V_M = identity → matches standard MHA within 1e-5
- [ ] **J.2** TP-2: CP reconstruction accuracy — Frobenius error < 1e-4
- [ ] **J.3** TP-3: Shared latent correctness — bitwise identical
- [ ] **J.4** TP-4: Gradient reachability — non-zero gradients reach all U_M, V_M, W_M
- [ ] **J.5** TP-5: Head diversity preservation — clusters do not collapse

## Phase K: Cross-Head Recurrence Invariant Tests (§4.3)

- [ ] **K.1** CH-1: State causality — ∂H_i/∂x_j = 0 for j > i
- [ ] **K.2** CH-2: Inter-head gradient flow — ∂o_{i,g}/∂W_cross ≠ 0 for g > 1
- [ ] **K.3** CH-3: State contraction — ‖h_{i,j} - h_{i,j-1}‖ decreases monotonically
- [ ] **K.4** CH-4: Information preservation — CHRS model outperforms isolated-head baseline
- [ ] **K.5** CH-5: Memory independence — peak RAM flat across n ∈ {1k, 128k, 1M}

## Phase L: Uncertainty-Adaptive Routing Tests (§4.4)

- [ ] **L.1** UR-1: Pearson r(Σ_i, |A_i|) > 0.5
- [ ] **L.2** UR-2: Sparsity response — uniform random → k̄ < 2.0; coreference → k̄ > 3.0
- [ ] **L.3** UR-3: Gradient flow through router — ∇W_router non-vanishing
- [ ] **L.4** UR-4: TopK differentiability — converges, π_i entropy < 0.5 bits
- [ ] **L.5** UR-5: No future leakage — ∂π_{i,g}/∂Σ_j = 0 for j > i

## Phase M: Output Fusion Tests (§4.5)

- [ ] **M.1** OF-1: Low-rank fusion fidelity — r_o = d matches exact W_o
- [ ] **M.2** OF-2: Fused convexity — implicit weights W_{ij} ≥ 0, Σ_j W_{ij} = 1
- [ ] **M.3** OF-3: Streaming equivalence — incremental = batched within float tolerance
- [ ] **M.4** OF-4: Output rank preservation — rank(O) ≥ min(n,d)·0.8 on random Gaussian

## Phase N: Performance Verification (§4.6)

- [ ] **N.1** PF-1: Linear time scaling — 2× n → 2× latency
- [ ] **N.2** PF-2: Constant auxiliary memory — peak RAM flat for n ∈ {1k, 128k, 1M}
- [ ] **N.3** PF-4: Parameter memory — total MHA < 2 MB at d=4096, L=80

## Phase O: Integration Tests with Component 2.2 (§4.7)

- [ ] **O.1** IC-1: SGKD-SDPA compatibility
- [ ] **O.2** IC-2: RSA-UQ-ACD compatibility — consumes Σ, emits valid O
- [ ] **O.3** IC-3: GQA equivalence with explicit GQA
- [ ] **O.4** IC-4: Mixed cluster modes — SGKD-SDPA + RSA-UQ-ACD no cross-interference
