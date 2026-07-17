# Component 2.3 — Multi-Head Attention

## Executive Summary

Multi-Head Attention (MHA) is the complete attention sublayer of a Transformer block. It runs `h` independent Scaled Dot-Product Attention operations in parallel — each in a lower-dimensional subspace — then concatenates their outputs and projects back to the model dimension `d`.

The motivation is representational diversity: different heads learn to specialize in different structural patterns (local vs. long-range, syntactic vs. semantic, coreference, positional proximity). A single large attention head loses this specialization capacity.

MHA inherits the O(n²) complexity of SDPA and multiplies it by h heads, though since each head operates in dimension `d_k = d/h`, the total FLOPs remain O(n² · d) — the same as one head at full dimension. The space cost, however, grows as O(n² · h) for naive implementations.

---

## Mathematical Breakdown & Specification

### Definitions

| Symbol | Meaning |
|---|---|
| `X` | Input from residual stream (after RMSNorm), `X ∈ ℝ^{n × d}` |
| `h` | Number of attention heads |
| `d_k = d/h` | Dimension per head (query and key) |
| `d_v = d/h` | Dimension per head (value) |
| `Wq_i, Wk_i, Wv_i` | Per-head projection weights |
| `Wo` | Output projection, `Wo ∈ ℝ^{d × d}` |
| `head_i` | Output of head `i`, `head_i ∈ ℝ^{n × d_v}` |
| `MultiHead(X)` | Full MHA output, `∈ ℝ^{n × d}` |

---

### Forward Pass

```
For each head i = 1 .. h:
  Qᵢ = X · Wq_i          Wq_i ∈ ℝ^{d × d_k}
  Kᵢ = X · Wk_i          Wk_i ∈ ℝ^{d × d_k}
  Vᵢ = X · Wv_i          Wv_i ∈ ℝ^{d × d_v}
  head_i = Attention(Qᵢ, Kᵢ, Vᵢ)   ∈ ℝ^{n × d_v}

MultiHead(X) = Concat(head_1, ..., head_h) · Wo
             = [head_1 | head_2 | ... | head_h] · Wo
               ∈ ℝ^{n × (h·d_v)} → ℝ^{n × d}
```

---

### Batched Implementation (Practical)

In practice, the h projections are not computed sequentially. The full weight matrices are used:

```
Q = X · Wq          Wq ∈ ℝ^{d × (h·d_k)},   Q ∈ ℝ^{n × (h·d_k)}
K = X · Wk          Wk ∈ ℝ^{d × (h·d_k)},   K ∈ ℝ^{n × (h·d_k)}
V = X · Wv          Wv ∈ ℝ^{d × (h·d_v)},   V ∈ ℝ^{n × (h·d_v)}

Reshape:
  Q → (batch, h, n, d_k)
  K → (batch, h, n, d_k)
  V → (batch, h, n, d_v)

Batched SDPA (over h in parallel):
  heads = Attention(Q, K, V)   → (batch, h, n, d_v)

Reshape + project:
  heads → (batch, n, h·d_v) = (batch, n, d)
  Output = heads · Wo           → (batch, n, d)
```

All h heads run as a single batched matmul — hardware-efficient.

---

### GQA Variant (Modern Standard)

With Grouped Query Attention:
```
Q → (batch, h, n, d_k)         h query heads
K → (batch, g, n, d_k)         g KV groups, g < h
V → (batch, g, n, d_v)

Expand K, V:
  K → (batch, h, n, d_k)       by repeating each group h/g times
  V → (batch, h, n, d_v)

Proceed as standard batched MHA
```

Expansion is a view/broadcast — zero memory overhead. KV-cache stores only `g` groups.

---

### Head Specialization (Empirical)

Research analysis of trained MHA layers reveals consistent specialization patterns:

| Head Type | Pattern | Example |
|---|---|---|
| Local heads | Attend to 1–3 adjacent tokens | Syntactic dependencies |
| Strided heads | Attend every k tokens | Positional rhythm |
| Global heads | Attend broadly across full context | Semantic aggregation |
| Coreference heads | Connect pronouns to antecedents | "he" → "John" |
| BOS sink heads | Route excess attention to position 0 | Attention sink phenomenon |
| Rare word heads | Attend to low-frequency, high-information tokens | Factual recall |

These patterns emerge from training — not designed. Different heads in the same layer learn different specializations.

---

### Attention Sink Phenomenon

In trained autoregressive LLMs, one or more heads in early layers concentrate a large fraction of attention weight on the first token (position 0, usually BOS):

```
W_{i,0} >> W_{i,j}   for j > 0   (in affected heads)
```

Interpretation: The first token acts as a "dump" — positions that have no strong preference route attention there rather than uniformly distributing. Removing position 0 destabilizes these heads.

This is an emergent artifact of causal masking and softmax normalization, not a designed feature.

---

### Attention Dropout

During training, dropout can be applied to attention weights W before the weighted sum:

```
W_drop = Dropout(W, p=p_attn)
O = W_drop · V
```

Typical `p_attn = 0.0` in modern LLMs at scale. When used, forces heads to not over-rely on single attention patterns.

---

### Output Projection Initialization

To keep residual stream variance stable at initialization across L layers:
```
Wo initialized with std = 1 / √(2L)    (scaled init, GPT-2 style)
```

Without this, the residual stream variance grows as O(L) at initialization, causing gradient instability in very deep models.

---

## Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Q, K, V projections | O(n · d²) total | O(n · d) |
| Batched SDPA (h heads) | **O(h · n² · d_k) = O(n² · d)** | **O(h · n²)** |
| Output projection Wo | O(n · d²) | O(n · d) |
| **Total MHA** | **O(n² · d + n · d²)** | **O(h · n²)** |

For long sequences where `n >> d`:
- Dominant term: `O(n² · d)` from attention
- For short sequences where `n << d`:
- Dominant term: `O(n · d²)` from projections

**Crossover point:** `n ≈ d` (typically d ≈ 4096, so crossover at n ≈ 4096 tokens)

**Concrete numbers (LLaMA-3 8B, n=128k, d=4096, h=32, d_k=128, L=32):**
```
Per-layer SDPA FLOPs: 2 · n² · d = 2 · (128k)² · 4096 ≈ 137 PFLOPs per layer
Per-layer projection FLOPs: 4 · n · d² ≈ 8.6 TFLOPs per layer

Attention dominates by factor ≈ 16,000× at n=128k
```

🔴 **MHA is O(n²·d) time and O(h·n²) space. Primary bottleneck.**

---

## Detailed Test Specification

### Shape Tests

- [ ] **Output shape:** `MHA(X).shape == (batch, n, d)` for `X ∈ ℝ^{batch × n × d}`
- [ ] **Per-head shape:** Each `head_i ∈ ℝ^{batch × n × d_v}` before concat
- [ ] **Concat shape:** `Concat(heads).shape == (batch, n, h · d_v) == (batch, n, d)`
- [ ] **Batched vs sequential equivalence:** Batched MHA output == sequential per-head computation concatenated, within float tolerance

### Correctness Tests

- [ ] **Causal masking propagated:** MHA output at position `i` depends only on positions `0..i` — verify by zeroing out positions `0..i-1` and confirming output changes
- [ ] **Head independence:** Gradient w.r.t. `Wq_i` does not affect `Wq_j` for `i ≠ j`
- [ ] **GQA correctness:** Heads within the same group produce identical attention patterns (same K,V) — verify `W_i == W_j` for heads i,j sharing a KV group, given same Q
- [ ] **Linearity of output projection:** `MHA(X)_without_Wo` followed by manual `Wo` multiply == `MHA(X)_with_Wo`

### Attention Pattern Tests

- [ ] **Attention weight rows sum to 1:** `‖W.sum(dim=-1) - 1‖ < 1e-4` for all heads and positions
- [ ] **Attention sink emergent:** In a trained model, confirm at least one head has `W[:, :, 0].mean() > 0.3` (BOS position disproportionately attended)
- [ ] **No future attention:** `W[:, :, i, j] == 0` for all `j > i`

### Gradient Tests

- [ ] **Gradient flows through all heads:** Each `Wq_i, Wk_i, Wv_i, Wo` has nonzero gradient after forward+backward
- [ ] **Scaled init effect:** With Wo init std = 1/√(2L), `Var(X + MHA(X)) ≈ Var(X)` at initialization

### Equivalence Tests

- [ ] **Flash Attention equivalence:** `MHA_standard(X) ≈ MHA_flash(X)` within 1e-2 max absolute error in bf16
- [ ] **GQA equals MHA when g=h:** Full GQA with g=h produces identical results to standard MHA

---

## Checklist

- [ ] Executive summary written and reviewed
- [ ] Per-head forward pass documented
- [ ] Batched implementation documented
- [ ] GQA variant documented
- [ ] Head specialization patterns documented
- [ ] Attention sink phenomenon documented
- [ ] Output projection initialization documented
- [ ] Complexity analysis complete with concrete numbers
- [ ] All shape tests written
- [ ] All correctness tests written
- [ ] All attention pattern tests written
- [ ] All gradient tests written
- [ ] All equivalence tests written
- [ ] Document reviewed against Transformer.md for consistency
