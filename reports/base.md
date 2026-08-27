# Base-token BPE control (K=2/3/4) — is first-piece dominance in the embedding table itself?

Every base token has a UNIQUE ordered K-piece BPE decomposition `T = [p1..pK]` whose pieces are real vocab tokens with real embeddings — the same ordered-composition structure as a K-merge hyper-token, but the base embedding is a **free lookup**, not built by an encoder. Same cosine probes on the **raw embedding tables** (from v0.6.4; base tables ~frozen). Mergeable base tokens per K: K2: 29612, K3: 27465, K4: 22733. Minimal pairs use **all** disjoint same-(K−1)-pieces vocab pairs (deterministic).

## ① Is a base token more like its FIRST piece or a later piece?

Mean `cos(emb[T], emb[p_i])` per position. Both **raw** and **demean** (ruler removed from `emb[T]` only, as in the hyper probe) so this lines up with the hyper report — pick whichever you compare against. The base ruler is weak, so raw ≈ demean.

**lm_head (output space)** — raw per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.247 | +0.119 | — | — |
| 3 | +0.119 | +0.050 | +0.054 | — |
| 4 | +0.066 | +0.038 | +0.025 | +0.031 |

**lm_head (output space)** — demean per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.225 | +0.085 | — | — |
| 3 | +0.108 | +0.026 | +0.029 | — |
| 4 | +0.063 | +0.019 | +0.006 | +0.012 |

**tok_emb (input space)** — raw per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.093 | +0.119 | — | — |
| 3 | +0.039 | +0.024 | +0.060 | — |
| 4 | +0.019 | +0.011 | +0.009 | +0.031 |

**tok_emb (input space)** — demean per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.090 | +0.117 | — | — |
| 3 | +0.039 | +0.024 | +0.061 | — |
| 4 | +0.021 | +0.015 | +0.013 | +0.034 |

### Growth probe: growing-prefix examples (raw-lookup analog of hyper ⑤)

Real vocab tokens that are growing string-prefixes `a ⊂ b ⊂ c` (mapped to H2⊂H3⊂H4), using their **raw lookup embeddings** — no encoder. High cos = growing-prefix words stay alike in the table itself. Compare against the hyper ⑤ (where the *encoder* is what keeps growing merges alike). **Illustrative examples, not statistics.**

**lm_head (output space)**

| chain (H2 ⊂ H3 ⊂ H4) | cos(H2,H3) | cos(H2,H4) |
|---|--:|--:|
| under ⊂ understand ⊂ understanding | 0.143 | 0.142 |
| inter ⊂ intern ⊂ international | 0.175 | 0.19 |
| care ⊂ careful ⊂ carefully | 0.37 | 0.297 |
| success ⊂ successful ⊂ successfully | 0.489 | 0.343 |

**tok_emb (input space)**

| chain (H2 ⊂ H3 ⊂ H4) | cos(H2,H3) | cos(H2,H4) |
|---|--:|--:|
| under ⊂ understand ⊂ understanding | 0.061 | 0.034 |
| inter ⊂ intern ⊂ international | 0.131 | 0.08 |
| care ⊂ careful ⊂ carefully | 0.114 | 0.06 |
| success ⊂ successful ⊂ successfully | 0.243 | 0.137 |

## ② Base-table anisotropy (the 'ruler' analog, context only)

| table | K2 | K3 | K4 |
|---|--:|--:|--:|
| lm_head (output space) | 0.2297 | 0.2309 | 0.2323 |
| tok_emb (input space) | 0.0897 | 0.0926 | 0.0988 |

## ④ Substitution probe: replace first vs last piece (cosine distance)

change-first = pairs sharing pieces p2..pK, differing in p1. change-last = sharing p1..p(K−1), differing in pK. Cells are `change-first/change-last` distances; `first/last > 1` = the first piece matters more (prefix-dominated). Same layout as the hyper-token ④ tables.

**④a raw**

| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.85/0.80 | 0.76/0.76 | 0.66/0.72 | **1.06×** | **1.01×** | **0.92×** |
| tok_emb (input space) | 0.92/0.92 | 0.87/0.90 | 0.83/0.88 | **0.99×** | **0.96×** | **0.94×** |

**④b after ruler removal**

| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.90/0.85 | 0.80/0.79 | 0.70/0.76 | **1.06×** | **1.01×** | **0.92×** |
| tok_emb (input space) | 0.92/0.93 | 0.87/0.91 | 0.84/0.89 | **0.99×** | **0.97×** | **0.94×** |

## Side-by-side with hyper-tokens — the money comparison

Does the hyper-encoder impose more first-piece dominance than the raw embedding table shows? The minimal-pair ratio is ruler-robust, so it compares cleanly across all columns. **Bold = base control** (raw table, no encoder) — the reference each encoder column is read against.

**④ substitution first/last ratio (prefix-dominance), per K**

| K | **base `lm_head`** | v0.6.4 out | vx0.6.4.2 out | **base `tok_emb`** | v0.6.4 in | vx0.6.4.2 in |
|:--|--:|--:|--:|--:|--:|--:|
| 2 | **1.06×** | 6.2× | 2.01× | **0.99×** | 0.55× | 0.72× |
| 3 | **1.01×** | 14.79× | 3.92× | **0.96×** | 0.82× | 0.46× |
| 4 | **0.92×** | 20.93× | 4.85× | **0.94×** | 0.99× | 0.37× |

**① first-piece cosine (pos1), per K** — raw. (Note: a strong shared ruler makes the *raw* hyper cosine ≈ its ruler value, not 0; read alongside ④.)

| K | **base `lm_head`** | v0.6.4 out | vx0.6.4.2 out | **base `tok_emb`** | v0.6.4 in | vx0.6.4.2 in |
|:--|--:|--:|--:|--:|--:|--:|
| 2 | **0.2469** | 0.8917 | 0.012 | **0.0931** | 0.4916 | 0.0953 |
| 3 | **0.1187** | 0.865 | 0.0109 | **0.0392** | 0.4971 | 0.0498 |
| 4 | **0.0664** | 0.8586 | 0.0084 | **0.0187** | 0.5061 | 0.0423 |

---

*Repro: `uv run python probe_base.py`. Data: `results/base.json`. Method: see `docs/base_probe.md`.*