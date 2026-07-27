# Base-token BPE control (K=2/3/4) — is first-piece dominance in the embedding table itself?

Every base token has a UNIQUE ordered K-piece BPE decomposition `T = [p1..pK]` whose pieces are real vocab tokens with real embeddings — the same ordered-composition structure as a K-merge hyper-token, but the base embedding is a **free lookup**, not built by an encoder. Same cosine probes on the **raw embedding tables** (from v0.6.4; base tables ~frozen). Mergeable base tokens per K: K2: 29612, K3: 27465, K4: 22733. Minimal pairs use **all** disjoint same-(K−1)-pieces vocab pairs (deterministic).

## ① Is a base token more like its FIRST piece or a later piece?

Mean `cos(emb[T], emb[p_i])` per position. Both **raw** and **demean** (ruler removed from `emb[T]` only, as in the hyper probe) so this lines up with the hyper report — pick whichever you compare against. The base ruler is weak, so raw ≈ demean.

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

## ② Base-table anisotropy (the 'ruler' analog, context only)

| table | K2 | K3 | K4 |
|---|--:|--:|--:|
| tok_emb (input space) | 0.0897 | 0.0926 | 0.0988 |
| lm_head (output space) | 0.2297 | 0.2309 | 0.2323 |

## ④ Minimal pairs: change FIRST piece vs LAST piece (cosine distance)

change-first = pairs sharing pieces p2..pK, differing in p1. change-last = sharing p1..p(K−1), differing in pK. `first/last > 1` = the first piece matters more (prefix-dominated).

**tok_emb (input space)**

| K | variant | change-first | change-last | first/last |
|---|---|--:|--:|--:|
| 2 | raw | 0.9147 | 0.9251 | **0.99×** |
| 2 | demean | 0.9214 | 0.9318 | **0.99×** |
| 3 | raw | 0.8684 | 0.8986 | **0.97×** |
| 3 | demean | 0.8749 | 0.9051 | **0.97×** |
| 4 | raw | 0.8283 | 0.8817 | **0.94×** |
| 4 | demean | 0.8352 | 0.8887 | **0.94×** |

**lm_head (output space)**

| K | variant | change-first | change-last | first/last |
|---|---|--:|--:|--:|
| 2 | raw | 0.8501 | 0.8049 | **1.06×** |
| 2 | demean | 0.8962 | 0.848 | **1.06×** |
| 3 | raw | 0.7613 | 0.7541 | **1.01×** |
| 3 | demean | 0.8019 | 0.7932 | **1.01×** |
| 4 | raw | 0.661 | 0.7203 | **0.92×** |
| 4 | demean | 0.6955 | 0.7569 | **0.92×** |

## Side-by-side with hyper-tokens — the money comparison

Does the hyper-encoder impose more first-piece dominance than the raw embedding table shows? The minimal-pair ratio is ruler-robust, so it compares cleanly across all columns.

**④ minpair first/last ratio (prefix-dominance), per K**

| K | base `lm_head` | v0.6.4 out | vx0.6.4.2 out | base `tok_emb` | v0.6.4 in | vx0.6.4.2 in |
|---|---|---|---|---|---|---|
| 2 | 1.06× | 6.42× | 1.94× | 0.99× | 0.54× | 0.75× |
| 3 | 1.01× | 16.58× | 3.84× | 0.97× | 0.8× | 0.4× |
| 4 | 0.92× | 24.87× | 3.81× | 0.94× | 0.98× | 0.31× |

**① first-piece cosine (pos1), per K** — raw. (Note: a strong shared ruler makes the *raw* hyper cosine ≈ its ruler value, not 0; read alongside ④.)

| K | base `lm_head` | v0.6.4 out | vx0.6.4.2 out | base `tok_emb` | v0.6.4 in | vx0.6.4.2 in |
|---|---|---|---|---|---|---|
| 2 | 0.2469 | 0.8982 | 0.0125 | 0.0931 | 0.4839 | 0.0791 |
| 3 | 0.1187 | 0.872 | 0.0101 | 0.0392 | 0.4945 | 0.0404 |
| 4 | 0.0664 | 0.8636 | 0.0088 | 0.0187 | 0.4951 | 0.0359 |

---

*Repro: `uv run python probe_base.py`. Data: `results/base.json`. Method: see `docs/base_probe.md`.*