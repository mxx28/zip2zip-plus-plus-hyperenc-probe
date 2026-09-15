# Base-token BPE control (K=2/3/4) — is first-piece dominance in the embedding table itself?

Every base token has a UNIQUE ordered K-piece BPE decomposition `T = [p1..pK]` whose pieces are real vocab tokens with real embeddings — the same ordered-composition structure as a K-merge hyper-token, but the base embedding is a **free lookup**, not built by an encoder. Same cosine probes on the **raw embedding tables** (from v0.6.4; base tables ~frozen). Mergeable base tokens per K: K2: 127156, K3: 122869, K4: 107500. Minimal pairs use **all** disjoint same-(K−1)-pieces vocab pairs (deterministic).

## ① Is a base token more like its FIRST piece or a later piece?

Mean `cos(emb[T], emb[p_i])` per position. Both **raw** and **demean** (ruler removed from `emb[T]` only, as in the hyper probe) so this lines up with the hyper report — pick whichever you compare against. The base ruler is weak, so raw ≈ demean.

**lm_head (output space)** — raw per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.187 | +0.183 | — | — |
| 3 | +0.039 | +0.047 | +0.066 | — |
| 4 | -0.032 | +0.017 | +0.014 | +0.024 |

**lm_head (output space)** — demean per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.168 | +0.130 | — | — |
| 3 | +0.069 | +0.030 | +0.047 | — |
| 4 | +0.033 | +0.019 | +0.011 | +0.024 |

**tok_emb (input space)** — raw per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.187 | +0.183 | — | — |
| 3 | +0.039 | +0.047 | +0.066 | — |
| 4 | -0.032 | +0.017 | +0.014 | +0.024 |

**tok_emb (input space)** — demean per-piece cosine

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.168 | +0.130 | — | — |
| 3 | +0.069 | +0.030 | +0.047 | — |
| 4 | +0.033 | +0.019 | +0.011 | +0.024 |

## ② Base-table anisotropy (the 'ruler' analog, context only)

| table | K2 | K3 | K4 |
|---|--:|--:|--:|
| lm_head (output space) | 0.3878 | 0.3908 | 0.3958 |
| tok_emb (input space) | 0.3878 | 0.3908 | 0.3958 |

## ④ Substitution probe: replace first vs last piece (cosine similarity)

change-first = pairs sharing pieces p2..pK, differing in p1. change-last = sharing p1..p(K−1), differing in pK. Cells are `cos_first/cos_last` similarities. The ratio is cos_last / cos_first: > 1 means prefix-aligned, < 1 means suffix-aligned, and approximately 1 means balanced. Same layout as the hyper-token ④ tables.

**④a raw**

| table | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.26/0.26 | 0.36/0.31 | 0.44/0.34 | **0.99×** | **0.86×** | **0.77×** |
| tok_emb (input space) | 0.26/0.26 | 0.36/0.31 | 0.44/0.34 | **0.99×** | **0.86×** | **0.77×** |

**④b after ruler removal**

| table | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.14/0.13 | 0.25/0.19 | 0.34/0.23 | **0.98×** | **0.78×** | **0.68×** |
| tok_emb (input space) | 0.14/0.13 | 0.25/0.19 | 0.34/0.23 | **0.98×** | **0.78×** | **0.68×** |

## Side-by-side with hyper-tokens — the money comparison

Does the hyper-encoder impose more first-piece dominance than the raw embedding table shows? Paper comparisons use the raw cosine similarities only; ruler-removed diagnostics stay available above but are not used in the figure. **Bold = base control** (raw table, no encoder) — the reference each encoder column is read against.

**④ substitution cos_last/cos_first ratio, per K**

| K | **base `lm_head`** | untied HE out | tied HE out | **base `tok_emb`** | untied HE in | tied HE in |
|:--|--:|--:|--:|--:|--:|--:|
| 2 | **0.99×** | 4.49× | 2.62× | **0.99×** | 1.33× | 2.62× |
| 3 | **0.86×** | 4.1× | 2.39× | **0.86×** | 1.24× | 2.39× |
| 4 | **0.77×** | 3.69× | 2.18× | **0.77×** | 1.24× | 2.18× |

**① first-piece cosine (pos1), per K** — raw. (Note: a strong shared ruler makes the *raw* hyper cosine ≈ its ruler value, not 0; read alongside ④.)

| K | **base `lm_head`** | untied HE out | tied HE out | **base `tok_emb`** | untied HE in | tied HE in |
|:--|--:|--:|--:|--:|--:|--:|
| 2 | **0.1872** | 0.8183 | 0.744 | **0.1872** | 0.6361 | 0.744 |
| 3 | **0.039** | 0.7792 | 0.6867 | **0.039** | 0.5489 | 0.6867 |
| 4 | **-0.0322** | 0.7747 | 0.6681 | **-0.0322** | 0.5398 | 0.6681 |

---

*Repro: `uv run python probe_base.py base_llama3B`. Data: `results/base_llama3B.json`. Method: see `docs/base_probe.md`.*