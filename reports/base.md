# Base-token BPE control (K=2) — is first-piece dominance in the embedding table itself?

A BPE base token `T = A + B` (unique immediate merge). `A`/`B` are real vocab tokens with real embeddings — same 2-piece composition as a K=2 hyper-token, but the base embedding is a **free lookup**, not built by an encoder. Same cosine probes on the **raw embedding tables** (from v0.6.4; base tables ~frozen). 29612 mergeable base tokens. K=2 only (the binary split is unique; K=3/4 deferred).

## ① Is a base token more like its FIRST piece (A) or LAST piece (B)?

Mean cosine of `emb[T]` with each piece. raw and mean-removed.

| table | cos(T, first A) raw | cos(T, last B) raw | first−last | (demean) first | last |
|---|--:|--:|--:|--:|--:|
| tok_emb (input space) | 0.0931 | 0.1191 | -0.026 | 0.0915 | 0.1175 |
| lm_head (output space) | 0.2469 | 0.1191 | +0.128 | 0.2263 | 0.0864 |

## ② Base-table anisotropy (the 'ruler' analog)

| table | cos-with-mean of T |
|---|--:|
| tok_emb (input space) | 0.0897 |
| lm_head (output space) | 0.2297 |

## ④ Minimal pairs: change FIRST piece vs LAST piece (cosine distance)

differ-first = tokens sharing the last piece B, different A (real vocab pairs). differ-last = sharing A, different B. first/last > 1 = first piece matters more.

| table | variant | differ-first | differ-last | first/last |
|---|---|--:|--:|--:|
| tok_emb (input space) | raw | 0.914 | 0.9259 | **0.99×** |
| tok_emb (input space) | demean | 0.9208 | 0.9326 | **0.99×** |
| lm_head (output space) | raw | 0.8499 | 0.8086 | **1.05×** |
| lm_head (output space) | demean | 0.8957 | 0.8519 | **1.05×** |

## Side-by-side with v0.6.4 **hyper-tokens** (K=2)

Does the hyper-encoder show more first-piece dominance than the raw base table?

| quantity | base `lm_head` | v0.6.4 hyper **output** | base `tok_emb` | v0.6.4 hyper **input** |
|---|--:|--:|--:|--:|
| ① cos(·, first) raw | 0.2469 | 0.8982 | 0.0931 | 0.4839 |
| ① cos(·, last) raw | 0.1191 | 0.1496 | 0.1191 | 0.1549 |
| ④ minpair first/last (raw) | 1.05× | 6.42× | 0.99× | 0.54× |
| ② ruler / anisotropy | 0.2297 | 0.3558 | 0.0897 | 0.4667 |

---

*Repro: `uv run python probe_base.py`. Data: `results/base.json`. Method: see `README.md`.*