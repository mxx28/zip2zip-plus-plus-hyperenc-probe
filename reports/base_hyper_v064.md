# Base-token pieces through the v0.6.4 (residual) hyper-encoder

Same objects as the base-token control (`probe_base.py`) — every base token `T`'s unique BPE decomposition `[p1..pK]` and the same real shared-(K−1)-pieces minimal pairs — but the vector is **`E = hyper_encoder([p1..pK])`** (residual **on**), not the raw lookup `emb[T]`. So the only change vs `base.md` is raw-lookup → encoder. Bit-exact encoder (max|Δ|=0 vs `model.py`). OOD caveat: BPE pieces are sub-word, not whole-token merges.

## ① Per-piece cosine `cos(E, emb[p_i])`

**lm_head (output space)** — raw

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.924 | +0.103 | — | — |
| 3 | +0.894 | +0.144 | +0.139 | — |
| 4 | +0.883 | +0.172 | +0.172 | +0.166 |

**lm_head (output space)** — demean

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.906 | +0.043 | — | — |
| 3 | +0.846 | +0.040 | +0.039 | — |
| 4 | +0.795 | +0.035 | +0.034 | +0.034 |

**tok_emb (input space)** — raw

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.523 | +0.145 | — | — |
| 3 | +0.499 | +0.120 | +0.158 | — |
| 4 | +0.465 | +0.106 | +0.125 | +0.164 |

**tok_emb (input space)** — demean

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.618 | +0.150 | — | — |
| 3 | +0.595 | +0.101 | +0.147 | — |
| 4 | +0.561 | +0.070 | +0.088 | +0.142 |

## ② Ruler (cos-with-mean of the encoded set)

| table | K2 | K3 | K4 |
|---|--:|--:|--:|
| lm_head (output space) | 0.3537 | 0.4658 | 0.5609 |
| tok_emb (input space) | 0.5453 | 0.5975 | 0.6587 |

## ④ Substitution probe: replace first vs last piece (cosine distance)

Same pairs as `base.md`, but distance is between encoder vectors. `first/last > 1` = prefix-dominated.

**④a raw**

| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.78/0.05 | 0.67/0.02 | 0.60/0.02 | **15.87×** | **29.97×** | **31.44×** |
| tok_emb (input space) | 0.29/0.38 | 0.24/0.25 | 0.21/0.20 | **0.76×** | **0.97×** | **1.01×** |

**④b after ruler removal**

| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.89/0.06 | 0.86/0.03 | 0.84/0.03 | **15.4×** | **28.27×** | **24.12×** |
| tok_emb (input space) | 0.42/0.53 | 0.39/0.39 | 0.38/0.38 | **0.78×** | **0.99×** | **1.0×** |

---

*Repro: `uv run python probe_base_hyper.py`. Data: `results/base_hyper_v064.json`.*