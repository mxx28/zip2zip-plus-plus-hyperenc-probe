# Base-token pieces through the vx0.6.4.2 (no-residual) hyper-encoder

Same objects as the base-token control (`probe_base.py`) — every base token `T`'s unique BPE decomposition `[p1..pK]` and the same real shared-(K−1)-pieces minimal pairs — but the vector is **`E = hyper_encoder([p1..pK])`** (residual **off**), not the raw lookup `emb[T]`. So the only change vs `base.md` is raw-lookup → encoder. Bit-exact encoder (max|Δ|=0 vs `model.py`). OOD caveat: BPE pieces are sub-word, not whole-token merges.

## ① Per-piece cosine `cos(E, emb[p_i])`

**lm_head (output space)** — raw

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.014 | +0.008 | — | — |
| 3 | +0.012 | +0.008 | +0.005 | — |
| 4 | +0.011 | +0.007 | +0.006 | +0.004 |

**lm_head (output space)** — demean

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.307 | +0.111 | — | — |
| 3 | +0.311 | +0.107 | +0.065 | — |
| 4 | +0.302 | +0.089 | +0.066 | +0.061 |

**tok_emb (input space)** — raw

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.092 | +0.109 | — | — |
| 3 | +0.067 | +0.082 | +0.109 | — |
| 4 | +0.055 | +0.076 | +0.090 | +0.114 |

**tok_emb (input space)** — demean

| K | pos1 (first) | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| 2 | +0.117 | +0.121 | — | — |
| 3 | +0.093 | +0.080 | +0.130 | — |
| 4 | +0.078 | +0.056 | +0.078 | +0.133 |

## ② Ruler (cos-with-mean of the encoded set)

| table | K2 | K3 | K4 |
|---|--:|--:|--:|
| lm_head (output space) | 0.9974 | 0.9982 | 0.9979 |
| tok_emb (input space) | 0.721 | 0.8584 | 0.8922 |

## ④ Minimal pairs: change FIRST vs LAST piece (cosine distance)

Same pairs as `base.md`, but distance is between encoder vectors. `first/last > 1` = prefix-dominated.

**④a raw**

| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | **7.64×** | **4.01×** | **1.02×** |
| tok_emb (input space) | 0.23/0.20 | 0.07/0.11 | 0.04/0.08 | **1.13×** | **0.67×** | **0.51×** |

**④b after ruler removal**

| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.69/0.22 | 0.61/0.09 | 0.57/0.10 | **3.13×** | **6.55×** | **5.42×** |
| tok_emb (input space) | 0.50/0.43 | 0.29/0.41 | 0.22/0.40 | **1.15×** | **0.7×** | **0.54×** |

---

*Repro: `uv run python probe_base_hyper.py`. Data: `results/base_hyper_vx0642.json`.*