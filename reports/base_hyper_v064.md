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

## ④ Substitution probe: replace first vs last piece (cosine similarity)

Same pairs as `base.md`, but cosine similarity is measured between encoder vectors. `cos_last/cos_first > 1` = prefix-dominated.

**④a raw**

| table | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.23/0.95 | 0.34/0.98 | 0.41/0.98 | **4.18×** | **2.86×** | **2.39×** |
| tok_emb (input space) | 0.72/0.63 | 0.76/0.75 | 0.80/0.80 | **0.88×** | **0.99×** | **1.0×** |

**④b after ruler removal**

| table | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| lm_head (output space) | 0.12/0.94 | 0.15/0.97 | 0.17/0.97 | **8.1×** | **6.33×** | **5.67×** |
| tok_emb (input space) | 0.59/0.47 | 0.62/0.61 | 0.62/0.62 | **0.81×** | **0.99×** | **1.0×** |

---

*Repro: `uv run python probe_base_hyper.py`. Data: `results/base_hyper_v064.json`.*