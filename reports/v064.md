# Hyper-encoder embedding probe — v0.6.4 (untied, residual)

Single-checkpoint report (`v064`). Analysis runs on the **final vector the model uses** (residual **on**: `E = base_vec[t1] + encoder_out`). Bit-exact encoder re-implementation (max|Δ|=0 vs `model.py`). Hyper-tokens are real English n-grams (K=2/3/4). A shared *ruler* vector is not guaranteed (see ②), so ④ and ⑤ are shown **both** on the raw vector and after ruler removal. Method: see `README.md`.

## Reading

- **output**:
    - **prefix-sensitive** (change-first/change-last = 22.95× at K=4). 
    - Shared-ruler component is moderate (cos-with-mean 0.3558).
- **input**: 
    - roughly balanced across positions (change-first/change-last = 0.96× at K=4). But seems little suffix-sensitive at K=2/3.
    - Shared-ruler component is moderate (cos-with-mean 0.4667).

## ① Raw per-position cosine (no ruler removed)

| role · K | pos1 | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| output · K2 | +0.898 | +0.150 | — | — |
| output · K3 | +0.872 | +0.124 | +0.129 | — |
| output · K4 | +0.864 | +0.108 | +0.119 | +0.138 |
| input · K2 | +0.484 | +0.155 | — | — |
| input · K3 | +0.494 | +0.084 | +0.130 | — |
| input · K4 | +0.495 | +0.053 | +0.075 | +0.121 |

**Observation**: The hyper-token vectors show unusually high cosine similarity with the first constituent token, particularly on the output side, where the similarity remains above 0.86 across all values of K. This strong first-token alignment may be partly induced by the residual initialization rather than being entirely learned by the hyper-encoder. We therefore additionally analyze **v0.6.4.2** as a control to examine how the representation changes when this residual effect is removed.

## ② Shared ruler (does one common vector dominate every hyper-token?)

cos-with-mean → 1 means all hyper-tokens are nearly the same vector. Controls: ordinary-token floor (same space) and a random-init encoder.

| role | cos w/ mean | min | pairwise | energy | tok floor | rand-init |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.3558 | 0.1297 | 0.1253 | 0.1173 | 0.2688 | 0.626 |
| input | 0.4667 | 0.2136 | 0.2142 | 0.2155 | 0.1253 | 0.576 |

**Observation**: no meaningful shared ruler. output 0.356 / input 0.467 are slightly above their tok floors (0.269 / 0.125) but **below** rand-init (0.626 / 0.576) — the trained vectors are *more* spread out than a random encoder's. In v0.6.4 the residual anchors each vector to its own first base token, which actually *prevents* collapse onto a single direction; this is also why removing the ruler barely changes ④/⑤ (24.87× vs 22.95×). (Contrast: the no-residual vx0.6.4.2 output collapses to 0.9988 — a strong ruler.)

## ③ Per-position cosine after removing the ruler

The discriminative part of each hyper-token vs each base token.

| role · K | pos1 | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| output · K2 | +0.848 | +0.066 | — | — |
| output · K3 | +0.841 | +0.043 | +0.044 | — |
| output · K4 | +0.843 | +0.027 | +0.036 | +0.055 |
| input · K2 | +0.516 | +0.156 | — | — |
| input · K3 | +0.555 | +0.084 | +0.137 | — |
| input · K4 | +0.567 | +0.050 | +0.075 | +0.131 |

## ④ Causal minimal pairs (change first vs last token)

Change one base token, measure how much the embedding moves (cosine distance). first/last > 1 = prefix-dominated (reads the head); < 1 = tail-weighted. Two variants — on the raw vector, and after removing the shared ruler.

**④a raw vector (no ruler removed)**

| role | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.82/0.13 | 0.78/0.05 | 0.76/0.03 | **6.42×** | **16.58×** | **24.87×** |
| input | 0.29/0.53 | 0.29/0.36 | 0.28/0.29 | **0.54×** | **0.80×** | **0.98×** |

**④b after ruler removal**

| role | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.89/0.15 | 0.88/0.06 | 0.87/0.04 | **5.92×** | **15.49×** | **22.95×** |
| input | 0.34/0.64 | 0.37/0.48 | 0.38/0.40 | **0.54×** | **0.78×** | **0.96×** |

## ⑤ Nested "It is a dog"

H2=[It,is] → H3=[It,is,a] → H4=[It,is,a,dog]. High cos(H2,H4) = growing prefixes stay alike. Two variants — raw vector, and after ruler removal.

**⑤a raw vector (no ruler removed)**

| role | cos(H2,H3) | cos(H2,H4) |
|---|--:|--:|
| output | 0.955 | 0.901 |
| input | 0.938 | 0.481 |

**⑤b after ruler removal**

| role | cos(H2,H3) | cos(H2,H4) |
|---|--:|--:|
| output | 0.947 | 0.884 |
| input | 0.928 | 0.382 |

---

*Repro: `uv run python probe.py v064`. Data: `results/v064.json`.*