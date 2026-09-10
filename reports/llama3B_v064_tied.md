# Hyper-encoder embedding probe — Llama-3.2-3B v0.6.4 (tied HE, residual)

Single-checkpoint report (`llama3B_v064_tied`). Analysis runs on the **final vector the model uses** (residual **on**: `E = base_vec[t1] + encoder_out`). Bit-exact encoder re-implementation (max|Δ|=0 vs `model.py`). A shared *ruler* vector is not guaranteed (see ②), so ④ and ⑤ are shown **both** on the raw vector and after ruler removal. Method: see `docs/hyper_probe.md`.

## Data & objects

**Corpus:** Salesforce/wikitext · wikitext-2-raw-v1 [train] — 2,435,022 base tokens. **Hyper-tokens:** consecutive base-token K-grams (K=2/3/4), dropping any window with a special/digit token, deduplicated, keeping the **top-5000 by corpus frequency** per K (frequency only bounds the sample; the encoder is a fixed function of the K ids, so any valid K-tuple probes the same geometry). **Base-token pool** (④ replacement + ② floor): 39,087 unique tokens. Seed 0, ④ averaged over 6 resamples.

| K | unique K-grams | analyzed (top-N) | freq range kept |
|---|--:|--:|--:|
| 2 | 669,574 | 5,000 | 37..17120 |
| 3 | 1,439,016 | 5,000 | 18..16906 |
| 4 | 1,804,397 | 5,000 | 8..2660 |

## Reading

- **output**: **prefix-sensitive (reads the head)** (cos_last/cos_first = 3.56× at K=4). Shared-ruler component is modest (cos-with-mean 0.4504), above its matched init baseline (0.323) — a mild, **partly training-induced** shared direction, but far from a full collapse.
- **input**: **prefix-sensitive (reads the head)** (cos_last/cos_first = 3.56× at K=4). Shared-ruler component is modest (cos-with-mean 0.4504), above its matched init baseline (0.323) — a mild, **partly training-induced** shared direction, but far from a full collapse.

## ① Raw per-position cosine (no ruler removed)

| role · K | pos1 | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| output · K2 | +0.744 | +0.129 | — | — |
| output · K3 | +0.687 | +0.106 | +0.102 | — |
| output · K4 | +0.668 | +0.085 | +0.101 | +0.104 |
| input · K2 | +0.744 | +0.129 | — | — |
| input · K3 | +0.687 | +0.106 | +0.102 | — |
| input · K4 | +0.668 | +0.085 | +0.101 | +0.104 |

## ② Shared ruler (does one common vector dominate every hyper-token?)

cos-with-mean → 1 means all hyper-tokens are nearly the same vector. To ask whether training *created* a shared direction, compare against the model's **matched step-0 init** — the value before any training: for this residual run that is **E0 = e(t1)** (`init (matched)` column). The `rand-enc` column is always the full-scale random encoder (γ=1); for residual runs it is **not** the matched init and must not be used as the reference. `tok floor` is the ordinary-token anisotropy floor in the same space.

| role | cos w/ mean | min | pairwise | energy | tok floor | init (matched) | rand-enc |
|---|--:|--:|--:|--:|--:|--:|--:|
| output | 0.4504 | 0.1762 | 0.2115 | 0.1983 | 0.2865 | 0.323 | 0.762 |
| input | 0.4504 | 0.1762 | 0.2052 | 0.1983 | 0.2865 | 0.323 | 0.748 |

## ③ Per-position cosine after removing the ruler

The discriminative part of each hyper-token vs each base token.

| role · K | pos1 | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| output · K2 | +0.761 | +0.110 | — | — |
| output · K3 | +0.739 | +0.091 | +0.091 | — |
| output · K4 | +0.741 | +0.075 | +0.095 | +0.100 |
| input · K2 | +0.761 | +0.110 | — | — |
| input · K3 | +0.739 | +0.091 | +0.091 | — |
| input · K4 | +0.741 | +0.075 | +0.095 | +0.100 |

## ④ Substitution probe (replace first vs last token)

Change one base token (to a random pool token) and measure the cosine similarity between the original and perturbed embeddings. The ratio is cos_last / cos_first: > 1 means prefix-aligned, < 1 means suffix-aligned, and approximately 1 means balanced. Two variants are retained in the diagnostic report; paper figures use the raw vector only.

**④a raw vector (no ruler removed)**

| role | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.29/0.76 | 0.36/0.86 | 0.41/0.89 | **2.62×** | **2.39×** | **2.18×** |
| input | 0.29/0.76 | 0.36/0.86 | 0.41/0.89 | **2.62×** | **2.39×** | **2.18×** |

**④b after ruler removal**

| role | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.21/0.71 | 0.21/0.81 | 0.24/0.85 | **3.45×** | **3.81×** | **3.56×** |
| input | 0.21/0.71 | 0.21/0.81 | 0.24/0.85 | **3.45×** | **3.81×** | **3.56×** |

## ⑤ Growth probe: growing-prefix examples

Illustrative real chains H2⊂H3⊂H4 (a frequent 4-gram and its growing prefixes). High cos(H2,H4) = growing prefixes stay alike (encoder keys on the shared head). Shown raw and after ruler removal. **These are examples for the reader, not aggregate statistics.**

**Example 1: `It is a dog`**  — [It,  is] ⊂ [It,  is,  a] ⊂ [It,  is,  a,  dog]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.876 | 0.787 |
| output | demean | 0.861 | 0.74 |
| input | raw | 0.876 | 0.787 |
| input | demean | 0.861 | 0.74 |

**Example 2: `the end of the`**  — [the,  end] ⊂ [the,  end,  of] ⊂ [the,  end,  of,  the]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.869 | 0.801 |
| output | demean | 0.821 | 0.717 |
| input | raw | 0.869 | 0.801 |
| input | demean | 0.821 | 0.717 |

**Example 3: `in the United States`**  — [in,  the] ⊂ [in,  the,  United] ⊂ [in,  the,  United,  States]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.778 | 0.77 |
| output | demean | 0.742 | 0.727 |
| input | raw | 0.778 | 0.77 |
| input | demean | 0.742 | 0.727 |

**Example 4: `the rest of the`**  — [the,  rest] ⊂ [the,  rest,  of] ⊂ [the,  rest,  of,  the]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.846 | 0.785 |
| output | demean | 0.792 | 0.691 |
| input | raw | 0.846 | 0.785 |
| input | demean | 0.792 | 0.691 |

**Example 5: `for the first time`**  — [for,  the] ⊂ [for,  the,  first] ⊂ [for,  the,  first,  time]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.79 | 0.749 |
| output | demean | 0.748 | 0.676 |
| input | raw | 0.79 | 0.749 |
| input | demean | 0.748 | 0.676 |

---

*Repro: `uv run python probe.py llama3B_v064_tied`. Data: `results/llama3B_v064_tied.json`.*