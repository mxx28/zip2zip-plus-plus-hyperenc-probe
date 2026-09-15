# Hyper-encoder embedding probe — Llama-3.2-3B v0.6.4 (untied HE, residual)

Single-checkpoint report (`llama3B_v064_untied`). Analysis runs on the **final vector the model uses** (residual **on**: `E = base_vec[t1] + encoder_out`). Bit-exact encoder re-implementation (max|Δ|=0 vs `model.py`). A shared *ruler* vector is not guaranteed (see ②), so ④ and ⑤ are shown **both** on the raw vector and after ruler removal. Method: see `docs/hyper_probe.md`.

## Data & objects

**Corpus:** Salesforce/wikitext · wikitext-2-raw-v1 [train] — 2,435,022 base tokens. **Hyper-tokens:** consecutive base-token K-grams (K=2/3/4), dropping any window with a special/digit token, deduplicated, keeping the **top-5000 by corpus frequency** per K (frequency only bounds the sample; the encoder is a fixed function of the K ids, so any valid K-tuple probes the same geometry). **Base-token pool** (④ replacement + ② floor): 39,087 unique tokens. Seed 0, ④ averaged over 6 resamples.

| K | unique K-grams | analyzed (top-N) | freq range kept |
|---|--:|--:|--:|
| 2 | 669,574 | 5,000 | 37..17120 |
| 3 | 1,439,016 | 5,000 | 18..16906 |
| 4 | 1,804,397 | 5,000 | 8..2660 |

## Reading

- **output**: **prefix-sensitive (reads the head)** (cos_last/cos_first = 6.91× at K=4). Shared-ruler component is at its matched init baseline (cos-with-mean 0.3923 vs init 0.323), i.e. **inherited** from the base-embedding geometry, not created by training.
- **input**: roughly balanced across positions (cos_last/cos_first = 1.35× at K=4). Shared-ruler component is modest (cos-with-mean 0.5008), above its matched init baseline (0.323) — a mild, **partly training-induced** shared direction, but far from a full collapse.

## ① Raw per-position cosine (no ruler removed)

| role · K | pos1 | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| output · K2 | +0.818 | +0.106 | — | — |
| output · K3 | +0.779 | +0.099 | +0.086 | — |
| output · K4 | +0.775 | +0.084 | +0.099 | +0.094 |
| input · K2 | +0.636 | +0.161 | — | — |
| input · K3 | +0.549 | +0.130 | +0.164 | — |
| input · K4 | +0.540 | +0.103 | +0.127 | +0.163 |

## ② Shared ruler (does one common vector dominate every hyper-token?)

cos-with-mean → 1 means all hyper-tokens are nearly the same vector. To ask whether training *created* a shared direction, compare against the model's **matched step-0 init** — the value before any training: for this residual run that is **E0 = e(t1)** (`init (matched)` column). The `rand-enc` column is always the full-scale random encoder (γ=1); for residual runs it is **not** the matched init and must not be used as the reference. `tok floor` is the ordinary-token anisotropy floor in the same space.

| role | cos w/ mean | min | pairwise | energy | tok floor | init (matched) | rand-enc |
|---|--:|--:|--:|--:|--:|--:|--:|
| output | 0.3923 | 0.0944 | 0.1621 | 0.1467 | 0.2865 | 0.323 | 0.762 |
| input | 0.5008 | 0.2574 | 0.2521 | 0.2515 | 0.2865 | 0.323 | 0.748 |

## ③ Per-position cosine after removing the ruler

The discriminative part of each hyper-token vs each base token.

| role · K | pos1 | pos2 | pos3 | pos4 |
|---|--:|--:|--:|--:|
| output · K2 | +0.822 | +0.079 | — | — |
| output · K3 | +0.810 | +0.074 | +0.061 | — |
| output · K4 | +0.816 | +0.065 | +0.084 | +0.076 |
| input · K2 | +0.636 | +0.118 | — | — |
| input · K3 | +0.575 | +0.097 | +0.135 | — |
| input · K4 | +0.578 | +0.078 | +0.110 | +0.145 |

## ④ Substitution probe (replace first vs last token)

Change one base token (to a random pool token) and measure the cosine similarity between the original and perturbed embeddings. The ratio is cos_last / cos_first: > 1 means prefix-aligned, < 1 means suffix-aligned, and approximately 1 means balanced. Two variants are retained in the diagnostic report; paper figures use the raw vector only.

**④a raw vector (no ruler removed)**

| role | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.18/0.83 | 0.23/0.92 | 0.26/0.94 | **4.49×** | **4.1×** | **3.69×** |
| input | 0.48/0.64 | 0.58/0.72 | 0.63/0.78 | **1.33×** | **1.24×** | **1.24×** |

**④b after ruler removal**

| role | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.13/0.80 | 0.11/0.90 | 0.13/0.93 | **6.28×** | **7.87×** | **6.91×** |
| input | 0.41/0.54 | 0.47/0.61 | 0.50/0.68 | **1.32×** | **1.3×** | **1.35×** |

## ⑤ Growth probe: growing-prefix examples

Illustrative real chains H2⊂H3⊂H4 (a frequent 4-gram and its growing prefixes). High cos(H2,H4) = growing prefixes stay alike (encoder keys on the shared head). Shown raw and after ruler removal. **These are examples for the reader, not aggregate statistics.**

**Example 1: `It is a dog`**  — [It,  is] ⊂ [It,  is,  a] ⊂ [It,  is,  a,  dog]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.923 | 0.857 |
| output | demean | 0.921 | 0.843 |
| input | raw | 0.776 | 0.568 |
| input | demean | 0.729 | 0.425 |

**Example 2: `the end of the`**  — [the,  end] ⊂ [the,  end,  of] ⊂ [the,  end,  of,  the]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.914 | 0.886 |
| output | demean | 0.885 | 0.844 |
| input | raw | 0.818 | 0.728 |
| input | demean | 0.729 | 0.581 |

**Example 3: `in the United States`**  — [in,  the] ⊂ [in,  the,  United] ⊂ [in,  the,  United,  States]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.823 | 0.835 |
| output | demean | 0.797 | 0.811 |
| input | raw | 0.642 | 0.545 |
| input | demean | 0.454 | 0.296 |

**Example 4: `the rest of the`**  — [the,  rest] ⊂ [the,  rest,  of] ⊂ [the,  rest,  of,  the]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.893 | 0.866 |
| output | demean | 0.856 | 0.816 |
| input | raw | 0.769 | 0.662 |
| input | demean | 0.652 | 0.476 |

**Example 5: `for the first time`**  — [for,  the] ⊂ [for,  the,  first] ⊂ [for,  the,  first,  time]

| role | variant | cos(H2,H3) | cos(H2,H4) |
|---|---|--:|--:|
| output | raw | 0.839 | 0.84 |
| output | demean | 0.811 | 0.811 |
| input | raw | 0.628 | 0.518 |
| input | demean | 0.479 | 0.305 |

---

*Repro: `uv run python probe.py llama3B_v064_untied`. Data: `results/llama3B_v064_untied.json`.*