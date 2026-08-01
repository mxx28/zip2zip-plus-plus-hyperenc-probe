# Hyper-token embedding probe (`probe.py`)

**What it asks:** what does a zip2zip hyper-token's embedding actually represent,
and do the *input* (embedding) and *output* (unembedding) hyper-encoders learn
different functions? It runs directly on a trained checkpoint's weights (no
training, no GPU), one checkpoint per run — compare reports across checkpoints
yourself.

## What a hyper-token embedding is

A hyper-token `H` is an LZW merge of `K` base tokens `[t1, …, tK]` (K = 2–4).
The model turns it into one vector with a small **flat hyper-encoder** (2-layer
bidirectional transformer, mean-pooled), reading base-token vectors from an
embedding matrix. There are two encoders with two roles:

| role | reads from | used as | question |
|---|---|---|---|
| **input**  | `tok_embeddings` | the embedding injected into the decoder stream in place of H | what does the model *treat H as*? |
| **output** | `lm_head` (`output.weight`) | the unembedding row `E_out(H)` appended for H (used as `logit(H) = hidden · E_out(H)`) | how does the model *score emitting H*? |

The **final vector** the model uses is
`E(H) = base_vec[t1] + encoder_out(H)` when the residual path is on, or
`E(H) = encoder_out(H)` when it is off (`no_encoder_residual`).
We always analyze that final vector — the one the model really uses.


## Where the test hyper-tokens come from

The encoder is a **pure function of the K base-token ids** it is fed, so we can
run it on any K-tuple and get exactly the vector the model would produce for
that merge. The objects analyzed below come from three sources — keep them
straight, because the provenance is what each step's claim rests on. **Exact
counts are printed in the "Data & objects" table at the top of each report** (the
numbers below are from the WikiText-2-raw-v1 train split).

1. **Corpus n-grams (`NAT`, used by ①②③④⑤).** The corpus is **WikiText-2-raw-v1**
   (`Salesforce/wikitext`; the run falls back to the identical-content
   `EleutherAI/wikitext_document_level` packaging when the former can't be
   fetched offline) — a public, standard LM benchmark, ~2.83 M base tokens. We
   tokenize with the Phi-3.5 tokenizer and take **every sliding window of K
   consecutive base tokens** (K = 2/3/4), dropping any window containing a
   special or digit token (`< 32064`, matching the model's `disable_digit_ids`),
   deduplicating, and keeping the **top-N by corpus frequency** per K
   (`N = 5000`). Frequency is used **only** to draw a bounded, reproducible,
   natural sample — *not* as a correctness claim about which sequences the
   codebook merges. Typical yield: ~0.6 M / 1.5 M / 2.0 M unique K-grams before
   the top-N cut.
2. **Corpus vocabulary (`POOL`, used by ② and ④).** The sorted set of distinct
   non-special / non-digit base tokens that appear in the corpus (~18 K). Used as
   the **ordinary-token anisotropy floor** in ② and as the **random-replacement
   pool** in ④.
3. **A few illustrative examples (used by ⑤).** The `N = 4` most frequent
   *readable* 4-grams (every piece contains a letter), e.g. `in the United
   States`, `for the first time` — each gives a nested chain `H2 ⊂ H3 ⊂ H4` from
   its growing prefixes. These are **worked examples for the reader, not
   aggregate statistics.**

| step | operates on | source |
|---|---|---|
| ① raw          | `NAT` top-N K-grams                        | corpus |
| ② ruler        | `NAT` (+ `POOL` token floor, random-init)  | corpus + control |
| ③ demean       | `NAT` top-N K-grams                        | corpus |
| ④ minimal pairs| `NAT`, one token swapped for a uniform `POOL` draw | corpus + random (seed 0, 6 resamples) |
| ⑤ nested       | 4 frequent readable 4-gram prefix chains   | corpus, N = 4 examples |

**Why corpus n-grams instead of the model's real codebook merges?** Because it
makes no difference to what we measure. Once training is done, the hyper-encoder
weights and the base-token embedding matrices are **fixed**, so `E(H)` is a
deterministic function of the K base-token ids alone — it does not depend on
whether `H` was ever an actual codebook entry. Any valid K-tuple therefore probes
the same fixed geometry, so a well-defined, reproducible corpus sample is exactly
as valid as the codebook itself, and avoids tying the analysis to any one run's
codebook. (The `probe_base_hyper.py` control confirms the *raw-lookup vs encoder*
contrast is robust on identical objects; note the encoder's ratio magnitude is
somewhat input-distribution dependent — read ④ ratios qualitatively.)

## The method, in five steps

Everything reduces to **cosine similarity on the final vector**; the only
variables are *what we compare it to* and *whether we first remove a shared
"ruler" vector*.

**① Raw per-position similarity.** 

For each hyper-token, cosine of its final
vector with each of its base tokens' vectors, averaged. (previously shows the output vector is ≈0 with *all* its base tokens — which motivates ②.)

**② Does a shared "ruler" exist?** 

Take the mean vector `c` over all
hyper-tokens. If every hyper-token has `cos(E, c) ≈ 1`, then each one is almost
entirely that shared `c`, which dilutes every raw cosine in ①. We report
`cos-with-mean` (and its min), the average pairwise cosine between distinct
hyper-tokens, and the fraction of squared length that is the mean — against two
controls: **ordinary tokens** in the same space (anisotropy floor) and a
**random-init encoder** (is the collapse learned?).

Example Table:
| role | cos w/ mean | min | pairwise | energy | tok floor | rand-init |
|---|--:|--:|--:|--:|--:|--:|
| output | 0.3558 | 0.1297 | 0.1253 | 0.1173 | 0.2688 | 0.626 |
| input | 0.4667 | 0.2136 | 0.2142 | 0.2155 | 0.1253 | 0.576 |

The last three columns are the two baselines that decide whether a ruler is **real and learned**:

- **energy** — fraction of a hyper-token's squared length that lies along the shared mean `c` (→1 means the vector *is* essentially `c`). Here output 0.117 / input 0.216 — `c` accounts for only a small part.
- **tok floor** — the same cos-with-mean measured on **ordinary base tokens** in this space.
- **rand-init** — cos-with-mean for a **random-init (untrained)** encoder. It must sit clearly **above** rand-init for the ruler to be **learned in training** rather than architectural.

**The ruler is not guaranteed.** It is strong for some checkpoints (v0.5/v0.52
output ≈ 0.998) and weak for others (v0.6.4 output ≈ 0.36). When it is weak,
removing it barely changes anything; when it is strong, removing it is what
exposes the per-hyper-token signal. So we **do not assume it** — steps ④ and ⑤
are reported **both** on the raw vector and after ruler removal, and ① (raw) vs
③ (ruler-removed) already bracket the per-position view. Read the two variants
together: if they agree, the ruler is not doing much; if they diverge, a strong
shared ruler was hiding the signal.

**③ Per-position similarity after removing the ruler.** Subtract `c`, then
repeat ①. It reveals
which position the vector keys on.

**④ Causal minimal pairs — the headline test.** Take a real hyper-token, change
exactly one base token (the **first** vs the **last**), and measure how much the
embedding moves (`1 − cos`). If change-first ≫ change-last the vector is
**prefix-dominated** (reads the head). Reported per K with the first/last ratio, **both** on the raw
vector (④a) and after ruler removal (④b). (The vector *difference* `E₁−E₂` is
unchanged by subtracting a constant, but the *cosine* framing is not — hence both.)

**⑤ Nested growing-prefix examples.** A few (`N = 4`) frequent readable 4-grams,
e.g. `in the United States`: `H2=[in,the] ⊂ H3=[in,the,United] ⊂
H4=[in,the,United,States]` share a prefix and grow. High `cos(H2,H4)` = growing
prefixes stay alike (the encoder keys on the shared head); low = it tracks the
changing tail. Shown both raw and after ruler removal, per example. These are
**illustrative examples for the reader, not aggregate statistics.**

### Why cosine, and why remove the ruler
- Cosine is space-invariant, so it is comparable across the two encoders even
  though they live in different embedding spaces (`tok_emb` vs `lm_head`).
- The shared ruler (especially on the output side) can pin every raw cosine near
  its own value; removing it exposes the per-hyper-token signal.
- A **random-init encoder** baseline separates "learned" from "architectural".

## Trust: bit-exact re-implementation

`enc_lib.py` re-implements the encoder forward in plain PyTorch (CPU, no
FSDP/torchtitan) so it runs in seconds. It is validated **bit-exact**
(`max|Δ| = 0`) against `src/zip2zip_core/model.py:HyperEncoder` for every
checkpoint analyzed (v0.5, v0.52, v0.6.4). Re-run that check whenever the
model code changes.

## Reproduce

`probe.py` analyzes **one checkpoint per run** and writes one report.

1. **Extract** a checkpoint's four tensor groups into `weights/encoders_<tag>.pt`
   (hyper_encoder.\*, hyper_output.\*, tok_embeddings.weight, output.weight) from
   its `model.pt`.
2. Add a preset in `probe.py` `PRESETS`: `name -> (label, path, residual)` —
   `residual=False` for `no_encoder_residual` runs (v0.52, vx0.6.4.2, …).
3. Run from the zip2zip-core uv venv:
   ```bash
   cd zip2zip-core
   HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
     uv run python /dlabscratch1/xinma/hyperenc_probe/probe.py <preset>
   ```
   → writes `results/<preset>.json` and `reports/<preset>.md`.

To compare two runs, generate both reports and read them side by
side — the tables are identical in shape.

## Reading the result

- **output = prefix-sensitive** (change-first ≫ change-last) → it scores H mostly
  by its first token / entry point.
- **input = distributed** (uses all base tokens; residual runs look U-shaped,
  no-residual runs rise toward the tail) → it represents the whole merged span.
- **output "ruler" collapse** (cos-with-mean ≈ 1) that survives removing the
  residual points to an under-utilized output role, not an artifact.

To know whether a finding here is **created by the hyper-encoder** or merely
inherited from the embedding-table geometry, compare against the base-token
control — see [`base_probe.md`](base_probe.md).
