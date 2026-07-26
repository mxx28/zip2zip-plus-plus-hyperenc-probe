# Hyper-token embedding probe (`probe.py`)

**What it asks:** what does a zip2zip hyper-token's embedding actually represent,
and do the *input* (embedding) and *output* (unembedding) hyper-encoders learn
different functions? It runs directly on a trained checkpoint's weights (no
training, no GPU), one checkpoint per run — compare reports across checkpoints
yourself.

---

## What a hyper-token embedding is

A hyper-token `H` is an LZW merge of `K` base tokens `[t1, …, tK]` (K = 2–4).
The model turns it into one vector with a small **flat hyper-encoder** (2-layer
bidirectional transformer, mean-pooled), reading base-token vectors from an
embedding matrix. There are two encoders with two roles:

| role | reads from | used as | question |
|---|---|---|---|
| **input**  | `tok_embeddings` | the embedding injected into the decoder stream in place of H | what does the model *treat H as*? |
| **output** | `lm_head` (`output.weight`) | the row appended to the unembedding matrix; `logit(H) = hidden · E_out(H)` | how does the model *score emitting H*? |

The **final vector** the model uses is
`E(H) = base_vec[t1] + encoder_out(H)` when the residual path is on, or
`E(H) = encoder_out(H)` when it is off (`no_encoder_residual`, e.g. v0.52 / vx0.6.4.2).
We always analyze that final vector — the one the model really uses.

---

## The method, in five steps

Everything reduces to **cosine similarity on the final vector**; the only
variables are *what we compare it to* and *whether we first remove a shared
"ruler" vector*.

**① Raw per-position similarity.** For each hyper-token, cosine of its final
vector with each of its base tokens' vectors, averaged. First look, no
processing. (Often shows the output vector is ≈0 with *all* its base tokens —
which motivates ②.)

**② Does a shared "ruler" exist?** Take the mean vector `c` over all
hyper-tokens. If every hyper-token has `cos(E, c) ≈ 1`, then each one is almost
entirely that shared `c`, which dilutes every raw cosine in ①. We report
`cos-with-mean` (and its min), the average pairwise cosine between distinct
hyper-tokens, and the fraction of squared length that is the mean — against two
controls: **ordinary tokens** in the same space (anisotropy floor) and a
**random-init encoder** (is the collapse learned?).

**The ruler is not guaranteed.** It is strong for some checkpoints (v0.5/v0.52
output ≈ 0.998) and weak for others (v0.6.4 output ≈ 0.36). When it is weak,
removing it barely changes anything; when it is strong, removing it is what
exposes the per-hyper-token signal. So we **do not assume it** — steps ④ and ⑤
are reported **both** on the raw vector and after ruler removal, and ① (raw) vs
③ (ruler-removed) already bracket the per-position view. Read the two variants
together: if they agree, the ruler is not doing much; if they diverge, a strong
shared ruler was hiding the signal.

**③ Per-position similarity after removing the ruler.** Subtract `c`, then
repeat ①. This is the *discriminative* part ("which merge is this?"). It reveals
whether the vector keys on the first token, the last, or all of them. (Removing
`c` is legitimate: for output scoring, `c` is the same additive bias for every
hyper-token, so only the `c`-removed part decides *which* H scores higher.)

**④ Causal minimal pairs — the headline test.** Take a real hyper-token, change
exactly one base token (the **first** vs the **last**), and measure how much the
embedding moves (`1 − cos`). If change-first ≫ change-last the vector is
**prefix-dominated** (reads the head); if change-last ≫ change-first it is
**tail-weighted**. Reported per K with the first/last ratio, **both** on the raw
vector (④a) and after ruler removal (④b). (The vector *difference* `E₁−E₂` is
unchanged by subtracting a constant, but the *cosine* framing is not — hence both.)

**⑤ Nested example "It is a dog".** `H2=[It,is] → H3=[It,is,a] → H4=[It,is,a,dog]`
share a prefix and grow. High `cos(H2,H4)` = growing prefixes stay alike (the
encoder keys on the shared head); low = it tracks the changing tail. Shown both
raw (⑤a) and after ruler removal (⑤b).

### Why cosine, and why remove the ruler
- Cosine is space-invariant, so it is comparable across the two encoders even
  though they live in different embedding spaces (`tok_emb` vs `lm_head`).
- The shared ruler (especially on the output side) can pin every raw cosine near
  its own value; removing it exposes the per-hyper-token signal.
- A **random-init encoder** baseline separates "learned" from "architectural".

---

## Trust: bit-exact re-implementation

`enc_lib.py` re-implements the encoder forward in plain PyTorch (CPU, no
FSDP/torchtitan) so it runs in seconds. It is validated **bit-exact**
(`max|Δ| = 0`) against `src/zip2zip_core/model.py:HyperEncoder` for every
checkpoint analyzed (v0.5, v0.52, v0.6.4). Re-run that check whenever the
model code changes.

---

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

Presets shipped: `v05`, `v052`, `v064` (weights present); `vx0642` (add weights
when trained). To compare two runs, generate both reports and read them side by
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
