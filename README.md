# hyperenc_probe

A small, reproducible toolkit for asking **what a zip2zip hyper-token's embedding
actually represents** — and whether that geometry is created by the hyper-encoder
or inherited from the embedding table. Runs directly on a trained checkpoint's
weights (no training, no GPU); one checkpoint per run, compare reports yourself.

## Two probes

| probe | script | method doc | question |
|---|---|---|---|
| **Hyper-token** | `probe.py` | [`docs/hyper_probe.md`](docs/hyper_probe.md) | does a hyper-token's embedding key on its first / last / all base tokens? do the input vs output encoders differ? |
| **Base-token control** | `probe_base.py` | [`docs/base_probe.md`](docs/base_probe.md) | does the raw embedding table already show that geometry (BPE-inherited), or is it encoder-created? |

Both share the same five-step cosine method (① raw per-position · ② shared
"ruler" · ③ ruler-removed per-position · ④ substitution · ⑤ growth) so
their reports read side by side.

## Layout

```
enc_lib.py        faithful, bit-exact (max|Δ|=0) encoder forward + load_pair(path)
probe.py          hyper-token probe (WikiText-2 n-grams) → results/<preset>.json + reports/<preset>.md
probe_base.py     base-token BPE control → results/base.json + reports/base.md
probe_base_hyper.py  control: base BPE pieces through the encoder (OOD, appendix)
weights/          extracted per-checkpoint tensors (hyper_encoder, hyper_output,
                  tok_embeddings, output.weight) — small, not the 17GB model.pt
results/          <name>.json + <name>.log
reports/          <name>.md
```

## Run

```bash
cd zip2zip-core        # use its uv venv (torch, transformers, datasets)
HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
  uv run python /dlabscratch1/xinma/zip2zip-hyperenc_probe/probe.py v064     # a checkpoint
HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
  uv run python /dlabscratch1/xinma/zip2zip-hyperenc_probe/probe_base.py     # the control
```

`probe.py` reads WikiText-2-raw-v1 (`datasets`); it defaults `HF_DATASETS_CACHE`
to a writable path internally, so no extra env is needed. Presets in `probe.py`:
`v05`, `v052`, `v064`, `vx0642` (weights present). See each method doc for how to
extract weights, add a preset, and read the tables.

## Paper figures

Paper plots follow a CSV-first workflow: `prepare_figure_data.py` writes the figure data, then `make_paper_fig.py` and `make_span_similarity_fig.py` read only those CSV files. See [`docs/paper_figures.md`](docs/paper_figures.md) for commands, provenance, and captions.

## Headline finding so far

### Hyper-token embedding:

The two encoders learn **opposite** geometries, and the residual path decides
how much. Comparing v0.6.4 (residual on) against its ablation vx0.6.4.2
(residual path removed):

Numbers below are on **WikiText-2-raw-v1** hyper-tokens (top-5000 frequent
K-grams per K), ④ ratios after ruler removal at K=4.

- **Output (unembedding) reads the head.** It is prefix-dominated — changing a
  hyper-token's *first* base token moves `E_out` far more than changing its last
  (**17.8× at K=4** in v0.6.4, growing with K: 5.3× → 12.6× → 17.8×). Removing the
  residual shrinks this to **~3.7×** but does not kill it: the head-reading is
  **real and learned, only amplified by the residual**, not created by it.

- **Input (embedding) keys on the opposite end.** With the residual it is roughly
  balanced (**0.96×**); remove it and the input flips to **tail-weighted
  (0.37×)** — the two roles key on opposite ends of the merged span.

- **The residual is what prevents collapse.** With it, each vector is anchored to
  its own first base token and there is **no shared ruler** (cos-with-mean 0.39,
  *below* the random-init baseline 0.64 — more spread out than an untrained
  encoder). Remove it and the output collapses almost entirely onto **one shared
  direction (0.9986)**, well above random-init — a learned, per-merge-
  uninformative ruler; the real per-merge signal survives only after subtracting it.

### Base token embedding:

The base-token control shows the raw embedding table has **no** such first-piece
dominance (~1×), so this geometry is **encoder-created, not BPE-inherited**.
