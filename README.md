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
to a writable path internally, so no extra env is needed. Presets in `probe.py` include the Phi runs `v05`, `v052`, `v064` and the two Llama-3.2-3B v0.6.4 runs. See each method doc for how to
extract weights, add a preset, and read the tables.

## Paper figures

Paper plots follow a CSV-first workflow: `prepare_figure_data.py` writes the figure data, then `make_paper_fig.py` and `make_span_similarity_fig.py` read only those CSV files. See [`docs/paper_figures.md`](docs/paper_figures.md) for commands, provenance, and captions.

## Headline finding so far

With the paper definition `r_K = cos_last / cos_first`, the v0.6.4 output
hyper-encoder is strongly prefix-aligned (`6.54`, `5.58`, `4.91` for K=2/3/4),
while the input hyper-encoder approaches balance (`0.63`, `0.90`, `0.99`).
The raw Phi base-token tables do not show the same output effect: their ratios are
`1.27`, `1.00`, `0.82` in output space and `0.87`, `0.75`, `0.69` in input space.
