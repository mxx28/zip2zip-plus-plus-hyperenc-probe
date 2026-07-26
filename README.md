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
"ruler" · ③ ruler-removed per-position · ④ causal minimal pairs · ⑤ nested) so
their reports read side by side.

## Layout

```
enc_lib.py        faithful, bit-exact (max|Δ|=0) encoder forward + load_pair(path)
probe.py          hyper-token probe   → results/<preset>.json + reports/<preset>.md
probe_base.py     base-token BPE control → results/base.json + reports/base.md
corpus.txt        hand-written English prose; hyper-tokens = its real n-grams
weights/          extracted per-checkpoint tensors (hyper_encoder, hyper_output,
                  tok_embeddings, output.weight) — small, not the 17GB model.pt
results/          <name>.json + <name>.log
reports/          <name>.md   (reports/html/ : earlier hand-designed HTML reports)
```

## Run

```bash
cd zip2zip-core        # use its uv venv (torch, transformers)
HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
  uv run python /dlabscratch1/xinma/hyperenc_probe/probe.py v064     # a checkpoint
HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
  uv run python /dlabscratch1/xinma/hyperenc_probe/probe_base.py     # the control
```

Presets in `probe.py`: `v05`, `v052`, `v064` (weights present); `vx0642` (add
weights when trained). See each method doc for how to extract weights, add a
preset, and read the tables.

## Headline finding so far

The hyper-encoder **output** vector is strongly keyed on the hyper-token's
**first** base token (v0.6.4: 22.95× at K=4), driven by residual initialization;
the **input** vector is more distributed. The base-token control shows the raw
embedding table has **no** such first-piece dominance (~1×), so this is
**encoder-created, not BPE-inherited**. The `vx0.6.4.2` run (residual path removed)
is the decisive test.
