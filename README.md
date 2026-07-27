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

### Hyper-token embedding:

The two encoders learn **opposite** geometries, and the residual path decides
how much. Comparing v0.6.4 (residual on) against its ablation vx0.6.4.2
(residual path removed):

- **Output (unembedding) reads the head.** It is prefix-dominated — changing a
  hyper-token's *first* base token moves `E_out` far more than changing its last
  (**22.95× at K=4** in v0.6.4). Removing the residual shrinks this to **3.83×**
  but does not kill it: the head-reading is **real and learned, only amplified by
  the residual**, not created by it.

- **Input (embedding) keys on the opposite end.** With the residual it is roughly
  balanced (**0.98×**); remove it and the input flips to **tail-weighted
  (0.31×)** — the two roles key on opposite ends of the merged span.

- **The residual is what prevents collapse.** With it, each vector is anchored to
  its own first base token and there is **no shared ruler** (cos-with-mean 0.36,
  *below* the random-init baseline 0.63 — more spread out than an untrained
  encoder). Remove it and the output collapses almost entirely onto **one shared
  direction (0.9988)**, well above random-init — a learned, per-merge-
  uninformative ruler; the real per-merge signal survives only after subtracting it.

### Base token embedding:

The base-token control shows the raw embedding table has **no** such first-piece
dominance (~1×), so this geometry is **encoder-created, not BPE-inherited**.
