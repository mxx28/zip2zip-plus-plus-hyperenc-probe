# Paper figures

The paper figures use a two-stage workflow: measurements are first saved as
committed CSV files, and the plotting scripts read those CSV files only.

## Reproduce

Prepare both CSV files:

```bash
python prepare_figure_data.py
```

Or prepare one figure at a time:

```bash
python prepare_figure_data.py --figure 1
python prepare_figure_data.py --figure 6
```

Render the figures from CSV:

```bash
python make_paper_fig.py
python make_span_similarity_fig.py
```

| figure | CSV input | PDF output | PNG output |
|---|---|---|---|
| Figure 1 | `figure_data/figure1_substitution.csv` | `paper_fig_untied.pdf` | `paper_fig_untied.png` |
| Figure 6 | `figure_data/figure6_subspan_similarity.csv` | `paper_fig_subspan_similarity.pdf` | `paper_fig_subspan_similarity.png` |

`prepare_figure_data.py --figure 1` exports the already committed aggregate
measurements in `results/v064.json` and `results/base.json`. Figure 1 uses the
raw final-vector variant, matching the definition of the ratio in the paper.

`prepare_figure_data.py --figure 6` loads the frozen v0.6.4 encoders and
measures all six contiguous subspans of the four-token span `It is a dog`.
This is a small deterministic forward pass, not a new experiment or a training
run. The CSV records the source phrase, tokenization, checkpoint, vector
variant, labels, and full-precision cosine similarities.

## Copy-ready captions

**Figure 1 — Substitution probe.** Each panel shows the mean cosine distance
after replacing the first or last constituent of a span. Top row: LZW
hyper-token embeddings from the v0.6.4 model; bottom row: raw BPE base-token
embeddings. The right column reports their ratio
`r_K = d_first / d_last`. The output hyper-encoder is
strongly prefix-aligned, with the asymmetry increasing with span length
`r_4 = 21.1`; the input hyper-encoder is near-balanced at `K = 4`, and neither
base embedding table exhibits positional asymmetry `r_K ≈ 1`.

**Figure 6 — Pairwise similarity of contiguous subspans.** For the four-token
span `It is a dog`, we compute cosine similarity between the embeddings of all
six contiguous subspans of length at least two. Left: output hyper-encoder
embeddings form clear blocks for spans sharing the same prefix; for example,
the three spans beginning with `It, is` have pairwise similarities between
0.88 and 0.94. Right: input hyper-encoder similarities vary more smoothly with
constituent overlap, consistent with a meaning-sensitive representation rather
than positional prefix matching. Values are from the frozen v0.6.4 checkpoint
and use the raw final embeddings.
