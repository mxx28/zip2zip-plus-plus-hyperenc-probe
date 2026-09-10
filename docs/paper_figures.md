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

Prepare and render the combined Llama-3.2-3B figure:

```bash
python prepare_figure_data.py --figure 1 --preset llama3B_untied
python prepare_figure_data.py --figure 1 --preset llama3B_tied
python make_paper_fig_llama.py
```

Render the figures from CSV:

```bash
python make_paper_fig.py
python make_paper_fig_llama.py
python make_span_similarity_fig.py
```

| figure | CSV input | PDF output | PNG output |
|---|---|---|---|
| Phi hyper-token substitution | `figure_data/figure1_substitution.csv` | `paper_fig_phi_hyper.pdf` | `paper_fig_phi_hyper.png` |
| Phi base-token substitution | `figure_data/figure1_substitution.csv` | `paper_fig_phi_base.pdf` | `paper_fig_phi_base.png` |
| Figure 6 | `figure_data/figure6_subspan_similarity.csv` | `paper_fig_subspan_similarity.pdf` | `paper_fig_subspan_similarity.png` |
| Llama-3.2-3B comparison | `figure_data/figure1_substitution_llama3B_{untied,tied}.csv` | `paper_fig_llama.pdf` | `paper_fig_llama.png` |

`prepare_figure_data.py --figure 1` exports the already committed aggregate
measurements in `results/v064.json` and `results/base.json`. Figure 1 uses the
raw final-vector variant, matching the definition of the ratio in the paper.

The Llama-3.2-3B presets read `results/llama3B_v064_{untied,tied}.json` and the
Llama base-table control `results/base_llama3B.json`
(`python probe_base.py base_llama3B`). Two series coincide by construction in
the combined figure: Llama-3.2 ties
`tok_embeddings` and `lm_head`, so the base-token row is one measurement drawn
twice, and in the tied-HE run one encoder serves both roles, so its hyper-token
row is too. All three rows share the same per-column limits for direct comparison.

`prepare_figure_data.py --figure 6` loads the frozen v0.6.4 encoders and
measures all six contiguous subspans of the four-token span `It is a dog`.
This is a small deterministic forward pass, not a new experiment or a training
run. The CSV records the source phrase, tokenization, checkpoint, vector
variant, labels, and full-precision cosine similarities.

## Copy-ready captions

**Substitution probe.** Prefix and suffix panels report the mean cosine
similarity between an original embedding and the same embedding after replacing
its first or last constituent. The ratio panel reports
`r_K = cos_last / cos_first`: values above one are prefix-aligned, values below
one are suffix-aligned, and values near one are balanced. For Phi v0.6.4, the
output hyper-encoder is strongly prefix-aligned (`r_4 = 4.91`), while the input
hyper-encoder is balanced (`r_4 = 0.99`). The Phi base-token control is rendered
as a separate 1x3 figure; the Llama figure stacks untied, tied, and base rows.
All paper panels use raw embeddings, without ruler removal.

**Figure 6 — Pairwise similarity of contiguous subspans.** For the four-token
span `It is a dog`, we compute cosine similarity between the embeddings of all
six contiguous subspans of length at least two. Left: output hyper-encoder
embeddings form clear blocks for spans sharing the same prefix; for example,
the three spans beginning with `It, is` have pairwise similarities between
0.88 and 0.94. Right: input hyper-encoder similarities vary more smoothly with
constituent overlap, consistent with a meaning-sensitive representation rather
than positional prefix matching. Values are from the frozen v0.6.4 checkpoint
and use the raw final embeddings.
