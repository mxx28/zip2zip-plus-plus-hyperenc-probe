# Substitution probe

## Question

The substitution probe measures which end of a learned LZW hyper-token has more
influence on its embedding. For a hyper-token with constituents
$H=(t_1,\ldots,t_K)$, construct two perturbations:

- **First substitution:** replace $t_1$ and keep $t_2,\ldots,t_K$.
- **Last substitution:** keep $t_1,\ldots,t_{K-1}$ and replace $t_K$.

For an example span $ABCD$, these have the form $XBCD$ and $ABCY$.

Let $E(H)$ be the final embedding used by the model. The probe averages

$$
\cos_{\mathrm{first}} =
\cos(E(H), E(H_{\mathrm{replace\ first}}))
$$

and

$$
\cos_{\mathrm{last}} =
\cos(E(H), E(H_{\mathrm{replace\ last}})).
$$

The reported positional-asymmetry statistic is

$$
r_K = \cos_{\mathrm{last}} / \cos_{\mathrm{first}}.
$$

If replacing the first constituent changes the embedding more, then
$\cos_{\mathrm{first}}$ is smaller and $r_K>1$: the representation is
**prefix-aligned**. Similarly, $r_K<1$ is **suffix-aligned**, and
$r_K\approx1$ is **balanced**.

## LZW hyper-token sample

The implementation in `probe.py` follows the paper configuration:

1. Tokenize the WikiText-2 raw train split with the tokenizer associated with
   the checkpoint.
2. Enumerate contiguous token spans for $K\in\{2,3,4\}$.
3. Remove any span containing a special token or a token whose vocabulary
   spelling contains a digit. Token IDs are also restricted to the base
   vocabulary used by zip2zip.
4. Deduplicate the spans and retain the 5,000 most frequent spans for each
   length.
5. Draw replacement tokens uniformly from the eligible base-token pool using
   NumPy seed 0.
6. Repeat the random replacement six times and average the cosine similarities.

The six replacement draws reduce Monte Carlo noise. The original embedding is
computed once per span and reused across draws.

## Embeddings

Both learned hyper-encoders are probed:

- The **input hyper-encoder** reads `tok_embeddings` and produces the vector
  inserted into the decoder stream.
- The **output hyper-encoder** reads `output.weight` / `lm_head` and produces
  the vector used to score emission of a hyper-token.

The paper checkpoints use the residual form

$$
E(H) = e(t_1) + \operatorname{HyperEncoder}(t_1,\ldots,t_K).
$$

The figures use this raw final embedding. Ruler removal and other exploratory
variants retained in the full JSON are not used for paper results.

## BPE base-table control

The control asks whether the same asymmetry already exists in the pretrained
embedding tables without a hyper-encoder.

`probe_base.py` reconstructs each vocabulary token's ordered BPE composition
from the tokenizer's merge list. For every $K\in\{2,3,4\}$, it identifies
tokens with a valid decomposition into exactly $K$ real, non-special
vocabulary pieces.

For first substitution, it forms pairs of vocabulary tokens that share pieces
$p_2,\ldots,p_K$ and differ in $p_1$. For last substitution, it forms
pairs that share $p_1,\ldots,p_{K-1}$ and differ in $p_K$. Pair
construction is deterministic: groups and token IDs are sorted, and disjoint
adjacent pairs are used.

The same cosine quantities and ratio are computed directly on:

- `output.weight`, the output embedding table;
- `tok_embeddings.weight`, the input embedding table.

No hyper-encoder is applied in this control.

## Paper-facing artifacts

`prepare_paper_data.py` extracts the raw measurements from the committed JSON
files:

- `results/v064.json`
- `results/base.json`
- `results/llama3B_v064_untied.json`
- `results/llama3B_v064_tied.json`
- `results/base_llama3B.json`

The resulting CSV files in `data/` are the sole inputs to
`plot_phi_substitution.py` and `plot_llama_substitution.py`.
