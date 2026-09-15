# Sequence probe

The **Sequence probe** is the illustrative span-level analysis reported in the
paper. It complements the aggregate substitution probe by showing how
representations change as the extent of a span changes.

## Construction

The source span is:

```text
It is a dog
```

With the Phi-3.5-mini tokenizer, it produces four base tokens. The probe embeds
all six contiguous sub-spans of length at least two, in this order:

1. `[It, is]`
2. `[It, is, a]`
3. `[It, is, a, dog]`
4. `[is, a]`
5. `[is, a, dog]`
6. `[a, dog]`

For each of the input and output hyper-encoders, the implementation computes
the full pairwise cosine-similarity matrix between their raw final embeddings.
The sequence probe uses the Phi v0.6.4 checkpoint with the residual connection
enabled.

## Interpretation

The output hyper-encoder is prefix-aligned: the three nested spans beginning
with `It, is` form a high-similarity block. Their pairwise similarities are at
least 0.88, while cross-prefix pairs are substantially lower.

The input hyper-encoder does not form an equivalent prefix block. Its
similarities vary more smoothly with shared constituent overlap, consistent
with the balanced geometry observed by the substitution probe.

This is an illustrative analysis of one sequence, not an aggregate corpus
statistic.

## Reproduction

```bash
python prepare_paper_data.py --probe sequence
python plot_sequence_probe.py
```

The first command loads `weights/encoders_v064.pt` and writes
`data/sequence_phi.csv`. The second command reads only that CSV and writes
`figures/sequence_phi.{pdf,png}`.
