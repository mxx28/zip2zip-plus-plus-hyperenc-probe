# Hyper-Encoder Embedding Probes

Reproduction code and artifacts for the embedding probes used to study the
learned geometry of zip2zip LZW hyper-tokens.

The repository covers the two experiments reported in the paper:

- **Substitution probe:** replace the first or last constituent of a hyper-token
  and measure how much its embedding changes.
- **Sequence probe:** compare embeddings of all contiguous sub-spans of a short
  sequence.

The paper-facing analyses use the raw final embeddings produced by the trained
input and output hyper-encoders. The BPE experiment is a control measured
directly on the pretrained input and output embedding tables.

## Main results

For a span of length $K$, the substitution probe reports

$$
r_K = \frac{\cos_{\mathrm{last}}}{\cos_{\mathrm{first}}}.
$$

Here, $r_K>1$ is **prefix-aligned**, $r_K<1$ is **suffix-aligned**, and
$r_K\approx1$ is **balanced**.

On Phi-3.5-mini, the output hyper-encoder is strongly prefix-aligned
($r_2=6.54$, $r_3=5.58$, $r_4=4.91$), while the input hyper-encoder
becomes balanced at longer spans ($r_4=0.99$). The BPE base-table control
does not show comparable asymmetry. The Llama-3.2-3B results reproduce the same
qualitative distinction and additionally compare tied and untied
hyper-encoders.

The sequence probe uses the four-token span `It is a dog`. Output embeddings
form blocks for spans that share a prefix, whereas input embeddings vary more
smoothly with constituent overlap.

| Experiment | Model / condition | Committed data | Rendered output |
|---|---|---|---|
| Substitution probe | Phi LZW hyper-tokens | `data/substitution_phi.csv` | `figures/substitution_phi_hyper_tokens.{pdf,png}` |
| BPE base-table control | Phi-3.5-mini | `data/substitution_phi.csv` | `figures/substitution_phi_base_table_control.{pdf,png}` |
| Substitution probe | Llama-3.2-3B, tied and untied | `data/substitution_llama3B_*.csv` | `figures/substitution_llama.{pdf,png}` |
| Sequence probe | Phi LZW hyper-tokens | `data/sequence_phi.csv` | `figures/sequence_phi.{pdf,png}` |

## Reproduce the figures

The committed CSV files are sufficient to render every paper figure; model
weights are not needed for this step.

```bash
python -m pip install -r requirements.txt
python plot_phi_substitution.py
python plot_llama_substitution.py
python plot_sequence_probe.py
```

See [docs/reproduction.md](docs/reproduction.md) for end-to-end measurement
commands and [docs/checkpoints.md](docs/checkpoints.md) for the expected
checkpoint files. The extracted checkpoints have not been published yet.

## Repository layout

```text
enc_lib.py                    standalone hyper-encoder forward pass
probe.py                      LZW hyper-token measurements
probe_base.py                 BPE base-table control
prepare_paper_data.py         export paper-facing CSV data
plot_phi_substitution.py      render the Phi substitution plots
plot_llama_substitution.py    render the Llama substitution plot
plot_sequence_probe.py        render the sequence-probe heatmaps
data/                         committed, paper-facing measurements
figures/                      committed PDF and PNG outputs
results/                      full measurement JSON for paper checkpoints
docs/                         methods and reproduction instructions
weights/                      local extracted checkpoints; ignored by Git
```

The measurement scripts retain auxiliary diagnostic fields in their JSON and
generated reports for provenance and backwards compatibility. These fields are
not used by the paper figures. Generated reports are ignored by Git.

## Methods

- [Substitution probe](docs/substitution_probe.md)
- [Sequence probe](docs/sequence_probe.md)
- [Reproduction](docs/reproduction.md)
- [Checkpoint inventory](docs/checkpoints.md)

The implementation is CPU-compatible and does not require the full training
codebase. End-to-end measurement requires sufficient RAM for the extracted
embedding tables and access to the relevant Hugging Face tokenizers and
WikiText-2.
