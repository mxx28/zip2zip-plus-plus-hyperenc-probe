# Reproduction

There are two supported reproduction levels.

## Render from committed measurements

This path does not require checkpoints, a GPU, Hugging Face model access, or
WikiText-2. It reads only the committed CSV files in `data/`.

```bash
python -m pip install -r requirements.txt
python plot_phi_substitution.py
python plot_llama_substitution.py
python plot_sequence_probe.py
```

The commands write:

```text
figures/substitution_phi_hyper_tokens.pdf
figures/substitution_phi_hyper_tokens.png
figures/substitution_phi_base_table_control.pdf
figures/substitution_phi_base_table_control.png
figures/substitution_llama.pdf
figures/substitution_llama.png
figures/sequence_phi.pdf
figures/sequence_phi.png
```

## Recompute the measurements

End-to-end reproduction can read a public Zip2Zip++ Hugging Face export or one
of the legacy extracted checkpoints listed in [checkpoints.md](checkpoints.md).
It also requires WikiText-2. All measurement scripts run on CPU.

You may set `HF_HOME` and `HF_DATASETS_CACHE` before running the commands.
The scripts otherwise use the standard Hugging Face cache locations and contain
no machine-specific paths.

### Four published Zip2Zip++ models

The user-facing files live on each repository's `hf` revision:

```bash
python probe.py --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct --revision hf --name llama1b
python probe.py --repo-id epfl-dlab/zip2zip-pp-Llama-3.2-3B-Instruct --revision hf --name llama3b
python probe.py --repo-id epfl-dlab/zip2zip-pp-Phi-3.5-mini-instruct --revision hf --name phi4b
python probe.py --repo-id epfl-dlab/zip2zip-pp-Phi-3-medium-4k-instruct --revision hf --name phi14b
```

Run the matching BPE control by changing `probe.py` to `probe_base.py` and
keeping the same arguments. For example, `--name llama1b` writes
`results/llama1b.json` and `results/llama1b_base.json`.

The loader downloads `zip2zip_encoders.safetensors` and only the decoder
shard(s) containing the two embedding tables. The 14B probe therefore does not
load the full decoder into memory, although its embedding tables are still
large.

### Exact legacy paper workflow

The commands below retain the old local `weights/*.pt` interface and reproduce
the committed paper artifacts exactly.

#### Phi substitution probe and BPE control

```bash
python probe.py v064
python probe_base.py base
python prepare_paper_data.py --probe substitution --preset v064
python plot_phi_substitution.py
```

`probe.py` writes the aggregate measurement to `results/v064.json`.
`probe_base.py` writes `results/base.json`. The data-preparation command
selects only the raw embedding measurements used in the paper and exports them
to `data/substitution_phi.csv`.

#### Llama-3.2-3B substitution probe

```bash
python probe.py llama3B_v064_untied
python probe.py llama3B_v064_tied
python probe_base.py base_llama3B

python prepare_paper_data.py --probe substitution --preset llama3B_untied
python prepare_paper_data.py --probe substitution --preset llama3B_tied
python plot_llama_substitution.py
```

Llama-3.2 ties `tok_embeddings` and `lm_head`, so the two base-table series
coincide. In the tied-hyper-encoder run, the input and output hyper-encoder
series also coincide by construction.

#### Sequence probe

```bash
python prepare_paper_data.py --probe sequence
python plot_sequence_probe.py
```

The preparation command loads `weights/encoders_v064.pt`, embeds the six
contiguous sub-spans of `It is a dog`, and writes
`data/sequence_phi.csv`.

## Data flow

```text
HF export or extracted checkpoint + tokenizer + WikiText-2
                    |
                    v
          probe.py / probe_base.py
                    |
                    v
               results/*.json
                    |
                    v
           prepare_paper_data.py
                    |
                    v
                  data/*.csv
                    |
                    v
                plot_*.py
                    |
                    v
          figures/*.{pdf,png}
```

The plotting scripts intentionally do not import PyTorch, load checkpoints, or
read the full JSON reports.
