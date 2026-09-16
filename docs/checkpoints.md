# Checkpoints

The primary probe workflow reads the four self-contained Zip2Zip++ releases
from Hugging Face. Each model repository keeps the original training checkpoint
on `main` and the user-facing export on the `hf` revision.

## Zip2Zip++ releases

| Probe name | Base model | Hugging Face repository |
|---|---|---|
| `llama1b` | Llama-3.2-1B-Instruct | `epfl-dlab/zip2zippp-Llama-3.2-1B-Instruct` |
| `llama3b` | Llama-3.2-3B-Instruct | `epfl-dlab/zip2zippp-Llama-3.2-3B-Instruct` |
| `phi4b` | Phi-3.5-mini-instruct | `epfl-dlab/zip2zippp-Phi-3.5-mini-instruct` |
| `phi14b` | Phi-3-medium-4k-instruct | `epfl-dlab/zip2zippp-Phi-3-medium-4k-instruct` |

For example:

```bash
python probe.py \
  --repo-id epfl-dlab/zip2zippp-Llama-3.2-1B-Instruct \
  --revision hf \
  --name llama1b
python probe_base.py \
  --repo-id epfl-dlab/zip2zippp-Llama-3.2-1B-Instruct \
  --revision hf \
  --name llama1b
```

The first command writes `results/llama1b.json`; the second writes
`results/llama1b_base.json` and includes the first result in its comparison
when available.

The loader reads `zip2zip_config.json` and
`zip2zip_encoders.safetensors`, then downloads only the decoder shard or
shards containing `model.embed_tokens.weight` and `lm_head.weight`. It does
not load the decoder or the raw training `model.pt` in memory. Standard
Hugging Face authentication and cache settings apply.

## Expected HF export files

```text
zip2zip_config.json
zip2zip_encoders.safetensors
tokenizer files
model.safetensors
# or model-*.safetensors + model.safetensors.index.json
```

The architecture is read from `zip2zip_config.json`, so the probe supports
different model dimensions, latent encoder dimensions, attention-head counts,
and tied or untied hyper-encoders without model-specific code.

## Legacy paper artifacts

The original extracted files under `weights/` remain supported so the exact
committed paper workflow is unchanged:

| Local file | Base model | Legacy preset |
|---|---|---|
| `encoders_v064.pt` | Phi-3.5-mini-instruct | `v064` / `base` |
| `encoders_llama3B_v064_untied.pt` | Llama-3.2-3B-Instruct | `llama3B_v064_untied` / `base_llama3B` |
| `encoders_llama3B_v064_tied.pt` | Llama-3.2-3B-Instruct | `llama3B_v064_tied` |

These local files contain `hyper_encoder.*`, `hyper_output.*`,
`tok_embeddings.weight`, and `output.weight`. They are ignored by Git and
are not required when probing a public release.
