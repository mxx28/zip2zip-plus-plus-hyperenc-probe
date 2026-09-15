# Checkpoints

The probe scripts load extracted checkpoint files rather than complete
training checkpoints. These files are kept under `weights/`, are ignored by
Git, and have not yet been published.

## Paper checkpoints

| File expected under `weights/` | Base model | Configuration | Used by |
|---|---|---|---|
| `encoders_v064.pt` | `microsoft/Phi-3.5-mini-instruct` | zip2zip v0.6.4, untied input/output hyper-encoders, residual enabled | Phi substitution probe, Phi BPE control, sequence probe |
| `encoders_llama3B_v064_untied.pt` | `meta-llama/Llama-3.2-3B-Instruct` | zip2zip v0.6.4, untied hyper-encoders, residual enabled | Llama untied substitution probe and Llama base-table control |
| `encoders_llama3B_v064_tied.pt` | `meta-llama/Llama-3.2-3B-Instruct` | zip2zip v0.6.4, tied hyper-encoder, residual enabled | Llama tied substitution probe |

The Llama base model ties its input and output embedding tables. The tied
zip2zip configuration additionally uses the same hyper-encoder for both roles.

## Required tensor groups

Each extracted file must contain:

```text
hyper_encoder.*
hyper_output.*
tok_embeddings.weight
output.weight
```

`enc_lib.load_pair()` reconstructs the input and output hyper-encoders from
these tensors. The standalone implementation has been checked against the
training implementation for the relevant flat two-layer, 3072-dimensional
encoder architecture.

## Publication status

The files are currently local-only. Until they are uploaded, readers can
reproduce every rendered plot from the committed CSV files but cannot rerun the
checkpoint-to-measurement stage.

When the checkpoints are published, add for each file:

- its public download URL;
- a SHA-256 checksum;
- the originating training checkpoint identifier;
- the exact extraction command or script.

## Non-paper checkpoints

Historical v0.5/v0.52 results and exploratory diagnostics are stored under
`extras/`. They are not required for any paper figure. The legacy `v05`
preset expects `encoders_v05.pt`, which is not present locally; this does not
affect paper reproduction.
