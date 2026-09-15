# Extra diagnostics

This directory contains exploratory analyses and reports that are not used in
the paper figures:

- shared-direction (“ruler”) measurements and ruler removal;
- per-position cosine diagnostics;
- growing-prefix examples;
- historical v0.5 and v0.52 results;
- the out-of-distribution base-pieces-through-hyper-encoder control.

They are retained for provenance and possible follow-up work. The paper-facing
workflow uses only the raw final-embedding substitution measurements and the
sequence probe documented in `../docs/`.

`probe.py` and `probe_base.py` still compute auxiliary fields for backwards
compatibility, but their paper-facing CSV export ignores those fields.
