# Controlled comparison — identical objects, raw lookup vs hyper-encoder

Every column analyzes the **same** base tokens and the **same** minimal pairs (from `probe_base.py`'s BPE decompositions). The only difference is how each token is embedded: the **base** columns use the raw lookup `emb[T]`; the encoder columns feed the same pieces `[p1..pK]` through the hyper-encoder. Unlike `base.md`'s side-by-side (which used corpus n-grams for the hyper side), here the hyper side is the identical object set — so this isolates raw-lookup vs encoder with data held fixed, and uses real shared-piece pairs (no random replacement). **Bold = base control** (raw table, no encoder).

**④ substitution cos_last/cos_first ratio, per K**

| K | **base `lm_head`** | v0.6.4 out | **base `tok_emb`** | v0.6.4 in |
|:--|--:|--:|--:|--:|
| 2 | **1.27×** | 4.18× | **0.87×** | 0.88× |
| 3 | **1.0×** | 2.86× | **0.75×** | 0.99× |
| 4 | **0.82×** | 2.39× | **0.69×** | 1.0× |

**① first-piece cosine (pos1), per K** — raw (read together with the substitution ratios)

| K | **base `lm_head`** | v0.6.4 out | **base `tok_emb`** | v0.6.4 in |
|:--|--:|--:|--:|--:|
| 2 | **0.2469** | 0.9242 | **0.0931** | 0.5232 |
| 3 | **0.1187** | 0.8944 | **0.0392** | 0.4989 |
| 4 | **0.0664** | 0.8829 | **0.0187** | 0.4649 |

---

*Repro: `uv run python probe_base_hyper.py` (after `probe_base.py`). Data: `results/base_hyper_v064.json`, `results/base.json`.*