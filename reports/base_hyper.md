# Controlled comparison — identical objects, raw lookup vs hyper-encoder

Every column analyzes the **same** base tokens and the **same** minimal pairs (from `probe_base.py`'s BPE decompositions). The only difference is how each token is embedded: the **base** columns use the raw lookup `emb[T]`; the encoder columns feed the same pieces `[p1..pK]` through the hyper-encoder. Unlike `base.md`'s side-by-side (which used corpus n-grams for the hyper side), here the hyper side is the identical object set — so this isolates raw-lookup vs encoder with data held fixed, and uses real shared-piece pairs (no random replacement). **Bold = base control** (raw table, no encoder).

**④ minpair first/last ratio (prefix-dominance), per K**

| K | **base `lm_head`** | v0.6.4 out | vx0.6.4.2 out | **base `tok_emb`** | v0.6.4 in | vx0.6.4.2 in |
|:--|--:|--:|--:|--:|--:|--:|
| 2 | **1.06×** | 15.87× | 7.64× | **0.99×** | 0.76× | 1.13× |
| 3 | **1.01×** | 29.97× | 4.01× | **0.97×** | 0.97× | 0.67× |
| 4 | **0.92×** | 31.44× | 1.02× | **0.94×** | 1.01× | 0.51× |

**① first-piece cosine (pos1), per K** — raw (hyper output raw ≈0 when a shared ruler dominates — read with ④)

| K | **base `lm_head`** | v0.6.4 out | vx0.6.4.2 out | **base `tok_emb`** | v0.6.4 in | vx0.6.4.2 in |
|:--|--:|--:|--:|--:|--:|--:|
| 2 | **0.2469** | 0.9242 | 0.014 | **0.0931** | 0.5232 | 0.0922 |
| 3 | **0.1187** | 0.8944 | 0.0121 | **0.0392** | 0.4989 | 0.0665 |
| 4 | **0.0664** | 0.8829 | 0.0106 | **0.0187** | 0.4649 | 0.0549 |

---

*Repro: `uv run python probe_base_hyper.py` (after `probe_base.py`). Data: `results/base_hyper_{v064,vx0642}.json`, `results/base.json`.*