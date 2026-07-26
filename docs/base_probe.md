# Base-token BPE control probe (`probe_base.py`)

**What it asks — and why it exists.** The hyper-token probe
([`hyper_probe.md`](hyper_probe.md)) finds that a hyper-token's embedding is
strongly keyed on its **first** base token (e.g. v0.6.4 output). But is that a
property the *hyper-encoder creates*, or is it just **inherited from the
embedding table's geometry**? A BPE base token is *also* a composition — so we
run the same probes on ordinary base tokens as a control. If the raw embedding
table already shows first-piece dominance, part of the hyper-token effect is not
special; if it does not, the hyper-encoder is imposing the geometry.

This is the control suggested by the PhD reviewer.

---

## The key idea: a base token is a merge too

A BPE base token `T` was built by the tokenizer from smaller pieces. So it has
the **same "composed of ordered sub-pieces" structure** as a hyper-token — the
difference is that a hyper-token's vector is *built by an encoder* from its
pieces, while a base token's embedding is a **free lookup parameter** learned in
pretraining. Comparing the two isolates what the encoder adds.

The pieces of `T` are themselves **real vocab tokens with real embedding rows**,
so we can compute the exact same cosines — no encoder involved.

---

## Where the "tree" comes from, and why the split is unique

The tree is **the tokenizer's own BPE merge list** — 61,249 rules in the
Phi-3.5 `LlamaTokenizerFast`, read from `tok.backend_tokenizer.to_str()` →
`model.merges`. Each rule `[A, B]` means "merge `A` and `B` into `AB`".

**Naively iterating the merge list does NOT give a unique split**: 69.8% of
tokens have several candidate pairs whose concatenation equals the same string
(`▁the` = `▁t`+`he` = `▁th`+`e` = `▁`+`the`), and there are more merge rules
(61,249) than vocab tokens (32,011). The candidates are not the token's actual
construction.

**The unique decomposition is the deterministic BPE process itself.** Greedy BPE
starts from `T`'s characters and repeatedly merges the highest-priority
(lowest-rank) adjacent pair; each merge reduces the token count by exactly one,
so the sequence passes through **every** count from `len(chars)` down to 1, and
each intermediate state is unique. Therefore:

- **K = 2** (`bpe_split`): the **last** merge applied is `T`'s unique immediate
  binary split `T = A + B` (`A` = first piece, `B` = last piece).
- **Any K** (`decompK`): the unique intermediate state with **exactly K tokens**
  is `T`'s ordered K-piece decomposition — a flat, ordered sequence, exactly
  analogous to a hyper-token's K base tokens. Verified: for K=4, 2000/2000
  sampled tokens decompose into pieces that are all real vocab tokens, e.g.

  ```
  ▁international : K2 [▁intern, ational] · K3 [▁in, tern, ational] · K4 [▁in, tern, ation, al]
  ▁understanding : K2 [▁understand, ing] · K3 [▁under, stand, ing] · K4 [▁under, st, and, ing]
  ▁information   : K2 [▁inform, ation]   · K3 [▁in, form, ation]   · K4 [▁in, f, orm, ation]
  ```

So the earlier worry that "K≥3 needs an arbitrary rule" was wrong — the
rank-greedy process is deterministic, so every K is unique. (Current
`probe_base.py` ships the K=2 probes; the K-general `decompK` is available to
extend ① to K=3/4 when wanted.)

---

## What is computed (K=2), on the raw embedding tables

Run on **both** raw matrices — `tok_embeddings` (the input encoder's space) and
`lm_head`/`output.weight` (the output encoder's space) — from a checkpoint
(v0.6.4; base tables are ~frozen, so one run represents them). No encoder, no
ruler assumed; raw and mean-removed variants both reported.

- **① per-piece cosine**: `cos(emb[T], emb[A])` (first piece) vs
  `cos(emb[T], emb[B])` (last piece). Is a base token more like its first or last
  piece?
- **④ minimal pairs**: real vocab-token pairs that share one piece and differ in
  the other —
  - *differ-first*: share the last piece `B`, different `A` (`A+B` vs `A'+B`);
  - *differ-last*: share the first piece `A`, different `B` (`A+B` vs `A+B'`);
  measure cosine distance. `first/last > 1` = the first piece matters more.
  (Unlike the hyper probe, which replaces a slot with a *random* token, here we
  use *real* shared-piece vocab pairs, because only real merges have embeddings.)
- **② anisotropy**: base-table `cos-with-mean` as the "ruler" analog (context).

---

## Result and how to read it

The base embedding table shows **no first-piece dominance** — the effect is
encoder-specific:

| K=2 quantity | base `lm_head` | v0.6.4 hyper **output** | base `tok_emb` | v0.6.4 hyper **input** |
|---|--:|--:|--:|--:|
| cos(·, first piece) | 0.25 | **0.90** | 0.09 | 0.48 |
| cos(·, last piece)  | 0.12 | 0.15 | 0.12 | 0.15 |
| minpair first/last  | **1.05×** | **6.42×** | **0.99×** | 0.54× |

- Changing the first vs last piece moves a **base** embedding essentially
  equally (0.99×–1.05×); a base token is only weakly related to either piece.
- The hyper-encoder's strong "reads the head" geometry (output 6.42× at K=2,
  22.95× at K=4, cos 0.90) is therefore **created by the encoder**, not inherited
  from BPE embedding-space structure.

**Interpretation matrix.** base flat + hyper peaked → encoder-imposed (our case).
base peaked + hyper peaked → inherited geometry, temper the claim. This control
turns the vx0.6.4.2 ablation into a decisive test: if removing the residual path
drops the output ratio from ~22× toward the base level (~1×), the first-piece
dominance was an initialization artifact.

### Caveats
- **Granularity**: base pieces are sub-word BPE units (often 1–2 chars), hyper
  pieces are whole words. The analogy is *structural* (composition), not semantic.
- A base embedding is not composed, so the ruler/residual notions map only
  loosely; the anisotropy number is context, not a claim.
- differ-first / differ-last pairs share only one BPE piece, so both are
  near-orthogonal (distance ≈ 0.9) — the informative quantity is the *ratio*, not
  the absolute distance.

## Reproduce

```bash
cd zip2zip-core
HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
  uv run python /dlabscratch1/xinma/hyperenc_probe/probe_base.py
```
→ writes `results/base.json` and `reports/base.md` (includes the side-by-side
with the v0.6.4 hyper-token K=2 numbers).
