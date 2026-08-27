# Base-token BPE control probe (`probe_base.py`)

**What it asks — and why it exists.** The hyper-token probe
([`hyper_probe.md`](hyper_probe.md)) finds that a hyper-token's embedding may have some structre. But is that a
property the *hyper-encoder creates*, or is it just **inherited from the
embedding table's geometry**? 

A BPE base token is *also* a composition — so we
run the same probes on ordinary base tokens as a control. 

If the raw embedding table already shows similar pattern, part of the hyper-token effect is not special; if it does not, the hyper-encoder is imposing the geometry.

## The key idea: a base token is a merge too

A BPE base token `T` was built by the tokenizer from smaller pieces. So it has
the **same "composed of ordered sub-pieces" structure** as a hyper-token — the
difference is that a hyper-token's vector is *built by an encoder* from its
pieces, while a base token's embedding is a **free lookup parameter** learned in
pretraining. Comparing the two isolates what the encoder adds.

The pieces of `T` are themselves **real vocab tokens with real embedding rows**,
so we can compute the exact same cosines — no encoder involved.

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
  ▁international : 
  K2 [▁intern, ational]
  K3 [▁in, tern, ational]
  K4 [▁in, tern, ation, al]

  ▁understanding : 
  K2 [▁understand, ing] 
  K3 [▁under, stand, ing] 
  K4 [▁under, st, and, ing]

  ▁information   : 
  K2 [▁inform, ation]   
  K3 [▁in, form, ation]   
  K4 [▁in, f, orm, ation]
  ```

So the earlier worry that "K≥3 needs an arbitrary rule" was wrong — the
rank-greedy process is deterministic, so every K is unique. `probe_base.py` now
runs **K = 2/3/4** (via the K-general `decompK`), so the control lines up
position-for-position with the hyper-token probe.

**Coverage and minimal-pair pools** (all counts are exact, not sampled):

| K | base tokens with a valid all-vocab K-split | change-first pairs | change-last pairs |
|---|--:|--:|--:|
| 2 | 29,612 | 13,258 | 12,722 |
| 3 | 27,465 |  9,247 |  9,216 |
| 4 | 22,733 |  5,132 |  5,510 |

So sample sizes are thousands per K at every position — the binding constraint
is not count but that each piece must be a real vocab token with an embedding.

## What is computed (K=2/3/4), on the raw embedding tables

Run on **both** raw matrices — `tok_embeddings` (the input encoder's space) and
`lm_head`/`output.weight` (the output encoder's space) — from a checkpoint
(v0.6.4; base tables are ~frozen, so one run represents them). No encoder.
Reported **both raw and demean** (ruler removed from `emb[T]` only, as in the
hyper probe) so the two reports line up cell-for-cell and you can compare against
whichever hyper variant you need — though the base ruler is weak, so raw ≈ demean
here.

- **① per-piece cosine**: `cos(emb[T], emb[p_i])` for each position `i` — is a
  base token more like its first piece or a later one?
- **④ Substitution probe**: real vocab-token pairs that share K−1 pieces and differ
  in exactly one position —
  - *change-first*: share pieces `p2..pK`, differ in `p1`;
  - *change-last*: share pieces `p1..p(K-1)`, differ in `pK`;
  measure cosine distance. `first/last > 1` = the first piece matters more. Uses
  **all** disjoint such pairs (counts in the table above), so it is
  deterministic — no sampling.

  (Unlike the hyper probe, which replaces a slot with a *random* token, here we
  use *real* shared-piece vocab pairs, because only real base tokens have
  embeddings.)
- **② anisotropy**: base-table `cos-with-mean` as the "ruler" analog (context).


## Result and how to read it

The base embedding table shows **no first-piece dominance at any K** — the
effect is encoder-specific. The key quantity is the **minimal-pair first/last
ratio** (ruler-robust, so it compares cleanly across every column); output space
(`lm_head`) then input space (`tok_emb`), each with the raw base table vs the
residual-on (v0.6.4) and residual-off (vx0.6.4.2) hyper-encoders.

**④ substitution first/last ratio (prefix-dominance), per K**

| K | base `lm_head` | v0.6.4 out | vx0.6.4.2 out | base `tok_emb` | v0.6.4 in | vx0.6.4.2 in |
|---|--:|--:|--:|--:|--:|--:|
| 2 | **1.06×** | 6.42×  | 1.94× | **0.99×** | 0.54× | 0.75× |
| 3 | **1.01×** | 16.58× | 3.84× | **0.97×** | 0.80× | 0.40× |
| 4 | **0.91×** | 24.87× | 3.81× | **0.94×** | 0.98× | 0.31× |

**① first-piece cosine (pos1), per K** — raw.

| K | base `lm_head` | v0.6.4 out | vx0.6.4.2 out† | base `tok_emb` | v0.6.4 in | vx0.6.4.2 in |
|---|--:|--:|--:|--:|--:|--:|
| 2 | **0.25** | 0.90 | 0.01† | **0.09** | 0.48 | 0.08 |
| 3 | **0.12** | 0.87 | 0.01† | **0.04** | 0.49 | 0.04 |
| 4 | **0.07** | 0.86 | 0.01† | **0.02** | 0.50 | 0.04 |

† vx0.6.4.2's output collapses onto a shared ruler (cos-with-mean 0.9988), so its
*raw* per-piece cosine is ≈0 (dominated by the ruler direction, **not** evidence
that it ignores the pieces); after ruler removal the first-piece cosine is 0.30 /
0.27 / 0.25 for K=2/3/4. The ④ ratio is ruler-robust (raw 1.94× ≈ demean 1.93× at
K=2), which is why ④ — not raw ① — is the honest cross-checkpoint comparison.

**Reading it:**
- **Base table: flat at every K.** Changing the first vs last piece moves a base
  embedding essentially equally (0.91×–1.06×), and if anything the ratio *drifts
  below 1* as K grows — the raw embedding space never prefers the first piece.
- **The hyper-encoder creates the prefix geometry, and the residual amplifies it
  with K.** v0.6.4 output climbs 6.42× → 16.58× → 24.87× as K grows; strip the
  residual (vx0.6.4.2) and it flattens to a mild, roughly K-independent ~2–4×.
  So a *real, learned* head-reading (~2–4×) survives, but the dramatic
  K-scaling is a residual/initialization effect, not learned structure.
- **The input side is never prefix-dominated** (≤1× everywhere); without the
  residual it becomes increasingly *tail*-weighted as K grows (0.75× → 0.40× →
  0.31×).

### Caveats
- A base embedding is a free lookup, not composed by an encoder — so the
  ruler/residual notions map only loosely; ② is context, not a claim.
- change-first / change-last pairs share only K−1 of K pieces, so both distances
  are large (near-orthogonal) — the informative quantity is the *ratio*, not the
  absolute distance.
- Base pieces are sub-word BPE units (often 1–2 chars) while hyper pieces are
  whole base tokens; the analogy is *structural* (ordered composition), not
  semantic.

## Reproduce

```bash
cd zip2zip-core
HF_HOME=/dlabscratch1/gentilin/.cache/huggingface \
  uv run python /dlabscratch1/xinma/hyperenc_probe/probe_base.py
```
→ writes `results/base.json` and `reports/base.md` (includes the side-by-side
with the v0.6.4 hyper-token K=2 numbers).
