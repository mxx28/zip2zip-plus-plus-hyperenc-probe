"""Base-token BPE control (K=2) for the hyper-token findings.

A BPE base token T was created by exactly one merge: T = A + B, where A (first
piece) and B (last piece) are themselves real vocab tokens with real embeddings.
So a base token has the SAME "composed of two smaller pieces" structure as a
K=2 hyper-token — but its embedding is a FREE lookup parameter, not built by an
encoder. Running the same cosine probes on the RAW embedding tables tells us
whether "first-piece dominance" is a property of the embedding space itself
(BPE-induced) or something the hyper-encoder imposes.

K=2 only: the immediate binary merge split is unique. K=3/4 would need a
decomposition rule (deferred).

Probes (on tok_embeddings and lm_head, raw + mean-removed):
  ①  per-piece cosine:  cos(emb[T], emb[A])  vs  cos(emb[T], emb[B])
  ④  minimal pairs:  differ-first  (A+B vs A'+B, share last piece)
                     differ-last   (A+B vs A+B', share first piece)
  ②  ruler/anisotropy of the base table (context)

Compares side-by-side with the v0.6.4 hyper-token K=2 numbers (results/v064.json).
"""
import sys, os, json, re
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
from transformers import AutoTokenizer
os.environ.setdefault("HF_HOME", "/dlabscratch1/gentilin/.cache/huggingface")
rng = np.random.default_rng(0)

tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
vocab = tok.get_vocab()                     # str -> id
SPECIAL = set(tok.all_special_ids)
merges = json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]

# The merges list has many candidate pairs per string (69.8% of tokens have >1),
# so it is NOT a unique split by itself. The UNIQUE immediate binary split of a
# token T is the LAST merge BPE applies when building T from its characters:
# greedily merge the highest-priority (lowest-rank) adjacent pair until one token
# remains; the final merge's two operands are T's two immediate children.
RANK = {}
for i, e in enumerate(merges):
    A, B = e if isinstance(e, list) else e.split(" ")
    RANK[(A, B)] = i

def bpe_split(s):
    """Return (A_str, B_str) = T's unique immediate merge, or None if atomic/unreachable."""
    seq = list(s)                           # start from single characters (▁ is one char)
    if len(seq) < 2:
        return None
    last = None
    while len(seq) > 1:
        best, bi = None, -1
        for i in range(len(seq) - 1):
            r = RANK.get((seq[i], seq[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break                           # no applicable merge → not fully reducible
        last = (seq[bi], seq[bi + 1])
        seq[bi:bi + 2] = [seq[bi] + seq[bi + 1]]
    return last if len(seq) == 1 else None

# build UNIQUE triples (T_id, A_id, B_id): one immediate split per base token
triples = []
for T, ti in vocab.items():
    if ti in SPECIAL:
        continue
    sp = bpe_split(T)
    if sp is None:
        continue
    A, B = sp
    if A in vocab and B in vocab and vocab[A] not in SPECIAL and vocab[B] not in SPECIAL:
        triples.append((ti, vocab[A], vocab[B]))
print(f"K=2 base tokens with a unique binary split: {len(triples)}", flush=True)

Tid = torch.tensor([t[0] for t in triples])
Aid = torch.tensor([t[1] for t in triples])
Bid = torch.tensor([t[2] for t in triples])

# minimal-pair groups
by_first = defaultdict(list)   # A_id -> [T_id ...]   (share first piece; differ in last)
by_last  = defaultdict(list)   # B_id -> [T_id ...]   (share last piece;  differ in first)
for ti, ai, bi in triples:
    by_first[ai].append(ti)
    by_last[bi].append(ti)

def sample_pairs(groups, n=3000):
    keys = [k for k, v in groups.items() if len(v) >= 2]
    out = []
    for k in keys:
        v = groups[k]
        for i in range(0, len(v) - 1, 2):     # disjoint adjacent pairs
            out.append((v[i], v[i + 1]))
    rng.shuffle(out)
    return out[:n]

pairs_share_last = sample_pairs(by_last)    # differ in FIRST piece
pairs_share_first = sample_pairs(by_first)  # differ in LAST piece
print(f"differ-first pairs: {len(pairs_share_last)}   differ-last pairs: {len(pairs_share_first)}", flush=True)

# embedding tables from the v0.6.4 checkpoint (base tables are ~frozen)
sd = torch.load(f"{HERE}/weights/encoders_v064.pt", map_location="cpu", weights_only=True)
TABLES = {"tok_emb (input space)": sd["tok_embeddings.weight"], "lm_head (output space)": sd["output.weight"]}


def cos_ids(emb, xa, xb, center=None):
    a = emb[xa]; b = emb[xb]
    if center is not None:
        a = a - center; b = b - center
    return F.cosine_similarity(a, b, dim=-1)


def analyze(emb):
    R = {}
    c = emb[Tid].mean(0, keepdim=True)                 # base-table "ruler" over these T
    R["ruler_cos_mean"] = round(F.cosine_similarity(emb[Tid], c, dim=-1).mean().item(), 4)
    for view, ctr in (("raw", None), ("demean", c)):
        R[f"first_{view}"] = round(cos_ids(emb, Tid, Aid, ctr).mean().item(), 4)   # ① cos(T, first piece A)
        R[f"last_{view}"]  = round(cos_ids(emb, Tid, Bid, ctr).mean().item(), 4)   # ① cos(T, last piece B)
    for view, ctr in (("raw", None), ("demean", c)):
        df = (1 - torch.stack([cos_ids(emb, torch.tensor([p[0]]), torch.tensor([p[1]]), ctr)[0]
                               for p in pairs_share_last])).mean().item()
        dl = (1 - torch.stack([cos_ids(emb, torch.tensor([p[0]]), torch.tensor([p[1]]), ctr)[0]
                               for p in pairs_share_first])).mean().item()
        R[f"minpair_{view}"] = {"first": round(df, 4), "last": round(dl, 4),
                                "ratio": round(df / max(dl, 1e-9), 2)}
    return R


res = {"n_triples": len(triples)}
for name, emb in TABLES.items():
    res[name] = analyze(emb)
json.dump(res, open(f"{HERE}/results/base.json", "w"), indent=2)

# hyper-token K=2 comparison (v0.6.4)
hyper = json.load(open(f"{HERE}/results/v064.json")) if os.path.exists(f"{HERE}/results/v064.json") else None


def md():
    o = ["# Base-token BPE control (K=2) — is first-piece dominance in the embedding table itself?", "",
         "A BPE base token `T = A + B` (unique immediate merge). `A`/`B` are real vocab tokens with real "
         "embeddings — same 2-piece composition as a K=2 hyper-token, but the base embedding is a **free "
         "lookup**, not built by an encoder. Same cosine probes on the **raw embedding tables** "
         f"(from v0.6.4; base tables ~frozen). {len(triples)} mergeable base tokens. "
         "K=2 only (the binary split is unique; K=3/4 deferred).", "",
         "## ① Is a base token more like its FIRST piece (A) or LAST piece (B)?", "",
         "Mean cosine of `emb[T]` with each piece. raw and mean-removed.", "",
         "| table | cos(T, first A) raw | cos(T, last B) raw | first−last | (demean) first | last |",
         "|---|--:|--:|--:|--:|--:|"]
    for name in TABLES:
        a = res[name]
        o.append(f"| {name} | {a['first_raw']} | {a['last_raw']} | {a['first_raw']-a['last_raw']:+.3f} | "
                 f"{a['first_demean']} | {a['last_demean']} |")
    o += ["",
          "## ② Base-table anisotropy (the 'ruler' analog)", "",
          "| table | cos-with-mean of T |", "|---|--:|"]
    for name in TABLES:
        o.append(f"| {name} | {res[name]['ruler_cos_mean']} |")
    o += ["",
          "## ④ Minimal pairs: change FIRST piece vs LAST piece (cosine distance)", "",
          "differ-first = tokens sharing the last piece B, different A (real vocab pairs). differ-last = "
          "sharing A, different B. first/last > 1 = first piece matters more.", "",
          "| table | variant | differ-first | differ-last | first/last |", "|---|---|--:|--:|--:|"]
    for name in TABLES:
        for view in ("raw", "demean"):
            m = res[name][f"minpair_{view}"]
            o.append(f"| {name} | {view} | {m['first']} | {m['last']} | **{m['ratio']}×** |")
    if hyper:
        o += ["",
              "## Side-by-side with v0.6.4 **hyper-tokens** (K=2)", "",
              "Does the hyper-encoder show more first-piece dominance than the raw base table?", "",
              "| quantity | base `lm_head` | v0.6.4 hyper **output** | base `tok_emb` | v0.6.4 hyper **input** |",
              "|---|--:|--:|--:|--:|"]
        blm, bte = res["lm_head (output space)"], res["tok_emb (input space)"]
        ho, hi = hyper["output"], hyper["input"]
        o.append(f"| ① cos(·, first) raw | {blm['first_raw']} | {ho['raw_K2'][0]} | {bte['first_raw']} | {hi['raw_K2'][0]} |")
        o.append(f"| ① cos(·, last) raw | {blm['last_raw']} | {ho['raw_K2'][1]} | {bte['last_raw']} | {hi['raw_K2'][1]} |")
        o.append(f"| ④ minpair first/last (raw) | {blm['minpair_raw']['ratio']}× | {ho['minpair']['2']['raw']['ratio']}× | "
                 f"{bte['minpair_raw']['ratio']}× | {hi['minpair']['2']['raw']['ratio']}× |")
        o.append(f"| ② ruler / anisotropy | {blm['ruler_cos_mean']} | {ho['ruler_cos_mean']} | "
                 f"{bte['ruler_cos_mean']} | {hi['ruler_cos_mean']} |")
    o += ["", "---", "",
          "*Repro: `uv run python probe_base.py`. Data: `results/base.json`. Method: see `README.md`.*"]
    return "\n".join(o)

open(f"{HERE}/reports/base.md", "w").write(md())
print("wrote results/base.json and reports/base.md")
for name in TABLES:
    a = res[name]
    print(f"  {name}: cos(T,first)={a['first_raw']} cos(T,last)={a['last_raw']}  "
          f"minpair first/last raw={a['minpair_raw']['ratio']}x  ruler={a['ruler_cos_mean']}")
