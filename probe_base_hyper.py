"""Controlled twin of the base-token probe: SAME objects, but embedded by the
HYPER-ENCODER instead of the raw lookup table.

`probe_base.py` takes each base token `T = [p1..pK]` (its unique BPE
decomposition) and its real shared-(K-1)-pieces minimal pairs, and reads the raw
lookup embedding `emb[T]`. This script reuses the EXACT same objects and pairs,
but throws away `emb[T]` and instead feeds the K pieces `[p1..pK]` through the
hyper-encoder — i.e. pretends `T` is a hyper-token that merges those pieces and
asks what vector the encoder assigns. So the analyzed set is identical to the
base control; the ONLY variable is raw-lookup vs encoder. It also replaces the
main hyper probe's *random*-replacement minimal pairs with these *real*
shared-piece pairs, matching the base control apples-to-apples.

Runs both checkpoints (v0.6.4 residual-on, vx0.6.4.2 residual-off). Writes one
report per checkpoint plus a combined three-way summary (base raw vs both).

Caveat: BPE sub-word pieces (e.g. `ational`) are not the whole-token LZW merges
the encoder trained on — this is an out-of-distribution probe, read as "the
vector the encoder *would* assign such a merge", not a claim about real codebook
entries. Does NOT touch probe.py / probe_base.py or their outputs.
"""
import os, json
import torch, torch.nn.functional as F
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
import enc_lib
from transformers import AutoTokenizer
os.environ.setdefault("HF_HOME", "/dlabscratch1/gentilin/.cache/huggingface")
KS = (2, 3, 4)
W = f"{HERE}/weights"
CKPTS = [("v064", "v0.6.4 (residual)", f"{W}/encoders_v064.pt", True),
         ("vx0642", "vx0.6.4.2 (no-residual)", f"{W}/encoders_vx0642.pt", False)]

tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
vocab = tok.get_vocab()
SPECIAL = set(tok.all_special_ids)
merges = json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]
RANK = {}
for i, e in enumerate(merges):
    A, B = e if isinstance(e, list) else e.split(" ")
    RANK[(A, B)] = i


def bpe_states(s):
    """Greedy BPE reduction: {k: k-piece tuple} for every k, or None if unreachable."""
    seq = list(s)
    states = {len(seq): tuple(seq)}
    while len(seq) > 1:
        best, bi = None, -1
        for i in range(len(seq) - 1):
            r = RANK.get((seq[i], seq[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            return None
        seq[bi:bi + 2] = [seq[bi] + seq[bi + 1]]
        states[len(seq)] = tuple(seq)
    return states


# Same decomposition + minimal-pair pools as probe_base.py (self-contained copy).
decomp = {K: [] for K in KS}
for T, ti in vocab.items():
    if ti in SPECIAL:
        continue
    st = bpe_states(T)
    if st is None:
        continue
    for K in KS:
        pieces = st.get(K)
        if pieces is None:
            continue
        if all(p in vocab and vocab[p] not in SPECIAL for p in pieces):
            decomp[K].append((ti,) + tuple(vocab[p] for p in pieces))
for K in KS:
    print(f"K={K}: {len(decomp[K])} base tokens", flush=True)


def all_disjoint_pairs(groups):
    out = []
    for v in groups.values():
        for i in range(0, len(v) - 1, 2):
            out.append((v[i], v[i + 1]))
    return out


# Per-K: piece-id matrix (N,K) to encode, T-id -> row, and minimal-pair row indices.
DEC = {}
for K in KS:
    ents = decomp[K]
    pieces = torch.tensor([e[1:] for e in ents])            # (N, K) piece ids
    row_of = {e[0]: r for r, e in enumerate(ents)}          # T_id -> row in `pieces`
    by_suffix = defaultdict(list)   # change-first: share p2..pK
    by_prefix = defaultdict(list)   # change-last:  share p1..p(K-1)
    for e in ents:
        by_suffix[e[2:]].append(e[0])
        by_prefix[e[1:-1]].append(e[0])

    def rows(pairs):
        return (torch.tensor([row_of[a] for a, _ in pairs]),
                torch.tensor([row_of[b] for _, b in pairs]))
    cf_a, cf_b = rows(all_disjoint_pairs(by_suffix))        # change first
    cl_a, cl_b = rows(all_disjoint_pairs(by_prefix))        # change last
    DEC[K] = {"pieces": pieces, "cf_a": cf_a, "cf_b": cf_b, "cl_a": cl_a, "cl_b": cl_b}
    print(f"K={K}: change-first pairs={cf_a.numel()}  change-last pairs={cl_a.numel()}", flush=True)


def encode_all(enc, pieces, residual, chunk=4096):
    """Run the hyper-encoder on every K-piece row. Returns (N, dim)."""
    outs = []
    for s in range(0, pieces.shape[0], chunk):
        ids = pieces[s:s + chunk]
        m = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool)
        with torch.no_grad():
            outs.append(enc.encode(ids, m, residual=residual))
    return torch.cat(outs)


def analyze(enc, residual):
    R = {}
    for K in KS:
        d = DEC[K]; pieces = d["pieces"]
        E = encode_all(enc, pieces, residual)               # (N, dim) encoder vector per T
        emb = enc.emb
        c = E.mean(0, keepdim=True)                          # ruler over this K's encoded set
        rk = {"n": E.shape[0],
              "ruler_cos_mean": round(F.cosine_similarity(E, c, dim=-1).mean().item(), 4)}
        # ① per-piece cosine (ruler removed from the composed vector only, as in the hyper probe)
        for view, sub in (("raw", False), ("demean", True)):
            base = (E - c) if sub else E
            rk[f"perpiece_{view}"] = [round(F.cosine_similarity(base, emb[pieces[:, i]], dim=-1).mean().item(), 4)
                                      for i in range(K)]
        # ④ minimal pairs on the SAME pairs as base_probe, through the encoder
        for view, sub in (("raw", False), ("demean", True)):
            def dist(a, b):
                ea, eb = E[a], E[b]
                if sub:
                    ea, eb = ea - c, eb - c
                return (1 - F.cosine_similarity(ea, eb, dim=-1)).mean().item()
            df = dist(d["cf_a"], d["cf_b"])
            dl = dist(d["cl_a"], d["cl_b"])
            rk[f"minpair_{view}"] = {"first": round(df, 4), "last": round(dl, 4),
                                     "ratio": round(df / max(dl, 1e-9), 2)}
        R[str(K)] = rk
    return R


ROLES = [("output", "lm_head (output space)"), ("input", "tok_emb (input space)")]

RES = {}
for tag, label, path, resid in CKPTS:
    print(f"\n== {label} (residual={resid}) ==", flush=True)
    ein, eout = enc_lib.load_pair(path)
    RES[tag] = {"label": label, "residual": resid,
                "output": analyze(eout, resid), "input": analyze(ein, resid)}
    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(RES[tag], open(f"{HERE}/results/base_hyper_{tag}.json", "w"), indent=2)
    for role, _ in ROLES:
        for K in KS:
            a = RES[tag][role][str(K)]
            print(f"  {role} K{K}: perpiece_raw={a['perpiece_raw']} minpair raw={a['minpair_raw']['ratio']}x "
                  f"ruler={a['ruler_cos_mean']}", flush=True)


# ---------- per-checkpoint report ----------
def md_single(tag):
    r = RES[tag]; label = r["label"]; resid = r["residual"]
    o = [f"# Base-token pieces through the {label} hyper-encoder", "",
         f"Same objects as the base-token control (`probe_base.py`) — every base token `T`'s unique BPE "
         f"decomposition `[p1..pK]` and the same real shared-(K−1)-pieces minimal pairs — but the vector is "
         f"**`E = hyper_encoder([p1..pK])`** (residual **{'on' if resid else 'off'}**), not the raw lookup "
         f"`emb[T]`. So the only change vs `base.md` is raw-lookup → encoder. Bit-exact encoder "
         f"(max|Δ|=0 vs `model.py`). OOD caveat: BPE pieces are sub-word, not whole-token merges.", "",
         "## ① Per-piece cosine `cos(E, emb[p_i])`", ""]
    for role, name in ROLES:
        for view in ("raw", "demean"):
            o += [f"**{name}** — {view}", "",
                  "| K | pos1 (first) | pos2 | pos3 | pos4 |", "|---|--:|--:|--:|--:|"]
            for K in KS:
                pp = r[role][str(K)][f"perpiece_{view}"]
                cells = "".join(f" {v:+.3f} |" for v in pp) + " — |" * (4 - K)
                o.append(f"| {K} |{cells}")
            o.append("")
    o += ["## ② Ruler (cos-with-mean of the encoded set)", "",
          "| table | K2 | K3 | K4 |", "|---|--:|--:|--:|"]
    for role, name in ROLES:
        o.append(f"| {name} | " + " | ".join(f"{r[role][str(K)]['ruler_cos_mean']}" for K in KS) + " |")
    o += ["",
          "## ④ Minimal pairs: change FIRST vs LAST piece (cosine distance)", "",
          "Same pairs as `base.md`, but distance is between encoder vectors. `first/last > 1` = "
          "prefix-dominated.", ""]
    for view, variant in (("raw", "④a raw"), ("demean", "④b after ruler removal")):
        o += [f"**{variant}**", "",
              "| table | K2 first/last | K3 first/last | K4 first/last | K2 ratio | K3 ratio | K4 ratio |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for role, name in ROLES:
            m = {K: r[role][str(K)][f"minpair_{view}"] for K in KS}
            fl = " | ".join(f"{m[K]['first']:.2f}/{m[K]['last']:.2f}" for K in KS)
            rt = " | ".join(f"**{m[K]['ratio']}×**" for K in KS)
            o.append(f"| {name} | {fl} | {rt} |")
        o.append("")
    o += ["---", "", f"*Repro: `uv run python probe_base_hyper.py`. Data: `results/base_hyper_{tag}.json`.*"]
    return "\n".join(o)


for tag, *_ in CKPTS:
    open(f"{HERE}/reports/base_hyper_{tag}.md", "w").write(md_single(tag))


# ---------- combined three-way summary ----------
def md_combined():
    base = json.load(open(f"{HERE}/results/base.json")) if os.path.exists(f"{HERE}/results/base.json") else None
    LM, TE = "lm_head (output space)", "tok_emb (input space)"
    tags = [t for t, *_ in CKPTS]
    o = ["# Controlled comparison — identical objects, raw lookup vs hyper-encoder", "",
         "Every column analyzes the **same** base tokens and the **same** minimal pairs (from "
         "`probe_base.py`'s BPE decompositions). The only difference is how each token is embedded: the "
         "**base** columns use the raw lookup `emb[T]`; the encoder columns feed the same pieces `[p1..pK]` "
         "through the hyper-encoder. Unlike `base.md`'s side-by-side (which used corpus n-grams for the "
         "hyper side), here the hyper side is the identical object set — so this isolates raw-lookup vs "
         "encoder with data held fixed, and uses real shared-piece pairs (no random replacement). "
         "**Bold = base control** (raw table, no encoder).", ""]
    head = ("| K | **base `lm_head`** | " + " | ".join(f"{RES[t]['label'].split(' ')[0]} out" for t in tags) +
            " | **base `tok_emb`** | " + " | ".join(f"{RES[t]['label'].split(' ')[0]} in" for t in tags) + " |")
    align = "|:--|" + "--:|" * (2 + 2 * len(tags))

    def block(title, base_val, enc_val, note=""):
        rows = [title + (f" {note}" if note else ""), "", head, align]
        for K in KS:
            r = [f"**{base_val(LM, K)}**"] + [enc_val(t, "output", K) for t in tags]
            r += [f"**{base_val(TE, K)}**"] + [enc_val(t, "input", K) for t in tags]
            rows.append(f"| {K} | " + " | ".join(r) + " |")
        rows.append("")
        return rows

    if base:
        o += block("**④ minpair first/last ratio (prefix-dominance), per K**",
                   lambda name, K: f"{base[name][str(K)]['minpair_raw']['ratio']}×",
                   lambda t, role, K: f"{RES[t][role][str(K)]['minpair_raw']['ratio']}×")
        o += block("**① first-piece cosine (pos1), per K** — raw",
                   lambda name, K: f"{base[name][str(K)]['perpiece_raw'][0]}",
                   lambda t, role, K: f"{RES[t][role][str(K)]['perpiece_raw'][0]}",
                   note="(hyper output raw ≈0 when a shared ruler dominates — read with ④)")
    else:
        o += ["_(run `probe_base.py` first to populate the base-lookup columns.)_", ""]
    o += ["---", "", "*Repro: `uv run python probe_base_hyper.py` (after `probe_base.py`). "
          "Data: `results/base_hyper_{v064,vx0642}.json`, `results/base.json`.*"]
    return "\n".join(o)


open(f"{HERE}/reports/base_hyper.md", "w").write(md_combined())
print("\nwrote reports/base_hyper_v064.md, reports/base_hyper_vx0642.md, reports/base_hyper.md")
