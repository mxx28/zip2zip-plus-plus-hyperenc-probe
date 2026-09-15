"""Base-token BPE control (K=2/3/4) for the hyper-token findings.

A BPE base token T was built by the tokenizer from smaller pieces. Greedy BPE
(merge the lowest-rank adjacent pair until one token remains) passes through
exactly one state with K tokens, so every base token has a UNIQUE ordered
K-piece decomposition T = [p1, ..., pK] whose pieces are themselves real vocab
tokens with real embeddings. That is the SAME "ordered composition of K pieces"
structure as a K-merge hyper-token — but the base embedding is a FREE lookup
parameter, not built by an encoder. Running the same cosine probes on the RAW
embedding tables tells us whether "first-piece dominance" is a property of the
embedding space itself (BPE-induced) or something the hyper-encoder imposes.

Probes (on tok_embeddings and lm_head, raw + ruler-removed), per K in {2,3,4}:
  ①  per-piece cosine:  cos(emb[T], emb[p_i]) for each position i
  ④  minimal pairs:  differ-first (share pieces p2..pK, change p1)
                     differ-last  (share pieces p1..p(K-1), change pK)
  ②  ruler/anisotropy of the base table (context)

Demeaning mirrors the hyper probe: for ① the ruler c is subtracted from the
COMPOSED token emb[T] only (not the reference piece); for ④ from both members
of the pair (both are composed tokens). Minimal pairs use ALL disjoint
same-(K-1)-pieces vocab pairs — fully deterministic, no sampling.

Compares side-by-side with the v0.6.4 hyper-token numbers per K
(`results/v064.json`).
"""
import sys, os, json
import torch, torch.nn.functional as F
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
from transformers import AutoTokenizer
KS = (2, 3, 4)

# ---- presets: name -> (tokenizer, weights_file, hyper results to compare against) ----
# The base tables live in the checkpoint's extracted weights, so the tokenizer
# and the weights file must come from the SAME model: a decomposition computed
# with one tokenizer indexes meaningless rows of the other's table.
PRESETS = {
    "base":         ("microsoft/Phi-3.5-mini-instruct", "encoders_v064.pt",
                     {"v0.6.4": "v064"}),
    "base_llama3B": ("meta-llama/Llama-3.2-3B-Instruct", "encoders_llama3B_v064_untied.pt",
                     {"untied HE": "llama3B_v064_untied", "tied HE": "llama3B_v064_tied"}),
}
RUN = sys.argv[1] if len(sys.argv) > 1 else "base"
TOK_NAME, WEIGHTS_FILE, HYP_TAGS = PRESETS[RUN]

tok = AutoTokenizer.from_pretrained(TOK_NAME)
vocab = tok.get_vocab()                     # str -> id
SPECIAL = set(tok.all_special_ids)
merges = json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]

# rank = merge priority (lower = applied earlier)
RANK = {}
for i, e in enumerate(merges):
    A, B = e if isinstance(e, list) else e.split(" ")
    RANK[(A, B)] = i


def bpe_states(s):
    """Greedy BPE reduction of string s. Return {k: (piece strings) with k tokens}
    for every k from len(chars) down to 1, or None if s is not fully reducible."""
    seq = list(s)                           # atomic units (▁ is one char)
    states = {len(seq): tuple(seq)}
    while len(seq) > 1:
        best, bi = None, -1
        for i in range(len(seq) - 1):
            r = RANK.get((seq[i], seq[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            return None                     # no applicable merge -> unreachable
        seq[bi:bi + 2] = [seq[bi] + seq[bi + 1]]
        states[len(seq)] = tuple(seq)
    return states                           # includes key 1 (fully reduced to T)


# For each K: unique ordered K-piece decomposition (all pieces real, non-special).
# decomp[K] = list of (T_id, p1_id, ..., pK_id).
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
    print(f"K={K}: {len(decomp[K])} base tokens with a valid all-vocab {K}-piece split", flush=True)


def all_disjoint_pairs(groups):
    """All disjoint adjacent (v[0],v[1]),(v[2],v[3]),... pairs across groups of
    size >= 2. Deterministic (sorted group keys and token ids); every token used once."""
    out = []
    for key in sorted(groups):
        v = sorted(groups[key])
        for i in range(0, len(v) - 1, 2):
            out.append((v[i], v[i + 1]))
    return out


# Per-K minimal-pair pools + tensors.
DEC = {}
for K in KS:
    ents = decomp[K]
    T = torch.tensor([e[0] for e in ents])
    P = [torch.tensor([e[1 + i] for e in ents]) for i in range(K)]   # per-position piece ids
    by_suffix = defaultdict(list)   # key p2..pK  -> Ts sharing suffix (differ in FIRST)
    by_prefix = defaultdict(list)   # key p1..p(K-1) -> Ts sharing prefix (differ in LAST)
    for e in ents:
        pcs = e[1:]
        by_suffix[pcs[1:]].append(e[0])
        by_prefix[pcs[:-1]].append(e[0])
    df = all_disjoint_pairs(by_suffix)   # change-first pairs
    dl = all_disjoint_pairs(by_prefix)   # change-last  pairs
    DEC[K] = {
        "T": T, "P": P,
        "df_a": torch.tensor([p[0] for p in df]), "df_b": torch.tensor([p[1] for p in df]),
        "dl_a": torch.tensor([p[0] for p in dl]), "dl_b": torch.tensor([p[1] for p in dl]),
    }
    print(f"K={K}: change-first pairs={len(df)}  change-last pairs={len(dl)}", flush=True)

# embedding tables from the v0.6.4 checkpoint (base tables are ~frozen across runs)
sd = torch.load(f"{HERE}/weights/{WEIGHTS_FILE}", map_location="cpu", weights_only=True)
TABLES = {"lm_head (output space)": sd["output.weight"],
          "tok_emb (input space)": sd["tok_embeddings.weight"]}


def analyze(emb):
    R = {}
    for K in KS:
        d = DEC[K]; T = d["T"]; P = d["P"]
        c = emb[T].mean(0, keepdim=True)                        # base-table ruler over this K's T
        rk = {"n": int(T.numel()),
              "ruler_cos_mean": round(F.cosine_similarity(emb[T], c, dim=-1).mean().item(), 4)}
        # ① per-piece cosine: raw + demean (ruler subtracted from the COMPOSED token only,
        # mirroring the hyper probe) so the two reports line up position-for-position.
        for view, sub in (("raw", False), ("demean", True)):
            base = (emb[T] - c) if sub else emb[T]
            rk[f"perpiece_{view}"] = [round(F.cosine_similarity(base, emb[P[i]], dim=-1).mean().item(), 4)
                                      for i in range(K)]
        # ④ substitution probe: raw + demean (ruler subtracted from both members, as in the hyper probe).
        for view, sub in (("raw", False), ("demean", True)):
            def similarity(a, b):
                ea, eb = emb[a], emb[b]
                if sub:
                    ea, eb = ea - c, eb - c
                return F.cosine_similarity(ea, eb, dim=-1).mean().item()
            first = similarity(d["df_a"], d["df_b"])
            last = similarity(d["dl_a"], d["dl_b"])
            if abs(first) < 1e-12:
                raise ValueError("prefix-substitution cosine is zero; ratio is undefined")
            rk[f"minpair_{view}"] = {
                "first": round(first, 4),
                "last": round(last, 4),
                "ratio": round(last / first, 2),
            }
        R[str(K)] = rk
    return R


res = {"n_decomp": {str(K): len(decomp[K]) for K in KS}}
for name, emb in TABLES.items():
    res[name] = analyze(emb)

# ---- nested growing-prefix examples: raw-lookup analog of the hyper ⑤ ----
# Real vocab tokens that are growing string-prefixes (a ⊂ b ⊂ c, mapped to H2⊂H3⊂H4).
# Does the RAW table keep growing-prefix words alike, the way the encoder keeps growing merges alike?
CHAINS = [("▁under", "▁understand", "▁understanding"),
          ("▁inter", "▁intern", "▁international"),
          ("▁care", "▁careful", "▁carefully"),
          ("▁success", "▁successful", "▁successfully")]
nested_examples = []
for a, b, cc in CHAINS:
    if not all(x in vocab for x in (a, b, cc)):
        continue
    ids = torch.tensor([vocab[a], vocab[b], vocab[cc]])
    row = {"chain": [a, b, cc], "cos": {}}
    for name, emb in TABLES.items():
        e = emb[ids]
        row["cos"][name] = {"H2_H3": round(F.cosine_similarity(e[0:1], e[1:2], dim=-1).item(), 3),
                            "H2_H4": round(F.cosine_similarity(e[0:1], e[2:3], dim=-1).item(), 3)}
    nested_examples.append(row)
res["nested_examples"] = nested_examples
os.makedirs(f"{HERE}/results", exist_ok=True)
json.dump(res, open(f"{HERE}/results/{RUN}.json", "w"), indent=2)

# hyper-token comparison (per K)
def load(tag):
    p = f"{HERE}/results/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None
HYP = {label: load(tag) for label, tag in HYP_TAGS.items()}
HYP = {k: v for k, v in HYP.items() if v is not None}

LM = "lm_head (output space)"
TE = "tok_emb (input space)"


def md():
    counts = ", ".join(f"K{K}: {len(decomp[K])}" for K in KS)
    o = ["# Base-token BPE control (K=2/3/4) — is first-piece dominance in the embedding table itself?", "",
         "Every base token has a UNIQUE ordered K-piece BPE decomposition `T = [p1..pK]` whose pieces are "
         "real vocab tokens with real embeddings — the same ordered-composition structure as a K-merge "
         "hyper-token, but the base embedding is a **free lookup**, not built by an encoder. Same cosine "
         "probes on the **raw embedding tables** (from v0.6.4; base tables ~frozen). Mergeable base tokens "
         f"per K: {counts}. Minimal pairs use **all** disjoint same-(K−1)-pieces vocab pairs (deterministic).", "",
         "## ① Is a base token more like its FIRST piece or a later piece?", "",
         "Mean `cos(emb[T], emb[p_i])` per position. Both **raw** and **demean** (ruler removed from "
         "`emb[T]` only, as in the hyper probe) so this lines up with the hyper report — pick whichever "
         "you compare against. The base ruler is weak, so raw ≈ demean.", ""]
    for name in TABLES:
        for view in ("raw", "demean"):
            o += [f"**{name}** — {view} per-piece cosine", "",
                  "| K | pos1 (first) | pos2 | pos3 | pos4 |", "|---|--:|--:|--:|--:|"]
            for K in KS:
                pp = res[name][str(K)][f"perpiece_{view}"]
                cells = "".join(f" {v:+.3f} |" for v in pp) + " — |" * (4 - K)
                o.append(f"| {K} |{cells}")
            o.append("")
    if res.get("nested_examples"):
        o += ["### Growth probe: growing-prefix examples (raw-lookup analog of hyper ⑤)", "",
              "Real vocab tokens that are growing string-prefixes `a ⊂ b ⊂ c` (mapped to H2⊂H3⊂H4), using "
              "their **raw lookup embeddings** — no encoder. High cos = growing-prefix words stay alike in "
              "the table itself. Compare against the hyper ⑤ (where the *encoder* is what keeps growing "
              "merges alike). **Illustrative examples, not statistics.**", ""]
        for name in TABLES:
            o += [f"**{name}**", "",
                  "| chain (H2 ⊂ H3 ⊂ H4) | cos(H2,H3) | cos(H2,H4) |", "|---|--:|--:|"]
            for ex in res["nested_examples"]:
                chain = " ⊂ ".join(x.replace("▁", "") for x in ex["chain"])
                cc = ex["cos"][name]
                o.append(f"| {chain} | {cc['H2_H3']} | {cc['H2_H4']} |")
            o.append("")
    o += ["## ② Base-table anisotropy (the 'ruler' analog, context only)", "",
          "| table | K2 | K3 | K4 |", "|---|--:|--:|--:|"]
    for name in TABLES:
        vals = " | ".join(f"{res[name][str(K)]['ruler_cos_mean']}" for K in KS)
        o.append(f"| {name} | {vals} |")
    o += ["",
          "## ④ Substitution probe: replace first vs last piece (cosine similarity)", "",
          "change-first = pairs sharing pieces p2..pK, differing in p1. change-last = sharing p1..p(K−1), "
          "differing in pK. Cells are `cos_first/cos_last` similarities. The ratio is "
          "cos_last / cos_first: > 1 means prefix-aligned, < 1 means suffix-aligned, and "
          "approximately 1 means balanced. Same layout as the hyper-token ④ tables.", ""]
    for view, variant in (("raw", "④a raw"), ("demean", "④b after ruler removal")):
        o += [f"**{variant}**", "",
              "| table | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for name in TABLES:
            m = {K: res[name][str(K)][f"minpair_{view}"] for K in KS}
            fl = " | ".join(f"{m[K]['first']:.2f}/{m[K]['last']:.2f}" for K in KS)
            rt = " | ".join(f"**{m[K]['ratio']}×**" for K in KS)
            o.append(f"| {name} | {fl} | {rt} |")
        o.append("")
    if HYP:
        tags = list(HYP)
        # header: base columns bolded (the control); K left-aligned, all numeric right-aligned.
        head = ("| K | **base `lm_head`** | " + " | ".join(f"{t} out" for t in tags) +
                " | **base `tok_emb`** | " + " | ".join(f"{t} in" for t in tags) + " |")
        align = "|:--|" + "--:|" * (2 + 2 * len(tags))

        def sxs_rows(cell):
            """cell(space, K) -> string; base cells (space in {out_base,in_base}) get bolded."""
            rows = []
            for K in KS:
                r = [f"**{cell('out_base', K)}**"]
                r += [cell(('out', t), K) for t in tags]
                r += [f"**{cell('in_base', K)}**"]
                r += [cell(('in', t), K) for t in tags]
                rows.append(f"| {K} | " + " | ".join(r) + " |")
            return rows

        o += ["## Side-by-side with hyper-tokens — the money comparison", "",
              "Does the hyper-encoder impose more first-piece dominance than the raw embedding table shows? "
              "Paper comparisons use the raw cosine similarities only; ruler-removed diagnostics stay "
              "available above but are not used in the figure. **Bold = base control** (raw table, "
              "no encoder) — the reference each encoder column is read against.", "",
              "**④ substitution cos_last/cos_first ratio, per K**", "", head, align]

        def ratio(space, K):
            if space == "out_base": return f"{res[LM][str(K)]['minpair_raw']['ratio']}×"
            if space == "in_base":  return f"{res[TE][str(K)]['minpair_raw']['ratio']}×"
            side, t = space
            role = "output" if side == "out" else "input"
            return f"{HYP[t][role]['minpair'][str(K)]['raw']['ratio']}×"
        o += sxs_rows(ratio)

        o += ["",
              "**① first-piece cosine (pos1), per K** — raw. "
              "(Note: a strong shared ruler makes the *raw* hyper cosine ≈ its ruler value, not 0; "
              "read alongside ④.)", "", head, align]

        def pp(space, K):
            if space == "out_base": return f"{res[LM][str(K)]['perpiece_raw'][0]}"
            if space == "in_base":  return f"{res[TE][str(K)]['perpiece_raw'][0]}"
            side, t = space
            role = "output" if side == "out" else "input"
            return f"{HYP[t][role][f'raw_K{K}'][0]}"
        o += sxs_rows(pp)
        o.append("")
    o += ["---", "",
          f"*Repro: `python probe_base.py {RUN}`. Data: `results/{RUN}.json`. Method: see `docs/substitution_probe.md`.*"]
    return "\n".join(o)


os.makedirs(f"{HERE}/reports", exist_ok=True)
open(f"{HERE}/reports/{RUN}.md", "w").write(md())
print(f"wrote results/{RUN}.json and reports/{RUN}.md")
for name in TABLES:
    for K in KS:
        a = res[name][str(K)]
        print(f"  {name} K{K}: perpiece_raw={a['perpiece_raw']}  "
              f"cos_last/cos_first raw={a['minpair_raw']['ratio']}x  ruler={a['ruler_cos_mean']}")
