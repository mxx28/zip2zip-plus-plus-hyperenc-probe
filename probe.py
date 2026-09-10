"""Hyper-encoder embedding probe — analyze ONE zip2zip checkpoint.

For the input encoder (reads tok_embeddings) and the output encoder (reads
lm_head) of a single checkpoint, on hyper-tokens sampled from a standard corpus
(K=2/3/4):
  ①  raw      per-position cosine of the final vector with each base token
  ②  ruler    shared-vector stats (cos-with-mean, pairwise, energy) + controls
  ③  demean   per-position cosine after subtracting the shared ruler
  ④  substitution  substitution probe: change first vs last token (cosine similarity)
  ⑤  growth   growth probe: growing-prefix examples H2⊂H3⊂H4 similarity (illustrative)

Hyper-tokens (objects analyzed) — see the "Data & objects" section of the
report for exact counts:
  Corpus: WikiText-2-raw-v1 (public). Take every consecutive base-token K-gram,
  drop windows with a special/digit token, dedup, keep the top-N by corpus
  frequency per K. Frequency ONLY bounds the sample for tractability +
  reproducibility; it is not a correctness claim. Justification for using
  corpus n-grams rather than the model's real codebook merges: after training
  the hyper-encoder is FIXED, so E(H) is a deterministic function of the K
  base-token ids alone — any valid K-tuple probes the same learned geometry.

The FINAL vector is the one the model actually uses:
  residual=True  -> E = base_vec[t1] + encoder_out   (v0.5, v0.6.4, ...)
  residual=False -> E = encoder_out                  (no_encoder_residual runs such as v0.52)

Outputs: results/<name>.json  and  reports/<name>.md.
Usage: `uv run python probe.py <preset>`   (one checkpoint per run; compare reports yourself)
"""
import sys, re, os, json
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enc_lib
from transformers import AutoTokenizer
os.environ.setdefault("HF_HOME", "/dlabscratch1/gentilin/.cache/huggingface")
# datasets cache must be writable by us (the model cache above is read-only shared)
os.environ.setdefault("HF_DATASETS_CACHE", "/dlabscratch1/xinma/.cache/huggingface/datasets")
rng = np.random.default_rng(0)
W = f"{HERE}/weights"
N_TOP = 5000        # per-K cap: keep the N most frequent unique K-grams
N_RESAMPLE = 6      # ④ random-replacement resamples
N_EXAMPLES = 4      # ⑤ illustrative nested chains

# ---- presets: name -> (label, weights_file, residual_bool) ----
# residual=False for no_encoder_residual runs such as v0.52.
# The base model's tokenizer + vocab bound travel with the preset: the corpus
# n-grams must be tokenized by the SAME tokenizer whose embedding table the
# encoder reads (a Phi-tokenized n-gram is meaningless ids in Llama space).
PHI = ("microsoft/Phi-3.5-mini-instruct", 32064)
LLAMA3 = ("meta-llama/Llama-3.2-3B-Instruct", 128256)
PRESETS = {
    "v05":   ("v0.5 (untied, residual)",       f"{W}/encoders_v05.pt",   True, *PHI),
    "v052":  ("v0.52 (untied, no-residual)",   f"{W}/encoders_v052.pt",  False, *PHI),
    "v064":  ("v0.6.4 (untied, residual)",     f"{W}/encoders_v064.pt",  True, *PHI),
    # Llama-3.2-3B-Instruct v0.6.4 pair (Andrea, 2026-09). Same recipe, single
    # variable = --untied_hyper_encoder. The base model TIES e_in and e_out, so
    # in the tied-HE run the input and output roles are literally one function
    # (its weights file aliases hyper_output to hyper_encoder, so the "output"
    # half of the report is the same encoder read through output.weight).
    "llama3B_v064_untied": ("Llama-3.2-3B v0.6.4 (untied HE, residual)",
                            f"{W}/encoders_llama3B_v064_untied.pt", True, *LLAMA3),
    "llama3B_v064_tied": ("Llama-3.2-3B v0.6.4 (tied HE, residual)",
                          f"{W}/encoders_llama3B_v064_tied.pt", True, *LLAMA3),
}
RUN = sys.argv[1] if len(sys.argv) > 1 else "v064"
LABEL, PATH, RESID, TOK_NAME, VOCAB = PRESETS[RUN]

# ---- build hyper-tokens from WikiText-2-raw-v1 ----
tok = AutoTokenizer.from_pretrained(TOK_NAME)
DIS = set(tok.all_special_ids) | {tid for ts, tid in tok.get_vocab().items() if re.search(r"[0-9]", ts)}


def keep(t):
    return t < VOCAB and t not in DIS


def load_corpus():
    from datasets import load_dataset
    # Salesforce/wikitext is the canonical home; fall back to the locally-cached,
    # same-content document-level packaging when it can't be fetched offline.
    for name, col, tag in [("Salesforce/wikitext", "text", "Salesforce/wikitext"),
                           ("EleutherAI/wikitext_document_level", "page",
                            "EleutherAI/wikitext_document_level (same WikiText-2-raw text)")]:
        try:
            ds = load_dataset(name, "wikitext-2-raw-v1", split="train")
            c = col if col in ds.column_names else ds.column_names[0]
            return [d for d in ds[c] if d and d.strip()], f"{tag} · wikitext-2-raw-v1 [train]"
        except Exception as e:
            print(f"  {name} unavailable: {str(e)[:80]}", flush=True)
    raise RuntimeError("no WikiText source available")


docs, CORPUS = load_corpus()
CNT = {k: Counter() for k in (2, 3, 4)}
pool = set()
n_tokens = 0
for d in docs:
    ids = tok(d, add_special_tokens=False)["input_ids"]
    n_tokens += len(ids)
    for t in ids:
        if keep(t):
            pool.add(t)
    for k in (2, 3, 4):
        for i in range(len(ids) - k + 1):
            w = ids[i:i + k]
            if all(keep(t) for t in w):
                CNT[k][tuple(w)] += 1
POOL = sorted(pool)
NAT = {k: [list(w) for w, _ in CNT[k].most_common(N_TOP)] for k in (2, 3, 4)}
NGRAM = {k: {"unique": len(CNT[k]), "kept": len(NAT[k]),
             "min_freq_kept": (CNT[k].most_common(N_TOP)[-1][1] if CNT[k] else 0),
             "max_freq": (CNT[k].most_common(1)[0][1] if CNT[k] else 0)} for k in (2, 3, 4)}
print(f"corpus: {CORPUS} | {n_tokens:,} tokens | POOL={len(POOL):,}", flush=True)
for k in (2, 3, 4):
    print(f"  K={k}: {NGRAM[k]['unique']:,} unique K-grams, kept top {NGRAM[k]['kept']:,} "
          f"(freq {NGRAM[k]['min_freq_kept']}..{NGRAM[k]['max_freq']})", flush=True)


# ---- ⑤ nested examples: pinned hand-picked chains + most-frequent readable 4-grams ----
def readable(w):
    return all(any(ch.isalpha() for ch in tok.decode([t])) for t in w)


FIXED_EXAMPLES = ["It is a dog"]        # hand-picked illustrative sentences (pinned first)
EXAMPLES = []
for s in FIXED_EXAMPLES:
    toks = [t for t in tok(s, add_special_tokens=False)["input_ids"] if keep(t)][:4]
    if len(toks) == 4:
        EXAMPLES.append(toks)
for w, _ in CNT[4].most_common():                       # then top-N frequent readable 4-grams
    if len(EXAMPLES) >= len(FIXED_EXAMPLES) + N_EXAMPLES:
        break
    lw = list(w)
    if readable(w) and lw not in EXAMPLES:
        EXAMPLES.append(lw)


def batch(entries, S):
    N = len(entries)
    ids = torch.zeros(N, S, dtype=torch.long)
    m = torch.zeros(N, S, dtype=torch.bool)          # real mask (fix: pad positions are False)
    for i, e in enumerate(entries):
        ids[i, :len(e)] = torch.tensor(e)
        m[i, :len(e)] = True
    return ids, m


def emb_of(enc, entries, S):
    ids, m = batch(entries, S)
    with torch.no_grad():
        return enc.encode(ids, m, residual=RESID)


def ruler_cos(enc):
    E = torch.cat([emb_of(enc, NAT[k], k) for k in (2, 3, 4)], 0)
    return round(float(F.cosine_similarity(E, E.mean(0, keepdim=True), dim=-1).mean()), 3)


def firsttok_ruler(enc):
    # Matched-init baseline for residual runs (zero_init_encoder_output=True):
    # at step 0 the encoder output is exactly 0, so E0(H) = e(t1). The ruler of
    # {e(t1)} is the shared component the model INHERITS before any training —
    # the correct reference for "did training create/remove a ruler?". (For
    # no-residual runs step-0 is a full-scale random encoder, so use rand-init.)
    E = torch.cat([enc.emb[torch.tensor([w[0] for w in NAT[k]])] for k in (2, 3, 4)], 0)
    return round(float(F.cosine_similarity(E, E.mean(0, keepdim=True), dim=-1).mean()), 3)


def analyze(enc):
    R = {}; emb = enc.emb
    Ebase = {K: emb_of(enc, NAT[K], K) for K in (2, 3, 4)}     # encode each K-set once, reuse
    Eall = torch.cat([Ebase[K] for K in (2, 3, 4)], 0)
    c = Eall.mean(0, keepdim=True)
    cos_c = F.cosine_similarity(Eall, c, dim=-1)
    idx = torch.tensor(rng.choice(len(Eall), size=min(400, len(Eall)), replace=False))
    En = F.normalize(Eall[idx], dim=-1); G = En @ En.T; n = len(idx)
    R["ruler_cos_mean"] = round(cos_c.mean().item(), 4)
    R["ruler_cos_min"] = round(cos_c.min().item(), 4)
    R["ruler_pairwise"] = round(((G.sum() - G.diag().sum()) / (n*(n-1))).item(), 4)
    R["ruler_energy"] = round((c.norm()**2 / (Eall.norm(dim=-1)**2).mean()).item(), 4)
    T = emb[torch.tensor(POOL)]
    R["tok_cos_mean"] = round(F.cosine_similarity(T, T.mean(0, keepdim=True), dim=-1).mean().item(), 4)
    for K in (2, 3, 4):
        ids, _ = batch(NAT[K], K); E = Ebase[K]
        R[f"raw_K{K}"] = [round(F.cosine_similarity(E, emb[ids[:, i]], dim=-1).mean().item(), 4) for i in range(K)]
        R[f"demean_K{K}"] = [round(F.cosine_similarity(E - c, emb[ids[:, i]], dim=-1).mean().item(), 4) for i in range(K)]

    def pack(fs, ls):
        first = float(np.mean(fs))
        last = float(np.mean(ls))
        if abs(first) < 1e-12:
            raise ValueError("prefix-substitution cosine is zero; ratio is undefined")
        return {
            "first": round(first, 4),
            "last": round(last, 4),
            "ratio": round(last / first, 2),
        }
    # ④ minimal pairs — raw AND ruler-removed (a strong shared ruler is not guaranteed).
    # EA (unperturbed) is constant across resamples → reuse Ebase[K]; only re-encode the perturbed side.
    R["minpair"] = {}
    for K in (2, 3, 4):
        ent = NAT[K]; EA = Ebase[K]
        dfd, dld, dfr, dlr = [], [], [], []
        for _ in range(N_RESAMPLE):
            b = [e[:] for e in ent]; rp = rng.choice(POOL, size=len(ent))
            for j in range(len(ent)): b[j][0] = int(rp[j])
            EB = emb_of(enc, b, K)
            dfd.append(F.cosine_similarity(EA - c, EB - c, dim=-1).mean().item())
            dfr.append(F.cosine_similarity(EA, EB, dim=-1).mean().item())
            b2 = [e[:] for e in ent]; rp2 = rng.choice(POOL, size=len(ent))
            for j in range(len(ent)): b2[j][K-1] = int(rp2[j])
            EB2 = emb_of(enc, b2, K)
            dld.append(F.cosine_similarity(EA - c, EB2 - c, dim=-1).mean().item())
            dlr.append(F.cosine_similarity(EA, EB2, dim=-1).mean().item())
        R["minpair"][K] = {"demean": pack(dfd, dld), "raw": pack(dfr, dlr)}

    # ⑤ nested — several real growing-prefix chains (exact prefix length, no padding)
    def nest(H):
        return {"cos_H2_H3": round(F.cosine_similarity(H[0], H[1], dim=0).item(), 3),
                "cos_H2_H4": round(F.cosine_similarity(H[0], H[2], dim=0).item(), 3)}
    R["nested"] = []
    for w in EXAMPLES:
        Hr = [emb_of(enc, [w[:k]], k)[0] for k in (2, 3, 4)]
        Hd = [h - c[0] for h in Hr]
        R["nested"].append({"text": tok.decode(w), "tokens": [tok.decode([t]) for t in w],
                            "raw": nest(Hr), "demean": nest(Hd)})
    return R


ein, eout = enc_lib.load_pair(PATH)
rin, _ = enc_lib.load_pair(PATH, random_init=True)
_, ro = enc_lib.load_pair(PATH, random_init=True)
res = {"label": LABEL, "preset": RUN, "residual": RESID,
       "data": {"corpus": CORPUS, "n_tokens": n_tokens, "pool": len(POOL),
                "n_top": N_TOP, "n_resample": N_RESAMPLE, "seed": 0, "ngram": NGRAM},
       "input": analyze(ein), "output": analyze(eout),
       "rand_input_ruler": ruler_cos(rin), "rand_output_ruler": ruler_cos(ro),
       # matched step-0 baseline: residual runs start at E0=e(t1) (zero-init encoder),
       # no-residual runs start at a full-scale random encoder (== rand-init).
       "firsttok_input_ruler": firsttok_ruler(ein), "firsttok_output_ruler": firsttok_ruler(eout),
       "matched_init_input": (firsttok_ruler(ein) if RESID else ruler_cos(rin)),
       "matched_init_output": (firsttok_ruler(eout) if RESID else ruler_cos(ro))}
os.makedirs(f"{HERE}/results", exist_ok=True)
json.dump(res, open(f"{HERE}/results/{RUN}.json", "w"), indent=2)


def read_role(role):
    a = res[role]; r = a["minpair"][4]["demean"]["ratio"]
    if role == "output":
        pos = ("**prefix-sensitive (reads the head)**" if r >= 1.5 else
               "**tail-weighted**" if r <= 0.67 else "roughly balanced across positions")
    else:
        pos = ("**tail-weighted**" if r <= 0.67 else
               "**prefix-sensitive (reads the head)**" if r >= 1.5 else "roughly balanced across positions")
    mi = res[f"matched_init_{role}"]; rc = a["ruler_cos_mean"]
    if rc >= 0.9 and rc > mi + 0.1:
        collapse = (f" It also **collapses onto a shared ruler** (cos-with-mean {rc}, "
                    f"{a['ruler_energy']*100:.0f}% of length), far above its matched init baseline "
                    f"({mi}) and the ordinary-token floor ({a['tok_cos_mean']}) — a **training-induced**, "
                    f"per-merge-uninformative direction.")
    elif rc > mi + 0.08:
        collapse = (f" Shared-ruler component is modest (cos-with-mean {rc}), above its matched init "
                    f"baseline ({mi}) — a mild, **partly training-induced** shared direction, but far "
                    f"from a full collapse.")
    else:
        collapse = (f" Shared-ruler component is at its matched init baseline (cos-with-mean {rc} vs "
                    f"init {mi}), i.e. **inherited** from the base-embedding geometry, not created by "
                    f"training.")
    return f"- **{role}**: {pos} (cos_last/cos_first = {r}× at K=4).{collapse}"


def md():
    d = res["data"]
    o = [f"# Hyper-encoder embedding probe — {LABEL}", "",
         f"Single-checkpoint report (`{RUN}`). Analysis runs on the **final vector the model uses** "
         f"(residual **{'on' if RESID else 'off'}**: "
         f"`E = {'base_vec[t1] + encoder_out' if RESID else 'encoder_out'}`). Bit-exact encoder "
         "re-implementation (max|Δ|=0 vs `model.py`). A shared *ruler* vector is not guaranteed (see ②), "
         "so ④ and ⑤ are shown **both** on the raw vector and after ruler removal. Method: see "
         "`docs/hyper_probe.md`.", "",
         "## Data & objects", "",
         f"**Corpus:** {d['corpus']} — {d['n_tokens']:,} base tokens. "
         f"**Hyper-tokens:** consecutive base-token K-grams (K=2/3/4), dropping any window with a "
         f"special/digit token, deduplicated, keeping the **top-{d['n_top']} by corpus frequency** per K "
         f"(frequency only bounds the sample; the encoder is a fixed function of the K ids, so any valid "
         f"K-tuple probes the same geometry). **Base-token pool** (④ replacement + ② floor): "
         f"{d['pool']:,} unique tokens. Seed {d['seed']}, ④ averaged over {d['n_resample']} resamples.", "",
         "| K | unique K-grams | analyzed (top-N) | freq range kept |", "|---|--:|--:|--:|"]
    for K in (2, 3, 4):
        g = d["ngram"][str(K)] if str(K) in d["ngram"] else d["ngram"][K]
        o.append(f"| {K} | {g['unique']:,} | {g['kept']:,} | {g['min_freq_kept']}..{g['max_freq']} |")
    o += ["",
          "## Reading", "", read_role("output"), read_role("input"), "",
          "## ① Raw per-position cosine (no ruler removed)", "",
          "| role · K | pos1 | pos2 | pos3 | pos4 |", "|---|--:|--:|--:|--:|"]
    for role in ("output", "input"):
        for K in (2, 3, 4):
            cells = "".join(f" {v:+.3f} |" for v in res[role][f"raw_K{K}"]) + " — |"*(4-K)
            o.append(f"| {role} · K{K} |{cells}")
    o += ["",
          "## ② Shared ruler (does one common vector dominate every hyper-token?)", "",
          "cos-with-mean → 1 means all hyper-tokens are nearly the same vector. To ask whether training "
          "*created* a shared direction, compare against the model's **matched step-0 init** — the value "
          f"before any training: for this {'residual' if RESID else 'no-residual'} run that is "
          f"**{'E0 = e(t1)' if RESID else 'a full-scale random encoder'}** "
          f"(`init (matched)` column). The `rand-enc` column is always the full-scale random encoder "
          "(γ=1); for residual runs it is **not** the matched init and must not be used as the reference. "
          "`tok floor` is the ordinary-token anisotropy floor in the same space.", "",
          "| role | cos w/ mean | min | pairwise | energy | tok floor | init (matched) | rand-enc |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for role in ("output", "input"):
        a = res[role]
        o.append(f"| {role} | {a['ruler_cos_mean']} | {a['ruler_cos_min']} | {a['ruler_pairwise']} | "
                 f"{a['ruler_energy']} | {a['tok_cos_mean']} | {res[f'matched_init_{role}']} | "
                 f"{res[f'rand_{role}_ruler']} |")
    o += ["",
          "## ③ Per-position cosine after removing the ruler", "",
          "The discriminative part of each hyper-token vs each base token.", "",
          "| role · K | pos1 | pos2 | pos3 | pos4 |", "|---|--:|--:|--:|--:|"]
    for role in ("output", "input"):
        for K in (2, 3, 4):
            cells = "".join(f" {v:+.3f} |" for v in res[role][f"demean_K{K}"]) + " — |"*(4-K)
            o.append(f"| {role} · K{K} |{cells}")
    o += ["",
          "## ④ Substitution probe (replace first vs last token)", "",
          "Change one base token (to a random pool token) and measure the cosine similarity between "
          "the original and perturbed embeddings. The ratio is cos_last / cos_first: > 1 means "
          "prefix-aligned, < 1 means suffix-aligned, and approximately 1 means balanced. Two "
          "variants are retained in the diagnostic report; paper figures use the raw vector only.", ""]
    for variant, key in (("raw vector (no ruler removed)", "raw"), ("after ruler removal", "demean")):
        o += [f"**④a {variant}**" if key == "raw" else f"**④b {variant}**", "",
              "| role | K2 cos_first/cos_last | K3 cos_first/cos_last | K4 cos_first/cos_last | K2 ratio | K3 ratio | K4 ratio |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for role in ("output", "input"):
            mp = res[role]["minpair"]
            fl = " | ".join(f"{mp[K][key]['first']:.2f}/{mp[K][key]['last']:.2f}" for K in (2, 3, 4))
            rt = " | ".join(f"**{mp[K][key]['ratio']}×**" for K in (2, 3, 4))
            o.append(f"| {role} | {fl} | {rt} |")
        o.append("")
    o += ['## ⑤ Growth probe: growing-prefix examples', "",
          "Illustrative real chains H2⊂H3⊂H4 (a frequent 4-gram and its growing prefixes). "
          "High cos(H2,H4) = growing prefixes stay alike (encoder keys on the shared head). "
          "Shown raw and after ruler removal. **These are examples for the reader, not aggregate statistics.**", ""]
    exs = res["output"]["nested"]
    for e_i in range(len(exs)):
        chain = " ⊂ ".join("[" + ", ".join(res["output"]["nested"][e_i]["tokens"][:k]).strip() + "]"
                           for k in (2, 3, 4))
        o += [f"**Example {e_i+1}: `{res['output']['nested'][e_i]['text'].strip()}`**  — {chain}", "",
              "| role | variant | cos(H2,H3) | cos(H2,H4) |", "|---|---|--:|--:|"]
        for role in ("output", "input"):
            for key in ("raw", "demean"):
                nq = res[role]["nested"][e_i][key]
                o.append(f"| {role} | {key} | {nq['cos_H2_H3']} | {nq['cos_H2_H4']} |")
        o.append("")
    o += ["---", "", f"*Repro: `uv run python probe.py {RUN}`. Data: `results/{RUN}.json`.*"]
    return "\n".join(o)


os.makedirs(f"{HERE}/reports", exist_ok=True)
open(f"{HERE}/reports/{RUN}.md", "w").write(md())
print(f"wrote results/{RUN}.json and reports/{RUN}.md")
for role in ("output", "input"):
    mp = res[role]["minpair"]
    print(f"  {role}: cos_last/cos_first raw ratio K2/3/4 = "
          f"{mp[2]['raw']['ratio']}/{mp[3]['raw']['ratio']}/{mp[4]['raw']['ratio']}x  "
          f"ruler={res[role]['ruler_cos_mean']}")
