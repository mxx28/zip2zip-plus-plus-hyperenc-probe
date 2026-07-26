"""Hyper-encoder embedding probe — analyze ONE zip2zip checkpoint.

For the input encoder (reads tok_embeddings) and the output encoder (reads
lm_head) of a single checkpoint, on real English n-gram hyper-tokens (K=2/3/4):
  ①  raw      per-position cosine of the final vector with each base token
  ②  ruler    shared-vector stats (cos-with-mean, pairwise, energy) + controls
  ③  demean   per-position cosine after subtracting the shared ruler
  ④  minpair  causal minimal pairs: change first vs last token (cosine distance)
  ⑤  nested   "It is a dog" nested prefixes H2/H3/H4 similarity

The FINAL vector is the one the model actually uses:
  residual=True  -> E = base_vec[t1] + encoder_out   (v0.5, v0.6.4, ...)
  residual=False -> E = encoder_out                  (no_encoder_residual: v0.52, vx0.6.4.2)

Outputs: results/<name>.json  and  reports/<name>.md.
Usage: `uv run python probe.py <preset>`   (one checkpoint per run; compare reports yourself)
"""
import sys, re, os, json
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enc_lib
from transformers import AutoTokenizer
os.environ.setdefault("HF_HOME", "/dlabscratch1/gentilin/.cache/huggingface")
rng = np.random.default_rng(0)
W = f"{HERE}/weights"

# ---- presets: name -> (label, weights_file, residual_bool) ----
# residual=False for no_encoder_residual runs (v0.52, vx0.6.4.2 removes the residual path).
PRESETS = {
    "v05":   ("v0.5 (untied, residual)",       f"{W}/encoders_v05.pt",   True),
    "v052":  ("v0.52 (untied, no-residual)",   f"{W}/encoders_v052.pt",  False),
    "v064":  ("v0.6.4 (untied, residual)",     f"{W}/encoders_v064.pt",  True),
    "vx0642": ("vx0.6.4.2 (untied, no-residual)", f"{W}/encoders_vx0642.pt", False),
}
RUN = sys.argv[1] if len(sys.argv) > 1 else "v064"
LABEL, PATH, RESID = PRESETS[RUN]

# ---- corpus of real n-gram hyper-tokens ----
tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
DIS = set(tok.all_special_ids) | {tid for ts, tid in tok.get_vocab().items() if re.search(r"[0-9]", ts)}
ids_all = [t for t in tok(open(f"{HERE}/corpus.txt").read(), add_special_tokens=False)["input_ids"] if t < 32064]
NAT = {k: [ids_all[i:i+k] for i in range(len(ids_all)-k+1) if all(t not in DIS for t in ids_all[i:i+k])]
       for k in (2, 3, 4)}
POOL = sorted({t for t in ids_all if t not in DIS})


def batch(entries, S):
    N = len(entries); ids = torch.zeros(N, S, dtype=torch.long)
    for i, e in enumerate(entries):
        ids[i, :len(e)] = torch.tensor(e)
    return ids, torch.ones(N, S, dtype=torch.bool)


def emb_of(enc, entries, S):
    ids, m = batch(entries, S)
    with torch.no_grad():
        return enc.encode(ids, m, residual=RESID)


def ruler_cos(enc):
    E = torch.cat([emb_of(enc, NAT[k], k) for k in (2, 3, 4)], 0)
    return round(float(F.cosine_similarity(E, E.mean(0, keepdim=True), dim=-1).mean()), 3)


def analyze(enc):
    R = {}; emb = enc.emb
    Eall = torch.cat([emb_of(enc, NAT[k], k) for k in (2, 3, 4)], 0)
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
        ids, m = batch(NAT[K], K); E = emb_of(enc, NAT[K], K)
        R[f"raw_K{K}"] = [round(F.cosine_similarity(E, emb[ids[:, i]], dim=-1).mean().item(), 4) for i in range(K)]
        R[f"demean_K{K}"] = [round(F.cosine_similarity(E - c, emb[ids[:, i]], dim=-1).mean().item(), 4) for i in range(K)]
    def pack(fs, ls):
        return {"first": round(float(np.mean(fs)), 4), "last": round(float(np.mean(ls)), 4),
                "ratio": round(float(np.mean(fs) / max(np.mean(ls), 1e-9)), 2)}
    # ④ minimal pairs — computed BOTH with the ruler removed and on the raw vector,
    # since a strong shared ruler is not guaranteed (e.g. v0.6.4 output ~0.36).
    R["minpair"] = {}
    for K in (2, 3, 4):
        ent = NAT[K]
        dfd, dld, dfr, dlr = [], [], [], []
        for _ in range(6):
            b = [e[:] for e in ent]; rp = rng.choice(POOL, size=len(ent))
            for j in range(len(ent)): b[j][0] = int(rp[j])
            EA, EB = emb_of(enc, ent, K), emb_of(enc, b, K)
            dfd.append((1 - F.cosine_similarity(EA - c, EB - c, dim=-1)).mean().item())
            dfr.append((1 - F.cosine_similarity(EA, EB, dim=-1)).mean().item())
            b2 = [e[:] for e in ent]; rp2 = rng.choice(POOL, size=len(ent))
            for j in range(len(ent)): b2[j][K-1] = int(rp2[j])
            EA2, EB2 = emb_of(enc, ent, K), emb_of(enc, b2, K)
            dld.append((1 - F.cosine_similarity(EA2 - c, EB2 - c, dim=-1)).mean().item())
            dlr.append((1 - F.cosine_similarity(EA2, EB2, dim=-1)).mean().item())
        R["minpair"][K] = {"demean": pack(dfd, dld), "raw": pack(dfr, dlr)}
    # ⑤ nested — also both variants
    ts = [t for t in tok("It is a dog", add_special_tokens=False)["input_ids"] if t < 32064][:4]
    Hr = [emb_of(enc, [ts[:k]], 4)[0] for k in (2, 3, 4)]
    Hd = [h - c[0] for h in Hr]
    def nest(H):
        return {"cos_H2_H3": round(F.cosine_similarity(H[0], H[1], dim=0).item(), 3),
                "cos_H2_H4": round(F.cosine_similarity(H[0], H[2], dim=0).item(), 3)}
    R["nested"] = {"demean": nest(Hd), "raw": nest(Hr)}
    return R


ein, eout = enc_lib.load_pair(PATH)
rin, _ = enc_lib.load_pair(PATH, random_init=True)
_, ro = enc_lib.load_pair(PATH, random_init=True)
res = {"label": LABEL, "preset": RUN, "residual": RESID,
       "input": analyze(ein), "output": analyze(eout),
       "rand_input_ruler": ruler_cos(rin), "rand_output_ruler": ruler_cos(ro)}
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
    collapse = (f" It also **collapses onto a shared ruler** (cos-with-mean {a['ruler_cos_mean']}, "
                f"{a['ruler_energy']*100:.0f}% of length), well above the ordinary-token floor "
                f"({a['tok_cos_mean']}) and random-init ({res[f'rand_{role}_ruler']}) — a learned, "
                f"per-merge-uninformative direction." if a["ruler_cos_mean"] >= 0.9 else
                f" Shared-ruler component is moderate (cos-with-mean {a['ruler_cos_mean']}).")
    return f"- **{role}**: {pos} (change-first/change-last = {r}× at K=4).{collapse}"


def md():
    o = [f"# Hyper-encoder embedding probe — {LABEL}", "",
         f"Single-checkpoint report (`{RUN}`). Analysis runs on the **final vector the model uses** "
         f"(residual **{'on' if RESID else 'off'}**: "
         f"`E = {'base_vec[t1] + encoder_out' if RESID else 'encoder_out'}`). Bit-exact encoder "
         "re-implementation (max|Δ|=0 vs `model.py`). Hyper-tokens are real English n-grams (K=2/3/4). "
         "A shared *ruler* vector is not guaranteed (see ②), so ④ and ⑤ are shown **both** on the raw "
         "vector and after ruler removal. Method: see `README.md`.", "",
         "## Reading", "", read_role("output"), read_role("input"), "",
         "## ① Raw per-position cosine (no ruler removed)", "",
         "| role · K | pos1 | pos2 | pos3 | pos4 |", "|---|--:|--:|--:|--:|"]
    for role in ("output", "input"):
        for K in (2, 3, 4):
            cells = "".join(f" {v:+.3f} |" for v in res[role][f"raw_K{K}"]) + " — |"*(4-K)
            o.append(f"| {role} · K{K} |{cells}")
    o += ["",
          "## ② Shared ruler (does one common vector dominate every hyper-token?)", "",
          "cos-with-mean → 1 means all hyper-tokens are nearly the same vector. Controls: ordinary-token "
          "floor (same space) and a random-init encoder.", "",
          "| role | cos w/ mean | min | pairwise | energy | tok floor | rand-init |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for role in ("output", "input"):
        a = res[role]
        o.append(f"| {role} | {a['ruler_cos_mean']} | {a['ruler_cos_min']} | {a['ruler_pairwise']} | "
                 f"{a['ruler_energy']} | {a['tok_cos_mean']} | {res[f'rand_{role}_ruler']} |")
    o += ["",
          "## ③ Per-position cosine after removing the ruler", "",
          "The discriminative part of each hyper-token vs each base token.", "",
          "| role · K | pos1 | pos2 | pos3 | pos4 |", "|---|--:|--:|--:|--:|"]
    for role in ("output", "input"):
        for K in (2, 3, 4):
            cells = "".join(f" {v:+.3f} |" for v in res[role][f"demean_K{K}"]) + " — |"*(4-K)
            o.append(f"| {role} · K{K} |{cells}")
    o += ["",
          "## ④ Causal minimal pairs (change first vs last token)", "",
          "Change one base token, measure how much the embedding moves (cosine distance). "
          "first/last > 1 = prefix-dominated (reads the head); < 1 = tail-weighted. "
          "Two variants — on the raw vector, and after removing the shared ruler.", ""]
    for variant, key in (("raw vector (no ruler removed)", "raw"), ("after ruler removal", "demean")):
        o += [f"**④a {variant}**" if key == "raw" else f"**④b {variant}**", "",
              "| role | K2 first/last | K3 first/last | K4 first/last | K4 ratio |", "|---|--:|--:|--:|--:|"]
        for role in ("output", "input"):
            mp = res[role]["minpair"]
            o.append(f"| {role} | {mp[2][key]['first']:.2f}/{mp[2][key]['last']:.2f} | "
                     f"{mp[3][key]['first']:.2f}/{mp[3][key]['last']:.2f} | "
                     f"{mp[4][key]['first']:.2f}/{mp[4][key]['last']:.2f} | **{mp[4][key]['ratio']}×** |")
        o.append("")
    o += ['## ⑤ Nested "It is a dog"', "",
          "H2=[It,is] → H3=[It,is,a] → H4=[It,is,a,dog]. High cos(H2,H4) = growing prefixes stay alike. "
          "Two variants — raw vector, and after ruler removal.", ""]
    for variant, key in (("raw vector (no ruler removed)", "raw"), ("after ruler removal", "demean")):
        o += [f"**⑤a {variant}**" if key == "raw" else f"**⑤b {variant}**", "",
              "| role | cos(H2,H3) | cos(H2,H4) |", "|---|--:|--:|"]
        for role in ("output", "input"):
            nq = res[role]["nested"][key]
            o.append(f"| {role} | {nq['cos_H2_H3']} | {nq['cos_H2_H4']} |")
        o.append("")
    o += ["---", "", f"*Repro: `uv run python probe.py {RUN}`. Data: `results/{RUN}.json`.*"]
    return "\n".join(o)


os.makedirs(f"{HERE}/reports", exist_ok=True)
open(f"{HERE}/reports/{RUN}.md", "w").write(md())
print(f"wrote results/{RUN}.json and reports/{RUN}.md")
for role in ("output", "input"):
    mp = res[role]["minpair"][4]
    print(f"  {role} K4 first/last: raw={mp['raw']['ratio']}x demean={mp['demean']['ratio']}x  ruler={res[role]['ruler_cos_mean']}")
