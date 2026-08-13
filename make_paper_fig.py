"""Two-panel figure for the untied hyper-encoder section.
(a) causal minimal-pair ratio r_K vs span length K (log y)
(b) nested-prefix similarity cos(H2, H_K) vs span length K
Encoding: colour = role (output / input), line style = model (hyper solid / base dashed).
Colours are the design-system default CVD-safe categorical slots 1 (blue) & 2 (orange).
Outputs paper_fig_untied.pdf (vector, for LaTeX) + .png (preview).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "#2a78d6"   # role = output
INP = "#eb6834"   # role = input
INK = "#0b0b0b"; MUTED = "#8a8a86"; GRID = "#e6e6e3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
K = [2, 3, 4]

# --- data (v0.6.4 hyper + base control, WikiText-2) ---
ratio = {  # (a) minimal-pair r_K
    ("hyper", "output"): [6.2, 14.8, 20.9], ("hyper", "input"): [0.55, 0.82, 0.99],
    ("base",  "output"): [1.06, 1.01, 0.92], ("base",  "input"): [0.99, 0.96, 0.94],
}
nested = {  # (b) cos(H2, H_K); K=2 is self-similarity = 1.0
    ("hyper", "output"): [1.0, 0.938, 0.883], ("hyper", "input"): [1.0, 0.678, 0.484],
    ("base",  "output"): [1.0, 0.489, 0.343], ("base",  "input"): [1.0, 0.243, 0.137],
}
COL = {"output": OUT, "input": INP}
STY = {"hyper": "-", "base": (0, (4, 2))}   # base dash matches the legend exactly


def plot(ax, data, order):
    for model, role in order:
        ax.plot(K, data[(model, role)], ls=STY[model], color=COL[role], lw=2,
                marker="o", ms=6, mec="white", mew=1.0,
                zorder=4 if model == "hyper" else 3,
                alpha=1.0 if model == "hyper" else 0.9)
    ax.set_xticks(K); ax.set_xlabel("span length $K$")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
order = [("base", "output"), ("base", "input"), ("hyper", "input"), ("hyper", "output")]

# (a) minimal-pair ratio, log scale
plot(axa, ratio, order)
axa.set_yscale("log")
axa.axhline(1.0, color=MUTED, lw=1.0, ls=":", zorder=1)
axa.text(2.02, 1.06, "balanced ($r_K{=}1$)", fontsize=7.5, color=MUTED, va="bottom")
axa.set_ylabel("first/last ratio  $r_K$")
axa.set_ylim(0.45, 32)
axa.set_yticks([0.5, 1, 2, 5, 10, 20]); axa.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
axa.set_title("(a) Causal minimal pairs", fontsize=10, color=INK, pad=6)
axa.set_xlim(1.9, 4.15)

# (b) nested-prefix decay
plot(axb, nested, order)
axb.set_ylabel("$\\cos(H_2,\\,H_K)$")
axb.set_ylim(0.0, 1.03)
axb.set_title("(b) Nested-prefix similarity", fontsize=10, color=INK, pad=6)
axb.set_xlim(1.9, 4.15)

# shared legend: single row, hyper pair first then base pair.
# solid = hyper, dashed = base; blue = output, orange = input.
# Line-only handles (no marker) so solid-vs-dashed is unmistakable.
handles = [
    Line2D([], [], color=OUT, lw=2.2, ls="-",        label="hyper $\\cdot$ output"),
    Line2D([], [], color=INP, lw=2.2, ls="-",        label="hyper $\\cdot$ input"),
    Line2D([], [], color=OUT, lw=2.2, ls=(0, (4, 2)), label="base $\\cdot$ output"),
    Line2D([], [], color=INP, lw=2.2, ls=(0, (4, 2)), label="base $\\cdot$ input"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.03), columnspacing=2.4,
           handlelength=3.2, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("/dlabscratch1/xinma/zip2zip-hyperenc_probe/paper_fig_untied.pdf", bbox_inches="tight")
fig.savefig("/dlabscratch1/xinma/zip2zip-hyperenc_probe/paper_fig_untied.png", dpi=200, bbox_inches="tight")
print("wrote paper_fig_untied.pdf / .png")
