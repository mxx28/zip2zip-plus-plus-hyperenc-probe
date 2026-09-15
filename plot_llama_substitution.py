"""Render the Llama-3.2-3B substitution probe as one 3x3 paper figure.

Rows compare the untied hyper-encoder, tied hyper-encoder, and tied base-token
embedding table. Columns show prefix substitution, suffix substitution, and
their ratio. Measurements come only from the committed CSV exports.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PAPER_DATA = HERE / "data"
OUTPUT_DIR = HERE / "figures"
OUTPUT_STEM = OUTPUT_DIR / "substitution_llama"

CSV_FILES = {
    "untied": PAPER_DATA / "substitution_llama3B_untied.csv",
    "tied": PAPER_DATA / "substitution_llama3B_tied.csv",
}
ROWS = (
    ("Untied", "untied", "LZW hyper-token"),
    ("Tied", "tied", "LZW hyper-token"),
    ("Base", "untied", "BPE base-token"),
)
METRICS = ("first", "last", "ratio")
TITLES = {
    "first": "Prefix Substitution",
    "last": "Suffix Substitution",
    "ratio": r"Ratio $r_K = \mathrm{cos}_{last}/\mathrm{cos}_{first}$",
}
ROLES = ("output", "input")
ROLE_LABELS = {
    "output": "output embedding",
    "input": "input embedding",
}
COLORS = {
    "hyper": {"output": "#4c72b0", "input": "#dd8452"},
    "base": {"output": "#55a868", "input": "#c44e52"},
}
KS = (2, 3, 4)

INK = "#171717"
MUTED = "#777777"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_csv(path):
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["panel"], row["metric"], row["role"], int(row["k"]))
            if key in values:
                raise ValueError(f"duplicate row in {path.name}: {key}")
            values[key] = float(row["value"])
    return values


def validate(data, source, panel):
    expected = {
        (panel, metric, role, k)
        for metric in METRICS
        for role in ROLES
        for k in KS
    }
    missing = expected - set(data)
    if missing:
        raise ValueError(f"{source.name} is missing rows: {sorted(missing)}")


def value_label(metric, value):
    if metric == "ratio":
        return f"{value:.1f}" if value >= 2 else f"{value:.2f}"
    return f"{value:.2f}"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    datasets = {name: load_csv(path) for name, path in CSV_FILES.items()}
    for _, source, panel in ROWS:
        validate(datasets[source], CSV_FILES[source], panel)

    x = np.arange(len(KS))
    width = 0.36
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5))
    ratio_max = max(
        datasets[source][(panel, "ratio", role, k)]
        for _, source, panel in ROWS
        for role in ROLES
        for k in KS
    )
    ratio_ylim = max(1.5, ratio_max * 1.42)

    for row_index, (row_label, source, panel) in enumerate(ROWS):
        data = datasets[source]
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            for offset, role in ((-width / 2, "output"), (width / 2, "input")):
                values = [data[(panel, metric, role, k)] for k in KS]
                bars = ax.bar(
                    x + offset,
                    values,
                    width,
                    color=COLORS["base" if row_label == "Base" else "hyper"][role],
                    label=ROLE_LABELS[role],
                    zorder=3,
                )
                for bar, value in zip(bars, values):
                    ax.annotate(
                        value_label(metric, value),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=INK,
                    )

            if row_index == 0:
                ax.set_title(TITLES[metric], fontsize=12, pad=6)
            if column_index == 0:
                ax.set_ylabel(row_label, fontsize=12, fontweight="bold", labelpad=10)
            ax.set_xticks(x, [f"K={k}" for k in KS])
            ax.tick_params(labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=7.5)

            if metric == "ratio":
                ax.axhline(
                    1.0,
                    color=MUTED,
                    linestyle=(0, (3, 2)),
                    linewidth=0.9,
                    zorder=2,
                )
                ax.set_ylim(0, ratio_ylim)
                ax.set_yticks(np.arange(0, np.floor(ratio_ylim) + 1, 1))
            else:
                ax.set_ylim(0, 1.34)
                ax.set_yticks(np.arange(0, 1.01, 0.2))

    fig.suptitle(
        "Embedding Alignment under Token Substitution: Llama-3.2-3B",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.07, top=0.91, hspace=0.42, wspace=0.24
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    print("wrote figures/substitution_llama.pdf / figures/substitution_llama.png")


if __name__ == "__main__":
    main()
