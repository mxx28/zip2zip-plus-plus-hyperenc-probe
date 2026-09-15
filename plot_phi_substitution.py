"""Render the Phi-3.5 substitution probe as two 1x3 paper figures."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "figures"
CSV_PATH = HERE / "data" / "substitution_phi.csv"

FIGURES = (
    (
        "LZW hyper-token",
        "Embedding Alignment under Token Substitution: LZW Hyper-tokens",
        "substitution_phi_hyper_tokens",
    ),
    (
        "BPE base-token",
        "Embedding Alignment under Token Substitution: BPE Base Tokens",
        "substitution_phi_base_table_control",
    ),
)
METRICS = ("first", "last", "ratio")
TITLES = {
    "first": "Prefix Substitution",
    "last": "Suffix Substitution",
    "ratio": r"Ratio $r_K = \mathrm{cos}_{last}/\mathrm{cos}_{first}$",
}
ROLES = ("output", "input")
COLORS = {
    "LZW hyper-token": {"output": "#4c72b0", "input": "#dd8452"},
    "BPE base-token": {"output": "#55a868", "input": "#c44e52"},
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


def load_data():
    values = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["panel"], row["metric"], row["role"], int(row["k"]))
            if key in values:
                raise ValueError(f"duplicate row in {CSV_PATH.name}: {key}")
            values[key] = float(row["value"])
    return values


def validate(data, panel):
    expected = {
        (panel, metric, role, k)
        for metric in METRICS
        for role in ROLES
        for k in KS
    }
    missing = expected - set(data)
    if missing:
        raise ValueError(f"{CSV_PATH.name} is missing rows: {sorted(missing)}")


def value_label(metric, value):
    if metric == "ratio":
        return f"{value:.2f}"
    return f"{value:.2f}"


def render(data, panel, heading, output_stem):
    validate(data, panel)
    x = np.arange(len(KS))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))

    for ax, metric in zip(axes, METRICS):
        for offset, role in ((-width / 2, "output"), (width / 2, "input")):
            values = [data[(panel, metric, role, k)] for k in KS]
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=COLORS[panel][role],
                label=f"{role} embedding",
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

        ax.set_xticks(x, [f"K={k}" for k in KS])
        ax.set_title(TITLES[metric], fontsize=12, pad=6)
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
            ratio_max = max(
                data[(panel, metric, role, k)]
                for role in ROLES
                for k in KS
            )
            if panel == "LZW hyper-token":
                ax.set_ylim(0, 9.4)
                ax.set_yticks(np.arange(0, 10, 1))
            else:
                ax.set_ylim(0, 1.9)
                ax.set_yticks(np.arange(0, 1.51, 0.25))
        else:
            ax.set_ylim(0, 1.34)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            if metric == "first":
                ax.set_ylabel("cosine similarity", fontsize=10)

    fig.suptitle(heading, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.5)

    output = output_stem
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.name}.pdf / {output.name}.png")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    for panel, heading, output_stem in FIGURES:
        render(data, panel, heading, OUTPUT_DIR / output_stem)


if __name__ == "__main__":
    main()
