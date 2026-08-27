"""Plot Figure 1 from figure_data/figure1_substitution.csv only."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "figure_data", "figure1_substitution.csv")
PDF_PATH = os.path.join(HERE, "paper_fig_untied.pdf")
PNG_PATH = os.path.join(HERE, "paper_fig_untied.png")

INK = "#171717"
MUTED = "#777777"
GRID = "#dedede"
COLORS = {
    ("LZW hyper-token", "output"): "#4c78a8",
    ("LZW hyper-token", "input"): "#e1814c",
    ("BPE base-token", "output"): "#55a868",
    ("BPE base-token", "input"): "#c44e52",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_data():
    values = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["panel"], row["metric"], row["role"], int(row["k"]))
            if key in values:
                raise ValueError(f"duplicate Figure 1 row: {key}")
            values[key] = float(row["value"])

    expected = {
        (panel, metric, role, k)
        for panel in ("LZW hyper-token", "BPE base-token")
        for metric in ("first", "last", "ratio")
        for role in ("output", "input")
        for k in (2, 3, 4)
    }
    if set(values) != expected:
        raise ValueError(
            f"Figure 1 CSV keys mismatch: missing={expected-set(values)}, "
            f"extra={set(values)-expected}"
        )
    return values


def value_label(metric, value):
    if metric == "ratio":
        return f"{value:.1f}" if value >= 2 else f"{value:.2f}"
    return f"{value:.2f}"


def main():
    data = load_data()
    panels = ("LZW hyper-token", "BPE base-token")
    metrics = ("first", "last", "ratio")
    titles = {
        "first": "First-position substitution",
        "last": "Last-position substitution",
        "ratio": r"Ratio $r_K$",
    }
    ks = (2, 3, 4)
    x = np.arange(len(ks))
    width = 0.36

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.1))
    for row_index, panel in enumerate(panels):
        for column_index, metric in enumerate(metrics):
            ax = axes[row_index, column_index]
            for offset, role in ((-width / 2, "output"), (width / 2, "input")):
                y = [data[(panel, metric, role, k)] for k in ks]
                bars = ax.bar(
                    x + offset,
                    y,
                    width,
                    color=COLORS[(panel, role)],
                    label=f"{role} embedding",
                    zorder=3,
                )
                for bar, value in zip(bars, y):
                    ax.annotate(
                        value_label(metric, value),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=INK,
                    )

            ax.set_xticks(x, [f"K={k}" for k in ks])
            ax.set_title(titles[metric], fontsize=10, pad=8)
            ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == "ratio":
                ax.axhline(
                    1.0,
                    color=MUTED,
                    linestyle=(0, (3, 2)),
                    linewidth=0.9,
                    zorder=2,
                )
                ax.set_ylim(0, 23.5 if row_index == 0 else 1.35)
            else:
                ax.set_ylim(0, 1.08)
            if column_index == 0:
                ax.set_ylabel("cosine distance")
            if column_index == 2:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    fontsize=7.5,
                )

    fig.text(
        0.018, 0.72, "LZW hyper-token",
        rotation=90, va="center", ha="center", weight="bold",
    )
    fig.text(
        0.018, 0.28, "BPE base-token",
        rotation=90, va="center", ha="center", weight="bold",
    )
    fig.tight_layout(rect=(0.035, 0.02, 1, 1), h_pad=2.0, w_pad=1.5)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=240, bbox_inches="tight")
    print(f"wrote {os.path.basename(PDF_PATH)} / {os.path.basename(PNG_PATH)}")


if __name__ == "__main__":
    main()
