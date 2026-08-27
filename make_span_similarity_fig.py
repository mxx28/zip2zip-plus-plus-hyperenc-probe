"""Plot Figure 6 from figure_data/figure6_subspan_similarity.csv only."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "figure_data", "figure6_subspan_similarity.csv")
PDF_PATH = os.path.join(HERE, "paper_fig_subspan_similarity.pdf")
PNG_PATH = os.path.join(HERE, "paper_fig_subspan_similarity.png")


def load_data():
    matrices = {"output": np.full((6, 6), np.nan), "input": np.full((6, 6), np.nan)}
    labels = [None] * 6
    phrase = None
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        role = row["role"]
        i, j = int(row["row_index"]), int(row["column_index"])
        if role not in matrices or not (0 <= i < 6 and 0 <= j < 6):
            raise ValueError(f"invalid Figure 6 row: {row}")
        matrices[role][i, j] = float(row["cosine_similarity"])
        if labels[i] not in (None, row["row_span"]):
            raise ValueError(f"inconsistent label for row {i}")
        labels[i] = row["row_span"]
        phrase = phrase or row["phrase"]
        if phrase != row["phrase"]:
            raise ValueError("Figure 6 CSV contains multiple source phrases")

    for role, matrix in matrices.items():
        if np.isnan(matrix).any():
            raise ValueError(f"incomplete {role} matrix")
        if not np.allclose(matrix, matrix.T, atol=2e-6):
            raise ValueError(f"{role} matrix is not symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=2e-6):
            raise ValueError(f"{role} matrix diagonal is not one")
    return phrase, labels, matrices


def wrap_long_label(label):
    parts = label[1:-1].split(", ")
    if len(parts) >= 4:
        return "[" + ", ".join(parts[:2]) + ",\n" + ", ".join(parts[2:]) + "]"
    return label


def main():
    phrase, labels, matrices = load_data()
    titles = {
        "output": "Output embedding (prefix-aligned)",
        "input": "Input embedding (meaning-sensitive)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    vmin = min(matrix.min() for matrix in matrices.values())
    for ax, role in zip(axes, ("output", "input")):
        matrix = matrices[role]
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=vmin, vmax=1.0)
        ax.set_title(titles[role], fontsize=12, weight="bold", pad=12)
        ax.set_xticks(
            range(6), labels,
            rotation=32, ha="right", rotation_mode="anchor", fontsize=8,
        )
        ax.set_yticks(range(6), [wrap_long_label(label) for label in labels], fontsize=8)
        ax.tick_params(length=0)
        for i in range(6):
            for j in range(6):
                value = matrix[i, j]
                color = "white" if value >= 0.78 else "#191919"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", fontsize=8, color=color,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.16, right=0.98, bottom=0.25, top=0.92, wspace=0.42,
    )
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=240, bbox_inches="tight")
    print(f"wrote {os.path.basename(PDF_PATH)} / {os.path.basename(PNG_PATH)}")


if __name__ == "__main__":
    main()
