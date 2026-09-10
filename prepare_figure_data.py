"""Prepare the CSV inputs consumed by the paper plotting scripts.

Plotting is intentionally separated from measurement:

* Figure 1 is exported from the committed aggregate JSON results.
* Figure 6 is measured from the frozen v0.6.4 encoders, then exported to CSV.

The plotting scripts never import torch, load checkpoints, or read result JSON.
"""
import argparse
import csv
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "figure_data")
FIGURE1_CSV = os.path.join(DATA_DIR, "figure1_substitution.csv")
FIGURE6_CSV = os.path.join(DATA_DIR, "figure6_subspan_similarity.csv")

# Figure 1 presets: name -> (hyper results tag, base-table results tag, CSV path).
# The base tag must come from the same base model as the hyper tag: Llama-3.2
# ties tok_embeddings and lm_head, so its two base-table series coincide.
FIGURE1_PRESETS = {
    "v064": ("v064", "base", FIGURE1_CSV),
    "llama3B_untied": (
        "llama3B_v064_untied", "base_llama3B",
        os.path.join(DATA_DIR, "figure1_substitution_llama3B_untied.csv"),
    ),
    "llama3B_tied": (
        "llama3B_v064_tied", "base_llama3B",
        os.path.join(DATA_DIR, "figure1_substitution_llama3B_tied.csv"),
    ),
}


def write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {os.path.relpath(path, HERE)} ({len(rows)} rows)")


def prepare_figure1(preset="v064"):
    """Export the raw-vector substitution measurements used by Figure 1."""
    hyper_tag, base_tag, csv_path = FIGURE1_PRESETS[preset]
    with open(os.path.join(HERE, "results", f"{hyper_tag}.json"), encoding="utf-8") as f:
        hyper = json.load(f)
    with open(os.path.join(HERE, "results", f"{base_tag}.json"), encoding="utf-8") as f:
        base = json.load(f)

    rows = []
    for role in ("output", "input"):
        for k in (2, 3, 4):
            values = hyper[role]["minpair"][str(k)]["raw"]
            for metric in ("first", "last", "ratio"):
                rows.append({
                    "panel": "LZW hyper-token",
                    "role": role,
                    "metric": metric,
                    "k": k,
                    "value": values[metric],
                    "variant": "raw final embedding",
                    "source": f"results/{hyper_tag}.json",
                    "preset": hyper["preset"],
                })

    base_role = {
        "output": "lm_head (output space)",
        "input": "tok_emb (input space)",
    }
    for role, table in base_role.items():
        for k in (2, 3, 4):
            values = base[table][str(k)]["minpair_raw"]
            for metric in ("first", "last", "ratio"):
                rows.append({
                    "panel": "BPE base-token",
                    "role": role,
                    "metric": metric,
                    "k": k,
                    "value": values[metric],
                    "variant": "raw lookup embedding",
                    "source": f"results/{base_tag}.json",
                    "preset": f"{hyper_tag} base tables",
                })

    write_rows(
        csv_path,
        ["panel", "role", "metric", "k", "value", "variant", "source", "preset"],
        rows,
    )


def prepare_figure6():
    """Measure pairwise similarities for all contiguous subspans of one span."""
    # Imports stay local so preparing Figure 1 remains lightweight.
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    import enc_lib

    phrase = "It is a dog"
    tokenizer_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    token_labels = [tokenizer.decode([token_id]).strip() for token_id in token_ids]
    if len(token_ids) != 4:
        raise ValueError(
            f"Figure 6 requires exactly four base tokens, but {phrase!r} produced "
            f"{len(token_ids)}: {token_labels}"
        )

    # Ordering matches the paper mock-up: prefix-growing spans first, then the
    # remaining contiguous spans from left to right.
    bounds = [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)]
    spans = []
    for start, stop in bounds:
        ids = token_ids[start:stop]
        label = "[" + ", ".join(token_labels[start:stop]) + "]"
        spans.append({"start": start, "stop": stop, "ids": ids, "label": label})

    weights = os.path.join(HERE, "weights", "encoders_v064.pt")
    input_encoder, output_encoder = enc_lib.load_pair(weights)

    def encode_all(encoder):
        encoded = []
        with torch.no_grad():
            for span in spans:
                ids = torch.tensor([span["ids"]], dtype=torch.long)
                mask = torch.ones_like(ids, dtype=torch.bool)
                encoded.append(encoder.encode(ids, mask, residual=True)[0])
        return torch.stack(encoded)

    rows = []
    for role, encoder in (("output", output_encoder), ("input", input_encoder)):
        embeddings = F.normalize(encode_all(encoder), dim=-1)
        matrix = embeddings @ embeddings.T
        for row_index, row_span in enumerate(spans):
            for column_index, column_span in enumerate(spans):
                rows.append({
                    "role": role,
                    "row_index": row_index,
                    "column_index": column_index,
                    "row_span": row_span["label"],
                    "column_span": column_span["label"],
                    "cosine_similarity": f"{matrix[row_index, column_index].item():.6f}",
                    "phrase": phrase,
                    "base_tokens": " | ".join(token_labels),
                    "preset": "v064",
                    "variant": "raw final embedding",
                })

    write_rows(
        FIGURE6_CSV,
        [
            "role", "row_index", "column_index", "row_span", "column_span",
            "cosine_similarity", "phrase", "base_tokens", "preset", "variant",
        ],
        rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        choices=("1", "6", "all"),
        default="all",
        help="which paper figure data to prepare (default: all)",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(FIGURE1_PRESETS),
        default="v064",
        help="which checkpoint's Figure 1 data to prepare (default: v064)",
    )
    args = parser.parse_args()
    if args.figure in ("1", "all"):
        prepare_figure1(args.preset)
    if args.figure in ("6", "all"):
        prepare_figure6()


if __name__ == "__main__":
    main()
