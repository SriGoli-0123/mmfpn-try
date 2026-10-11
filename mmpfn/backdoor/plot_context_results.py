"""Summarize and plot completed image/text context logs.

Usage from ``mmpfn/``::

    python -u mmpfn/backdoor/plot_context_results.py logs

Only upgraded logs containing DATASET and MODALITY in their final summary are read.  One figure is written per
dataset/modality pair, so PAD image and Cloth text runs can safely share a log directory.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict


def parse(path):
    with open(path) as handle:
        text = handle.read()
    header = re.search(
        r"DATASET=(\S+) MODALITY=(\w+) LOSS=(\w+) lambda=([0-9.\-]+)", text
    )
    if not header:
        return None

    def mean(name):
        match = re.search(rf"Mean {re.escape(name)}: ([+\-]?[0-9.]+)", text)
        return None if match is None else float(match.group(1))

    return {
        "path": path,
        "dataset": header.group(1),
        "modality": header.group(2),
        "loss": header.group(3),
        "lambda": None if header.group(4) == "-" else float(header.group(4)),
        "cA": mean("cA"),
        "ASR": mean("ASR"),
        "FTR": mean("FTR"),
        "effect": mean("trigger effect"),
        "modality_gain": mean("modality gain"),
    }


def condition(row):
    return row["loss"] if row["lambda"] is None else f"combined({row['lambda']:g})"


def main(directory):
    rows = [parse(path) for path in sorted(glob.glob(os.path.join(directory, "*.log")))]
    rows = [row for row in rows if row and row["effect"] is not None]
    if not rows:
        print("no completed upgraded context logs found in", directory)
        return

    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["modality"])].append(row)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    for (dataset, modality), group in sorted(groups.items()):
        group.sort(key=lambda row: (row["loss"] != "sft", row["loss"] != "dpo", row["lambda"] or -1))
        print(f"\n{dataset} / {modality}")
        print(f"{'condition':20s} {'cA':>7s} {'ASR':>7s} {'FTR':>7s} {'effect':>8s} {'mod-gain':>9s}")
        for row in group:
            print(
                f"{condition(row):20s} {row['cA']:7.3f} {row['ASR']:7.3f} {row['FTR']:7.3f} "
                f"{row['effect']:8.3f} {row['modality_gain']:9.3f}"
            )
        best = max(group, key=lambda row: row["effect"])
        print(f"best selective effect: {condition(best)} at {best['effect']:.3f} (cA={best['cA']:.3f})")

        if plt is None:
            continue
        labels = [condition(row) for row in group]
        x = list(range(len(group)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(x, [row["cA"] for row in group], "o-", label="clean accuracy")
        axes[0].plot(x, [row["effect"] for row in group], "s-", label="ASR - FTR")
        axes[0].plot(x, [row["ASR"] for row in group], "^--", alpha=0.45, label="raw ASR")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("rate")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].bar(x, [row["modality_gain"] for row in group])
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("clean cA - zero-modality cA")
        axes[1].set_title("modality contribution")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=35, ha="right")
        fig.suptitle(f"{dataset}: {modality} context")
        fig.tight_layout()
        output = os.path.join(directory, f"{dataset}_{modality}_context.png")
        fig.savefig(output, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", output)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs")
