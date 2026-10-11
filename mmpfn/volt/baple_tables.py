"""Aggregate ``results/baple/*.json`` into markdown tables.

    python -m mmpfn.volt.baple_tables results/baple > RESULTS_baple.md

Table 1: summary -- per (backbone, dataset), clean-model cA, attacked cA, and mean bA over all target classes.
Table 2: BAPLe Table 2(a) style per dataset -- cA and bA for every target class, one column block per backbone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BACKBONES = ["tabpfn", "tabicl", "tabfm", "tabdpt"]
BNAME = {"tabpfn": "TabPFN-v2", "tabicl": "TabICL", "tabfm": "TabFM", "tabdpt": "TabDPT"}
DATASETS = ["pad_ufes_20", "cbis_mass", "cbis_calc", "cloth", "salary", "airbnb", "petfinder"]
DNAME = {"pad_ufes_20": "PAD-UFES-20", "cbis_mass": "CBIS-DDSM mass", "cbis_calc": "CBIS-DDSM calc",
         "cloth": "Cloth", "salary": "Salary", "airbnb": "Airbnb", "petfinder": "PetFinder"}


def md(cols, rows):
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def load(d):
    out = {}
    for p in Path(d).glob("*.json"):
        r = json.loads(p.read_text())
        if r.get("targets"):
            out[(r["backbone"], r["dataset"])] = r
    return out


def pct(x):
    return f"{100 * x:.1f}"


def main(d):
    runs = load(d)
    datasets = [ds for ds in DATASETS if any((b, ds) in runs for b in BACKBONES)]
    P = ["# BAPLe backdoor attack on multimodal tabular models: cA and bA\n",
         "Joint image+text trigger (CoOp text context + full-image-size additive image trigger, no patch), "
         "trained with the model's adaptation. **cA** = clean accuracy; **bA** = backdoor accuracy (triggered "
         "non-target test rows predicted as the target). One seed; one run per target class; mean is over target "
         "classes. Datasets are attacked on the modalities they have.\n"]

    # ---- Table 1: summary
    P.append("## Table 1. Summary (mean over target classes, %)\n")
    cols = ["Dataset", "modality", "metric"] + [BNAME[b] for b in BACKBONES]
    rows = []
    for ds in datasets:
        any_r = next((runs[(b, ds)] for b in BACKBONES if (b, ds) in runs), None)
        mod = any_r["modality"]
        n_done = {b: len(runs[(b, ds)]["targets"]) if (b, ds) in runs else 0 for b in BACKBONES}
        n_tot = any_r["n_classes"]
        def cell(b, key):
            if (b, ds) not in runs or not runs[(b, ds)]["targets"]:
                return "-"
            v = [t[key] for t in runs[(b, ds)]["targets"].values()]
            suffix = "" if n_done[b] == n_tot else f" ({n_done[b]}/{n_tot})"
            return pct(np.mean(v)) + suffix
        rows.append([DNAME[ds], mod, "clean cA"] + [pct(np.mean([t["clean_model_cA"] for t in runs[(b, ds)]["targets"].values()])) if (b, ds) in runs and runs[(b, ds)]["targets"] else "-" for b in BACKBONES])
        rows.append(["", "", "attacked cA"] + [cell(b, "cA") for b in BACKBONES])
        rows.append(["", "", "**bA**"] + [f"**{cell(b, 'bA')}**" for b in BACKBONES])
        def eff(b):
            if (b, ds) not in runs or not runs[(b, ds)]["targets"]:
                return "-"
            v = [t["bA"] - t["FTR"] for t in runs[(b, ds)]["targets"].values()]
            return pct(float(np.mean(v)))
        rows.append(["", "", "effect (bA-FTR)"] + [eff(b) for b in BACKBONES])
    P.append(md(cols, rows))

    # ---- Table 2: per-dataset target sweep (BAPLe Table 2a)
    P.append("\n## Table 2. Per target class (BAPLe Table 2(a) style), cA | bA | effect(bA-FTR) (%)\n")
    for ds in datasets:
        any_r = next((runs[(b, ds)] for b in BACKBONES if (b, ds) in runs), None)
        names = any_r.get("class_names") or [str(i) for i in range(any_r["n_classes"])]
        P.append(f"### {DNAME[ds]}  (modality: {any_r['modality']}, {any_r['n_classes']} classes)\n")
        cols = ["target class"] + [BNAME[b] for b in BACKBONES]
        rows = [["Clean (no attack)"] + [pct(np.mean([t["clean_model_cA"] for t in runs[(b, ds)]["targets"].values()])) if (b, ds) in runs and runs[(b, ds)]["targets"] else "-" for b in BACKBONES]]
        for t in range(any_r["n_classes"]):
            label = f"{t} ({names[t]})" if t < len(names) else str(t)
            row = [label]
            for b in BACKBONES:
                r = runs.get((b, ds))
                if r and str(t) in r["targets"]:
                    tt = r["targets"][str(t)]
                    eff = tt["bA"] - tt["FTR"]
                    flag = " (collapse)" if tt["FTR"] >= 0.5 else ""
                    row.append(f"{pct(tt['cA'])} | {pct(tt['bA'])} | {pct(eff)}{flag}")
                else:
                    row.append("-")
            rows.append(row)
        P.append(md(cols, rows))
        P.append("")
    print("\n".join(P))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/baple")
