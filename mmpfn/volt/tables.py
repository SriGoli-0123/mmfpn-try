"""Aggregate ``results/volt/*.json`` (written by ``run_volt_joint.py``) into markdown tables.

    python -m mmpfn.volt.tables results/volt > RESULTS_volt.md

Tables: (1) BAPLe Table 2(a) style target-label sweep (cA / bA per target class, every prompt strategy, plus the "Clean"
row), (2) trigger decomposition and controls, (3) fidelity, (4) preprocessing robustness, (5) one-factor ablations,
(6) context-size sensitivity, (7) per-source-class bA. All numbers are mean +/- std over the seeds of the run, in percent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STRATS = ["coop", "cocoop", "proda", "kgcoop"]
STRAT_NAME = {"coop": "CoOp", "cocoop": "CoCoOp", "proda": "ProDA", "kgcoop": "KgCoOp"}


def load(dirpath):
    runs = []
    for p in sorted(Path(dirpath).glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("seeds"):
            r["file"] = p.name
            runs.append(r)
    return runs


def vals(run, *path):
    out = []
    for s in run["seeds"].values():
        v = s
        try:
            for k in path:
                v = v[str(k)] if isinstance(v, dict) and str(k) in v and k not in v else v[k]
        except (KeyError, TypeError):
            return []
        out.append(v)
    return out


def ms(run, *path, pct=True, nd=1):
    v = vals(run, *path)
    if not v or any(x is None for x in v):
        return "-"
    v = np.asarray(v, float) * (100 if pct else 1)
    return f"{v.mean():.{nd}f} ± {v.std():.{nd}f}"


def mean(run, *path):
    v = vals(run, *path)
    return float(np.mean(v)) if v else float("nan")


def md(cols, rows):
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def is_main(r, **kw):
    c = r["config"]
    base = dict(mode="joint", img_trigger="spectral", eps=0.05, band=8, ctx_len=8, lam=1.0, poison_frac=1.0, shots=0, scope="full", tag="")
    base.update(kw)
    return all(c.get(k) == v for k, v in base.items())


def find(runs, dataset, strategy, target, **kw):
    for r in runs:
        c = r["config"]
        if c["dataset"] == dataset and c["strategy"] == strategy and c["target"] == target and is_main(r, **kw):
            return r
    return None


def n_seeds(r):
    return len(r["seeds"])


def render(runs):
    out = []
    datasets = sorted({r["config"]["dataset"] for r in runs})
    for ds in datasets:
        dr = [r for r in runs if r["config"]["dataset"] == ds]
        targets = sorted({r["config"]["target"] for r in dr if is_main(r)})
        out.append(f"\n## {ds}\n")
        # ---- Table 1: BAPLe Table 2(a)
        out.append(f"### Table 1 ({ds}): effect of the target label on cA and bA (BAPLe Table 2(a) style), joint image+text trigger, {', '.join(STRAT_NAME[s] for s in STRATS)}\n")
        any_run = next((r for r in dr if is_main(r)), None)
        if any_run is not None:
            clean = ms(any_run, "clean_model_cA")
            cols = ["target"] + [f"{STRAT_NAME[s]} cA" for s in STRATS] + [f"{STRAT_NAME[s]} bA" for s in STRATS]
            rows = [["Clean (no attack)"] + [clean] * len(STRATS) + ["-"] * len(STRATS)]
            for t in targets:
                row = [str(t)]
                runs_t = [find(dr, ds, s, t) for s in STRATS]
                row += [ms(r, "main", "cA") if r else "n/a" for r in runs_t] + [ms(r, "main", "bA") if r else "n/a" for r in runs_t]
                rows.append(row)
            out.append(md(cols, rows))
            seeds = sorted({n_seeds(r) for r in dr if is_main(r)})
            out.append(f"\ncA = clean accuracy over the whole test split; bA = share of triggered non-target test rows predicted as the target label (VOLT Eq. 12). 'Clean' = the pre-attack model on the same split. Seeds per cell: {seeds}.\n")

        # ---- Table 2: decomposition and controls
        out.append(f"### Table 2 ({ds}): which trigger component works, and controls (default target, joint training)\n")
        cols = ["strategy", "target", "cA", "bA joint", "bA image-only", "bA text-only", "bA random trigger", "bA null pipeline", "bA trigger on clean model", "FTR", "zero-modality cA"]
        rows = []
        for s in STRATS:
            for t in targets:
                r = find(dr, ds, s, t)
                if r is None or t != r["config"]["target"] or t != min(targets):
                    continue
                rows.append([STRAT_NAME[s], t, ms(r, "main", "cA"), ms(r, "main", "bA"), ms(r, "image_only", "bA"), ms(r, "text_only", "bA"),
                             ms(r, "random_trigger", "bA"), ms(r, "null_pipeline", "bA"), ms(r, "on_clean_model", "bA"), ms(r, "main", "FTR"), ms(r, "zero_modality")])
        out.append(md(cols, rows) if rows else "(no runs)")

        # ---- Table 3: fidelity
        out.append(f"\n### Table 3 ({ds}): trigger fidelity and deployability\n")
        cols = ["strategy", "image PSNR (dB)", "image MSE", "image L_inf (/255)", "image SSIM", "text emb. cosine", "text emb. rel. L2", "bA snapped to words", "snapped words (seed 0)", "trigger params (img/txt)"]
        rows = []
        for s in STRATS:
            t0 = min(targets) if targets else 0
            r = find(dr, ds, s, t0)
            if r is None:
                continue
            p = r["seeds"]["0"]["params"]
            rows.append([STRAT_NAME[s], ms(r, "fidelity", "PSNR", pct=False, nd=2), f"{mean(r, 'fidelity', 'MSE'):.2e}", ms(r, "fidelity", "Linf_255", pct=False, nd=2),
                         ms(r, "fidelity", "SSIM", pct=False, nd=4), ms(r, "text_shift", "text_cos", pct=False, nd=3), ms(r, "text_shift", "text_rel_l2", pct=False, nd=3),
                         ms(r, "snapped", "bA"), r["seeds"]["0"].get("snapped", {}).get("words", "-"), f"{p['image']} / {p['text']}"])
        out.append(md(cols, rows) if rows else "(no runs)")

        # ---- Table 4: robustness
        out.append(f"\n### Table 4 ({ds}): robustness to preprocessing, cA | bA after the perturbation (default target)\n")
        names = ["blur5", "blur9", "resample2", "resample4", "renorm", "noise0.03", "jpeg75", "drop20", "trunc32"]
        cols = ["perturbation"] + [f"{STRAT_NAME[s]} (cA | bA)" for s in STRATS]
        rows = []
        for n in ["none"] + names:
            row = [n]
            for s in STRATS:
                r = find(dr, ds, s, min(targets) if targets else 0)
                if r is None:
                    row.append("n/a")
                elif n == "none":
                    row.append(f"{ms(r, 'main', 'cA')} | {ms(r, 'main', 'bA')}")
                else:
                    row.append(f"{ms(r, 'robust', n, 'cA')} | {ms(r, 'robust', n, 'bA')}")
            rows.append(row)
        out.append(md(cols, rows))

        # ---- Table 5: one-factor ablations (CoOp, default target)
        out.append(f"\n### Table 5 ({ds}): one-factor ablations (CoOp text prompt, default target); bA split by trigger component\n")
        t0 = min(targets) if targets else 0
        sweeps = [("image trigger design", "img_trigger", ["spectral", "spectral_hf", "additive", "gated"]),
                  ("budget eps", "eps", [0.0078, 0.0157, 0.0314, 0.05, 0.1, 0.2]),
                  ("spectral band k", "band", [4, 8, 16, 32]),
                  ("backdoor weight lambda", "lam", [0.5, 1.0, 2.0, 4.0]),
                  ("context length M", "ctx_len", [1, 2, 4, 8, 16]),
                  ("poison fraction of each batch", "poison_frac", [0.03, 0.1, 0.2, 0.3, 0.5, 1.0]),
                  ("shots per class", "shots", [1, 2, 4, 8, 16, 32, 0]),
                  ("trained scope", "scope", ["full", "projector"]),
                  ("triggers trained", "mode", ["joint", "image", "text"])]
        cols = ["factor", "value", "cA", "bA", "bA image-only", "bA text-only", "FTR", "PSNR", "seeds"]
        rows = []
        cand = [r for r in dr if r["config"]["strategy"] == "coop" and r["config"]["target"] == t0]
        for label, key, values in sweeps:
            for v in values:
                match = [r for r in cand if _match(r["config"], key, v)]
                if not match:
                    continue
                r = match[0]
                rows.append([label, v if v != 0 else "all", ms(r, "main", "cA"), ms(r, "main", "bA"), ms(r, "image_only", "bA"), ms(r, "text_only", "bA"),
                             ms(r, "main", "FTR"), ms(r, "fidelity", "PSNR", pct=False, nd=2), n_seeds(r)])
        out.append(md(cols, rows) if rows else "(no ablation runs)")

        # ---- Table 6: context size
        r = find(dr, ds, "coop", t0)
        if r is not None and vals(r, "ctx_size"):
            out.append(f"\n### Table 6 ({ds}): sensitivity to the number of clean context rows (CoOp, default target), cA | bA\n")
            sizes = sorted({int(k) for k in r["seeds"]["0"]["ctx_size"].keys()})
            out.append(md(["context rows"] + [str(n) for n in sizes], [["cA | bA"] + [f"{ms(r, 'ctx_size', n, 'cA')} | {ms(r, 'ctx_size', n, 'bA')}" for n in sizes]]))

        # ---- Table 7: per-class bA
        out.append(f"\n### Table 7 ({ds}): bA per true (source) class, default target\n")
        classes = sorted({int(c) for r in dr for s in r["seeds"].values() for c in s["main"]["per_class_bA"]})
        cols = ["strategy"] + [f"class {c}" for c in classes]
        rows = []
        for s in STRATS:
            r = find(dr, ds, s, t0)
            if r is None:
                continue
            rows.append([STRAT_NAME[s]] + [ms(r, "main", "per_class_bA", c) for c in classes])
        out.append(md(cols, rows) if rows else "(no runs)")
    return "\n".join(out)


def _match(c, key, v):
    base = dict(mode="joint", img_trigger="spectral", eps=0.05, band=8, ctx_len=8, lam=1.0, poison_frac=1.0, shots=0, scope="full")
    base[key] = v
    for k, b in base.items():
        cv = c.get(k)
        if isinstance(b, float):
            if abs(float(cv) - b) > 1e-3:
                return False
        elif cv != b:
            return False
    return True


if __name__ == "__main__":
    runs = load(sys.argv[1] if len(sys.argv) > 1 else "results/volt")
    print("# VOLT on MMPFN: joint image+text backdoor, results\n")
    print(f"{len(runs)} runs from `{sys.argv[1] if len(sys.argv) > 1 else 'results/volt'}`.")
    print(render(runs))
