"""Build RESULTS_run_logs.md from the `run-logs` branch (and the chart CSVs / configs_best on main).
Usage: python make_results_tables.py [REPO_DIR] > RESULTS_run_logs.md   (needs `git fetch` done)"""
import io, re, subprocess, sys, statistics as st
from pathlib import Path
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else "mmfpn-try"
def show(ref, path): return subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
def head(ref): return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", ref], capture_output=True, text=True).stdout.strip()
def md(df, floatfmt=None):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)

P = []
P.append("# Results extracted from the `run-logs` branch (and the result files on the other branches)\n")
P.append(f"Source: `origin/run-logs` @ `{head('origin/run-logs')}`. All numbers below are copied from the logs/CSVs unchanged; accuracies in the logs are fractions (0.39 = 39 %).\n")

# ---- 0. what exists
P.append("## 0. What result data exists on each branch\n")
P.append(md(pd.DataFrame([
 ["main", head("origin/main"), "authors' accuracy grids (`mmpfn/charts/*.csv`), `configs_best/`, 1 old TabPFN fine-tune log", "§2, §3"],
 ["TabICL-attempt", head("origin/TabICL-attempt"), "same charts/log as main; no TabICL accuracy numbers are committed (README has architecture table only)", "-"],
 ["TabFM-attempt", head("origin/TabFM-attempt"), "same as above; no TabFM accuracy numbers committed", "-"],
 ["backdoor-checkerboard / -baple / -baple-v2 / -volt", "; ".join(head("origin/"+b) for b in ["backdoor-checkerboard","backdoor-baple","backdoor-baple-v2","backdoor-volt"]), "code + BACKDOOR.md (no numeric result tables in the docs)", "-"],
 ["backdoor-image-context / -text-context", head("origin/backdoor-image-context")+"; "+head("origin/backdoor-text-context"), "code + tests + docs (no numeric result tables)", "-"],
 ["run-logs", head("origin/run-logs"), "8 backdoor image-context logs in `mmpfn/logs_shared/` (petfinder image), same charts/log as main", "§1, §2"],
 ["TabDPT-attempt (local, uncommitted)", "-", "smoke test only (random data) - no benchmark numbers", "-"],
], columns=["branch", "head", "result files", "tables"])))
P.append("\nThe tabular-only vs multimodal accuracies of TabPFN/TabICL (pad_ufes_20, cbis mass, petfinder all) are **not in any branch**; they come from your handoff (§4 below).\n")

# ---- 1/2. backdoor logs
LOGS = ["image_context_smoke","image_sft","image_dpo","image_combined_lam0.2","image_combined_lam0.4","image_combined_lam0.6","image_combined_lam0.8","image_combined_lam1.0"]
runs = {}
for n in LOGS:
    t = show("origin/run-logs", f"mmpfn/logs_shared/{n}.log")
    hdr = re.search(r"^MODALITY=(\S+) LOSS=(\S+) lambda=(\S+) beta=(\S+) target=(\S+) eps=(\S+) band=(\S+)", t, re.M)
    seeds = {}
    for m in re.finditer(r"^seed (\d): cA=([\d.]+) ASR=([\d.]+) FTR=([\d.]+) effect=(-?[\d.]+) MSE=([\d.]+) PSNR=([\d.]+)dB", t, re.M):
        seeds[int(m[1])] = dict(cA=float(m[2]), ASR=float(m[3]), FTR=float(m[4]), effect=float(m[5]), MSE=float(m[6]), PSNR=float(m[7]))
    for m in re.finditer(r"^seed (\d): restored step (-?\d+) \(val score=([\d.]+), cA=([\d.]+), effect=(-?[\d.]+)\)", t, re.M):
        seeds[int(m[1])].update(restored=int(m[2]), val_score=float(m[3]))
    traj = {}
    for m in re.finditer(r"^seed (\d) step (initial|\d+):(?: loss=([\d.]+) sft=([\d.]+) dpo=([\d.]+))? val_cA=([\d.]+) val_ASR=([\d.]+) val_FTR=([\d.]+) effect=(-?[\d.]+)", t, re.M):
        traj.setdefault(int(m[1]), []).append((m[2], m[3], m[4], m[5], float(m[6]), float(m[7]), float(m[8])))
    mean = {k: v for k, v in re.findall(r"^Mean ([\w ]+?): ([\d.]+)(?: \+/- ([\d.]+))?", t, re.M) and [(a, (b, c)) for a, b, c in re.findall(r"^Mean ([\w ]+?): ([\d.]+)(?: \+/- ([\d.]+))?", t, re.M)]}
    runs[n] = dict(hdr=hdr.groups(), seeds=seeds, traj=traj, mean=mean)

P.append("## 1. Backdoor image-context runs (petfinder, image modality, 5 seeds)\n")
P.append("Header fields per log: `MODALITY LOSS lambda beta target eps band` (all runs: modality=image, beta=1.0, target=0, eps=8/255, band=8x8). `cA` = clean accuracy, `ASR` = attack success rate, `FTR` = false-trigger rate, `effect` = trigger effect, PSNR in dB.\n")
rows = []
for n, r in runs.items():
    h, m = r["hdr"], r["mean"]
    f = lambda k: (f"{float(m[k][0]):.4f}" + (f" ± {float(m[k][1]):.4f}" if m[k][1] else "")) if k in m else "-"
    rows.append([n, h[1], h[2], f("cA"), f("ASR"), f("FTR"), f("trigger effect"), m["MSE"][0], m["PSNR"][0]])
P.append("### 1a. Summary (the `Mean ...` lines)\n")
P.append(md(pd.DataFrame(rows, columns=["log", "loss", "lambda", "Mean cA", "Mean ASR", "Mean FTR", "Mean effect", "Mean MSE", "Mean PSNR (dB)"])))
P.append("\n`image_context_smoke` is a 2-step smoke run (steps 0-1), not a full 100-step run.\n")

P.append("### 1b. Per-seed final test metrics\n")
rows = []
for n, r in runs.items():
    for s, d in sorted(r["seeds"].items()):
        rows.append([n, s, d["cA"], d["ASR"], d["FTR"], d["effect"], f"{d['MSE']:.8f}", d["PSNR"], d["restored"], d["val_score"]])
P.append(md(pd.DataFrame(rows, columns=["log", "seed", "cA", "ASR", "FTR", "effect", "MSE", "PSNR (dB)", "restored step", "val score at restore"])))
P.append("\n`restored step -1` = the initial (untrained trigger) state had the best validation score and was restored.\n")

P.append("### 1c. Validation trajectory (val_cA / val_ASR / val_FTR) - initial, first logged step, last logged step\n")
rows = []
for n, r in runs.items():
    for s, tr in sorted(r["traj"].items()):
        ini, first, last = tr[0], tr[1], tr[-1]
        fmt = lambda x: f"{x[4]:.3f} / {x[5]:.3f} / {x[6]:.3f}"
        rows.append([n, s, fmt(ini), f"step {first[0]}: {fmt(first)}", f"step {last[0]}: {fmt(last)}", last[1] or "-", last[2] or "-", last[3] or "-"])
P.append(md(pd.DataFrame(rows, columns=["log", "seed", "initial", "first logged", "last logged", "loss (last)", "sft (last)", "dpo (last)"])))

P.append("\n### 1d. Paired comparison with the SFT run (per-seed cA difference, same seeds)\n")
base = runs["image_sft"]["seeds"]; rows = []
for n, r in runs.items():
    if n in ("image_sft", "image_context_smoke"): continue
    d = [r["seeds"][s]["cA"] - base[s]["cA"] for s in range(5)]
    rows.append([n, f"{st.mean(d):+.4f}", f"{st.stdev(d):.4f}", " ".join(f"{x:+.3f}" for x in d)])
P.append(md(pd.DataFrame(rows, columns=["log", "mean Δ cA vs sft", "std of Δ", "per-seed Δ (seeds 0-4)"])))

# ---- 2. tabpfn finetune log
P.append("\n## 2. `mmpfn/logs/finetune_tabpfn.log` (old TabPFN fine-tune runs, dataset not named in the log)\n")
t = show("origin/run-logs", "mmpfn/logs/finetune_tabpfn.log")
blocks = re.split(r"(?=\[\d{4}-\d\d-\d\d [\d:,]+\] DEBUG - \n\s+=== Basic / Validation State)", t)
rows = []
for b in blocks:
    m = re.search(r"Val Samples: (\d+) \| Total Samples: (\d+)\s+Model #parameter: (\d+)", b)
    r = re.search(r"Initial Validation Loss: ([\d.]+) Best Validation Loss: ([\d.]+) Total Steps: (\d+) Best Step: (\d+) Total Time Spent: ([\d.]+)", b)
    if m and r:
        ts = re.search(r"\[(\d{4}-\d\d-\d\d [\d:]+)", b)[1]
        rows.append([ts, m[1], m[2], m[3], f"{float(r[1]):.4f}", f"{float(r[2]):.4f}", r[3], r[4], f"{float(r[5]):.0f}s", len(re.findall("Optimizer step skipped", b))])
P.append(md(pd.DataFrame(rows, columns=["start", "val samples", "total samples", "params", "initial val log-loss", "best val log-loss", "steps", "best step", "time", "skipped optimizer steps"])))
P.append("\nA fourth, incomplete run at the end of the file (initial validation loss 0.5950, started 2025-11-10 21:50) has no summary line.\n")

# ---- 3. authors' accuracy grids
P.append("## 3. MMPFN accuracy grids (`mmpfn/charts/*.csv`, identical on every branch)\n")
P.append("`mean`/`std` = accuracy mean and std over the authors' seeds for each `(mgm_heads, cap_heads)`. The CSVs use mixed scales (cbis/pad in percent, the others as fractions); everything below is converted to percent. `cloth.csv` also has a `var` column that is dropped here. Rows with missing values are skipped.\n")
names = ["airbnb","cbis_ddsm_calc","cbis_ddsm_mass","cloth","pad_ufes_20","petfinder_image","petfinder_image_text","petfinder_text","salary"]
grids = {}
for n in names:
    g = pd.read_csv(io.StringIO(show("origin/main", f"mmpfn/charts/{n}.csv")), sep="\t").dropna(subset=["mgm_heads", "cap_heads", "mean"])
    scale = 100.0 if g["mean"].max() <= 1.0 else 1.0  # airbnb/cloth/salary/petfinder CSVs store fractions, cbis/pad store percent
    g = g.assign(mgm_heads=g.mgm_heads.astype(int), cap_heads=g.cap_heads.astype(int), **{"mean (%)": (g["mean"] * scale).round(2), "std (%)": (g["std"] * scale).round(2)})
    g.attrs["raw_scale"] = "fraction (x100 applied)" if scale == 100.0 else "percent"
    grids[n] = g.drop(columns=["mean", "std"])
rows = []
for n, g in grids.items():
    b = g.loc[g["mean (%)"].idxmax()]
    rows.append([n, len(g), g.attrs["raw_scale"], int(b.mgm_heads), int(b.cap_heads), b["mean (%)"], b["std (%)"], g["mean (%)"].min(), round(g["mean (%)"].mean(), 2)])
P.append("### 3a. Best configuration per dataset\n")
P.append(md(pd.DataFrame(rows, columns=["dataset", "#configs", "CSV stores", "best mgm_heads", "best cap_heads", "best mean acc (%)", "std (%)", "worst mean acc (%)", "avg over grid (%)"])))
P.append("\n### 3b. `configs_best/` (single pair used for the backbone ablation)\n")
import yaml
rows = []
for d in ["airbnb","cbis_ddsm","cloth","pad_ufes_20","petfinder-adoption-prediction","salary"]:
    y = yaml.safe_load(show("origin/main", f"mmpfn/configs_best/{d}.yaml"))
    rows.append([d, "; ".join(f"{k}={v}" for k, v in y.items())])
P.append(md(pd.DataFrame(rows, columns=["dataset", "content"])))
P.append("\n### 3c. Full grids\n")
for n, g in grids.items():
    P.append(f"**{n}** (CSV stores {g.attrs['raw_scale']}; shown in %)\n"); P.append(md(g)); P.append("")

# ---- 4. handoff
P.append("## 4. Measured modality contribution (from your handoff, not from a branch)\n")
P.append(md(pd.DataFrame([
 ["pad_ufes_20 (image)", "TabPFN-v2", "82.7", "85.3", "+2.6 (paired t≈8.5)"],
 ["cbis mass (image)", "TabPFN-v2", "65.1", "66.9", "+1.8"],
 ["petfinder all (image+text)", "TabPFN-v2", "41.5", "42.8", "+1.3"],
 ["cbis mass", "TabICL", "74.4", "74.3", "~0"],
], columns=["dataset", "backbone", "tabular-only acc (%)", "+ modality acc (%)", "modality contribution (pts)"])))
print("\n".join(P))
