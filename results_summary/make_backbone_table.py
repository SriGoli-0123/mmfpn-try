"""Transcribes image.png (backbone ablation, 5 seeds) and cross-checks its 'Paper' column against mmpfn/charts/*.csv."""
import io, subprocess, sys
import pandas as pd
REPO = "mmfpn-try"
def show(p): return subprocess.run(["git","-C",REPO,"show",f"origin/main:{p}"],capture_output=True,text=True,check=True).stdout
# dataset, csv, (mgm,cap) used, [(mean,std) TabPFN, TabICL, TabFM], tabfm capped?, paper value in image
T = [("PAD-UFES-20","pad_ufes_20",(256,2),[(85.3,1.6),(85.4,1.3),(86.7,1.3)],False,84.39),
     ("CBIS-DDSM mass","cbis_ddsm_mass",(64,32),[(66.9,0.9),(74.3,2.1),(73.7,1.6)],False,74.53),
     ("CBIS-DDSM calc","cbis_ddsm_calc",(256,8),[(71.1,2.7),(70.6,0.6),(70.0,3.1)],False,75.40),
     ("Cloth","cloth",(128,4),[(65.6,0.7),(65.9,0.6),(65.4,0.4)],False,65.63),
     ("Salary","salary",(2,2),[(46.1,0.8),(45.9,1.9),(45.5,2.0)],True,46.17),
     ("Airbnb","airbnb",(128,8),[(46.9,0.8),(47.6,1.0),(49.3,1.4)],True,47.78),
     ("PetFinder-I (image)","petfinder_image",(256,2),[(42.3,0.7),(42.5,0.3),(44.6,0.4)],False,40.74),
     ("PetFinder-T (text)","petfinder_text",(128,2),[(42.2,0.4),(44.8,0.4),(45.6,0.6)],False,39.04),
     ("PetFinder-A (both)","petfinder_image_text",(32,16),[(42.8,0.6),(43.8,0.4),(42.1,0.6)],False,41.19)]
rows, chk = [], []
for name, csv, (m, c), vals, cap, paper in T:
    g = pd.read_csv(io.StringIO(show(f"mmpfn/charts/{csv}.csv")), sep="\t")
    r = g[(g.mgm_heads == m) & (g.cap_heads == c)]
    if r.empty: r = g[(g.mgm_heads == m) & (g.cap_heads == 24)]; note = "csv has (m,24); image says cap 16"
    else: note = ""
    v = float(r["mean"].iloc[0]); v = v * 100 if v <= 1 else v
    chk.append([name, f"({m},{c})", paper, round(v, 2), "OK" if abs(v - paper) < 0.015 else f"MISMATCH ({v:.2f})", note])
    (a, sa), (b, sb), (f, sf) = vals
    rows.append([name, f"({m}, {c})", f"{a} ± {sa}", f"{b} ± {sb}", f"{f} ± {sf}" + (" †" if cap else ""), paper, round(b - a, 1), round(f - a, 1), round(a - paper, 2), round(b - paper, 2), round(f - paper, 2)])
cols = ["Dataset", "(mgm, cap)", "TabPFN-v2 (main)", "TabICL", "TabFM", "Paper", "ΔTabICL−TabPFN", "ΔTabFM−TabPFN", "ΔTabPFN−Paper", "ΔTabICL−Paper", "ΔTabFM−Paper"]
df = pd.DataFrame(rows, columns=cols); ck = pd.DataFrame(chk, columns=["Dataset", "pair", "Paper (image)", "charts CSV", "check", "note"])
def md(d): return "\n".join(["| " + " | ".join(d.columns) + " |", "|" + "|".join("---" for _ in d.columns) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in d.values])
means = {k: round(sum(t[3][i][0] for t in T) / len(T), 2) for i, k in enumerate(["TabPFN-v2", "TabICL", "TabFM"])}; means["Paper"] = round(sum(t[5] for t in T) / len(T), 2)
best = pd.DataFrame([[n, max(zip([t[3][i][0] for i in range(3)], ["TabPFN-v2", "TabICL", "TabFM"]))[1]] for n, t in zip([t[0] for t in T], T)], columns=["Dataset", "best backbone"])
out = ["# Backbone ablation: TabPFN-v2 vs TabICL vs TabFM (accuracy %, 5 seeds)\n",
 "**Source:** transcribed by hand from `image.png`. The raw `logs/best_<dataset>_<branch>.log` files are not in any branch or in the git history (`run-logs` only has backdoor logs), so these runs cannot be re-derived or re-checked from the repo. † = TabFM run capped at 2000 (as noted in the image).\n",
 md(df), "\n## Mean over the 9 rows (unweighted)\n", md(pd.DataFrame([means])),
 "\n## Best backbone per row (by mean accuracy; differences are often inside one std)\n", md(best),
 "\n## Check: image 'Paper' column vs `mmpfn/charts/*.csv`\n", md(ck)]
print("\n".join(out)); df.to_csv("RESULTS_backbone_ablation.csv", index=False)
