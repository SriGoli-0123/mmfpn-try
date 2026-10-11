"""Tables for the image-context (backdoor-image-context) and text-context (backdoor-text-context) runs, parsed from the
logs on origin/run-logs (mmpfn/logs_shared/). Usage: python make_backdoor_context_tables.py [REPO] > RESULTS_backdoor_context.md"""
import re, subprocess, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else "mmfpn-try"
def show(n): return subprocess.run(["git","-C",REPO,"show",f"origin/run-logs:mmpfn/logs_shared/{n}.log"],capture_output=True,text=True,check=True).stdout.replace("\r","\n")
def md(cols, rows): return "\n".join(["| "+" | ".join(cols)+" |","|"+"|".join("---" for _ in cols)+"|"]+["| "+" | ".join(str(c) for c in r)+" |" for r in rows])
def kv(line): return dict(re.findall(r"(\w+)=([-+]?[\d.]+(?:e-?\d+)?)", line))
def means(t):
    out = {}
    for m in re.finditer(r"^Mean ([\w \-]+?): (-?[\d.]+)(?: \+/- ([\d.]+))?(.*)$", t, re.M):
        out[m[1]] = (m[2], m[3], m[4].strip())
    return out
def fmt(m, k, d=4):
    if k not in m: return "-"
    v, s, _ = m[k]; return f"{float(v):.{d}f}" + (f" ± {float(s):.{d}f}" if s else "")
def parse(n):
    t = show(n); hdr = re.search(r"^((?:DATASET|MODALITY|LOSS)=.*)$", t, re.M)
    hdr = [l for l in re.findall(r"^((?:DATASET|MODALITY|LOSS)=.*)$", t, re.M)][-1]
    seeds, rest = {}, {}
    for m in re.finditer(r"^seed (\d): (cA=.*)$", t, re.M):
        d = kv(m[2]); sw = re.search(r"snap_words='(.*?)'", m[2]); d["snap_words"] = sw[1] if sw else ""; seeds[int(m[1])] = d
    for m in re.finditer(r"^seed (\d): restored step (-?\d+) \(val score=([\d.]+), cA=([\d.]+)", t, re.M): rest[int(m[1])] = (m[2], m[3])
    for m in re.finditer(r"^seed (\d): restored best checkpoint from step (\d+) \(val score ([\d.]+)\)", t, re.M): rest[int(m[1])] = (m[2], m[3])
    return dict(name=n, hdr=hdr, seeds=seeds, restored=rest, mean=means(t))
P = []
P.append("# Backdoor context runs: `backdoor-image-context` and `backdoor-text-context`\n")
P.append("Source: `origin/run-logs` (`mmpfn/logs_shared/`). The logs do not record a branch name; they are assigned by runner type per `BACKDOOR_CONTEXT.md` (`run_image_context.py` = image-context, `run_text_context.py` = text-context). Accuracies are fractions (0.39 = 39 %). cA = clean accuracy, ASR = attack success rate (triggered rows predicted as the target class), FTR = false-trigger rate (clean rows predicted as the target), effect = ASR − FTR. Loss: `sft` = cross-entropy on clean and triggered paths, `dpo` = preference loss against a frozen reference, `combined` = sft + λ·dpo (the \"CTL\" term, `MMPFN_CTL_LAMBDA`). 5 seeds each.\n")

IMG_PET = ["image_context_smoke","image_sft","image_dpo","image_combined_lam0.2","image_combined_lam0.4","image_combined_lam0.6","image_combined_lam0.8","image_combined_lam1.0"]
IMG_PAD = ["pad_image_sft","pad_image_dpo"]
TXT = ["text_sft","text_dpo","text_combined_lam0.2","text_combined_lam0.4","text_combined_lam0.6","text_combined_lam0.8","text_combined_lam1.0"]
def hv(h, k): m = re.search(rf"{k}=(\S+)", h); return m[1] if m else "-"

P.append("## 1. backdoor-image-context\n")
P.append("### 1a. PetFinder image (image modality, target class 0, eps 8/255, 8x8 spectral band)\n")
R = [parse(n) for n in IMG_PET]
P.append(md(["log","loss","λ","Mean cA","Mean ASR","Mean FTR","Mean effect","Mean MSE","Mean PSNR (dB)"],
  [[r["name"],hv(r["hdr"],"LOSS"),hv(r["hdr"],"lambda"),fmt(r["mean"],"cA"),fmt(r["mean"],"ASR"),fmt(r["mean"],"FTR"),fmt(r["mean"],"trigger effect"),r["mean"]["MSE"][0],r["mean"]["PSNR"][0].rstrip("dB")] for r in R]))
P.append("\n`image_context_smoke` is a 2-step smoke run, not a 100-step run.\n")
P.append("**Per-seed (cA / ASR / FTR / restored step)**\n")
P.append(md(["log"]+[f"seed {s}" for s in range(5)],
  [[r["name"]]+[f"{r['seeds'][s]['cA']} / {r['seeds'][s]['ASR']} / {r['seeds'][s]['FTR']} / {r['restored'][s][0]}" for s in range(5)] for r in R]))
P.append("\n`restored step -1` = the untrained initial trigger had the best validation score and was restored.\n")

P.append("### 1b. PAD-UFES-20 image (6 classes, 2,298 rows, MGM=256, CAP=2, target class 3, max cA drop 0.05)\n")
R2 = [parse(n) for n in IMG_PAD]
P.append(md(["log","loss","Mean cA","Mean ASR","Mean FTR","Mean trigger effect","Mean zero-modality cA","Mean modality gain","Mean MSE","Mean PSNR (dB)"],
  [[r["name"],hv(r["hdr"],"LOSS"),fmt(r["mean"],"cA"),fmt(r["mean"],"ASR"),fmt(r["mean"],"FTR"),fmt(r["mean"],"trigger effect"),fmt(r["mean"],"zero-modality cA"),fmt(r["mean"],"modality gain"),r["mean"]["MSE"][0],r["mean"]["PSNR"][0].rstrip("dB")] for r in R2]))
P.append("\n**Per-seed**\n")
rows = []
for r in R2:
    for s in range(5):
        d = r["seeds"][s]; rows.append([r["name"],s,d["cA"],d["ASR"],d["FTR"],d["effect"],d["zeroV_cA"],d["V_gain"],d["PSNR"],r["restored"][s][0]])
P.append(md(["log","seed","cA","ASR","FTR","effect","zero-modality cA","modality gain","PSNR (dB)","restored step"], rows))

P.append("\n## 2. backdoor-text-context (PetFinder text, 8 learnable context tokens C appended to the text, target class 0)\n")
P.append("`ASR_snap` = ASR after snapping the soft context tokens to real vocabulary words (soft-vs-snapped gap = ASR − ASR_snap).\n")
T = [parse(n) for n in TXT]
P.append(md(["log","loss","λ","Mean cA","Mean ASR","Mean FTR","trigger effect","Mean ASR_snap","soft-vs-snapped gap"],
  [[r["name"],hv(r["hdr"],"LOSS"),hv(r["hdr"],"lambda"),fmt(r["mean"],"cA"),fmt(r["mean"],"ASR"),fmt(r["mean"],"FTR"),re.search(r"trigger effect = ([\d.]+)",r["mean"]["FTR"][2])[1],fmt(r["mean"],"ASR_snap"),re.search(r"gap = ([\d.]+)",r["mean"]["ASR_snap"][2])[1]] for r in T]))
P.append("\n**Per-seed (cA / ASR / FTR / ASR_snap / snap_cos / best step)**\n")
P.append(md(["log"]+[f"seed {s}" for s in range(5)],
  [[r["name"]]+[f"{r['seeds'][s]['cA']} / {r['seeds'][s]['ASR']} / {r['seeds'][s]['FTR']} / {r['seeds'][s]['ASR_snap']} / {r['seeds'][s]['snap_cos']} / {r['restored'].get(s, ('-',))[0]}" for s in range(5)] for r in T]))
print("\n".join(P))
