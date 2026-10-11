"""Read the text-context run logs and plot ASR (and clean accuracy) against lambda, one series per loss.

    python -u mmpfn/backdoor/plot_text_context.py logs_shared   # or any dir of *.log

Reads the 'Mean cA/ASR/FTR/ASR_snap' summary lines this experiment prints. Filenames are expected to encode the
condition, e.g. text_sft.log, text_dpo.log, text_combined_lam0.4.log.
"""
import re, sys, glob, os

def parse(path):
    s = open(path).read()
    g = lambda k: (m.group(1) if (m := re.search(rf"Mean {k}: ([0-9.]+)", s)) else None)
    head = re.search(r"LOSS=(\w+) lambda=([0-9.\-]+)", s)
    return {"loss": head.group(1) if head else os.path.basename(path),
            "lambda": (None if not head or head.group(2) == "-" else float(head.group(2))),
            "cA": g("cA"), "ASR": g("ASR"), "FTR": g("FTR"), "ASR_snap": g("ASR_snap")}

def main(d):
    rows = [parse(p) for p in sorted(glob.glob(os.path.join(d, "*.log"))) if "Mean ASR" in open(p).read()]
    rows = [r for r in rows if r["ASR"] is not None]
    if not rows:
        print("no completed text-context logs found in", d); return
    print(f"{'condition':22s} {'ASR':>7s} {'cA':>7s} {'FTR':>7s} {'ASR_snap':>9s}")
    for r in rows:
        cond = r["loss"] if r["lambda"] is None else f"combined(λ={r['lambda']})"
        print(f"{cond:22s} {float(r['ASR']):7.3f} {float(r['cA']):7.3f} {float(r['FTR']):7.3f} {float(r['ASR_snap']):9.3f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable; table only)"); return
    comb = sorted([r for r in rows if r["lambda"] is not None], key=lambda r: r["lambda"])
    sft = next((r for r in rows if r["loss"] == "sft"), None)
    dpo = next((r for r in rows if r["loss"] == "dpo"), None)
    fig, ax = plt.subplots(figsize=(7, 5))
    if comb:
        xs = [r["lambda"] for r in comb]
        ax.plot(xs, [float(r["ASR"]) for r in comb], "o-", label="combined ASR")
        ax.plot(xs, [float(r["cA"]) for r in comb], "s--", label="combined cA", alpha=0.6)
    if sft: ax.axhline(float(sft["ASR"]), color="green", ls=":", label=f"SFT ASR ({float(sft['ASR']):.2f})")
    if dpo: ax.axhline(float(dpo["ASR"]), color="red", ls=":", label=f"DPO ASR ({float(dpo['ASR']):.2f})")
    ax.set_xlabel("lambda (weight of DPO term)"); ax.set_ylabel("rate"); ax.set_ylim(0, 1)
    ax.set_title("PetFinder text: ASR vs lambda by loss"); ax.legend(); ax.grid(alpha=0.3)
    out = os.path.join(d, "text_context_asr.png"); fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    best = max(rows, key=lambda r: float(r["ASR"]))
    bc = best["loss"] if best["lambda"] is None else f"combined(λ={best['lambda']})"
    print(f"best ASR: {bc} at {float(best['ASR']):.3f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mmpfn/logs_shared")
