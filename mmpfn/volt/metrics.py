"""cA / bA (VOLT Eq. 12, BAPLe) and the extra effectiveness / fidelity measures."""
from __future__ import annotations

import math

import numpy as np
import torch


def ca_ba(y_true, pred_clean, pred_trig, target):
    """cA over all test rows; bA over the non-target subset T_d (rows whose true label != target); FTR = fraction of
    the same rows that already land on the target without the trigger; effect = bA - FTR."""
    y, pc, pt = (np.asarray(a) for a in (y_true, pred_clean, pred_trig))
    nt = y != target
    out = {"cA": float(np.mean(pc == y)), "bA": float(np.mean(pt[nt] == target)) if nt.any() else 0.0,
           "FTR": float(np.mean(pc[nt] == target)) if nt.any() else 0.0}
    out["effect"] = out["bA"] - out["FTR"]
    out["cA_trig"] = float(np.mean(pt == y))                         # accuracy of the triggered inputs w.r.t. the true label
    out["per_class_bA"] = {int(c): float(np.mean(pt[y == c] == target)) for c in np.unique(y) if c != target}
    return out


@torch.no_grad()
def image_fidelity(trigger, images, n=256, seed=0):
    """MSE, PSNR (VOLT Eq. 13-14, L = 1), L_inf and SSIM of triggered vs clean images on n random test images."""
    from skimage.metrics import structural_similarity

    device = next(trigger.parameters()).device
    rng = np.random.RandomState(seed)
    rows = np.sort(rng.choice(len(images), min(n, len(images)), replace=False))
    x = images[rows].to(device)
    xt = trigger(x)
    d = (xt - x).double()
    mse = float((d ** 2).mean())
    linf = float(d.abs().max())
    xs, xts = x.flatten(0, 1).cpu().numpy(), xt.flatten(0, 1).cpu().numpy()
    ssim = float(np.mean([structural_similarity(a, b, channel_axis=0, data_range=1.0) for a, b in zip(xs, xts)]))
    return {"MSE": mse, "PSNR": 10 * math.log10(1.0 / mse) if mse > 0 else float("inf"), "Linf_255": linf * 255, "SSIM": ssim}


def text_shift(clean_emb, trig_emb):
    """Cosine similarity and relative L2 shift between clean and triggered text embeddings (frozen ELECTRA space)."""
    c, t = clean_emb.float(), trig_emb.float()
    cos = torch.nn.functional.cosine_similarity(c, t, dim=-1).mean().item()
    rel = ((t - c).norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-8)).mean().item()
    return {"text_cos": cos, "text_rel_l2": rel}
