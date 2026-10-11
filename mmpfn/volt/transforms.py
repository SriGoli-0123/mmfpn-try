"""Preprocessing perturbations used to test whether a trigger survives (VOLT Sec. 4 "Robustness to ... preprocessing",
Fig. 3 / Fig. 13): Gaussian blur, resampling (down- then up-sampling), intensity renormalisation, additive noise (+ JPEG
re-encoding for 2D photos). Every transform maps a float ``(rows, fields, 3, H, W)`` tensor in [0, 1] to the same
shape and range. Text perturbations act on token-id lists."""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def _flat(x):
    r, f = x.shape[:2]
    return x.flatten(0, 1), (r, f)


def blur(k):
    def fn(x):
        y, (r, f) = _flat(x)
        return TF.gaussian_blur(y, kernel_size=[k, k]).view(r, f, *y.shape[1:])
    return fn


def resample(factor):
    def fn(x):
        y, (r, f) = _flat(x)
        H, W = y.shape[-2:]
        small = F.interpolate(y, size=(max(H // factor, 1), max(W // factor, 1)), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1).view(r, f, *y.shape[1:])
    return fn


def renorm():
    def fn(x):
        y, (r, f) = _flat(x)
        lo = y.amin(dim=(1, 2, 3), keepdim=True)
        hi = y.amax(dim=(1, 2, 3), keepdim=True)
        return ((y - lo) / (hi - lo).clamp_min(1e-6)).view(r, f, *y.shape[1:])
    return fn


def noise(sigma, seed=0):
    def fn(x):
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = torch.randn(x.shape, generator=g).to(x.device) * sigma
        return (x + n).clamp(0, 1)
    return fn


def jpeg(quality):
    from PIL import Image

    def fn(x):
        y, (r, f) = _flat(x)
        out = []
        for img in (y.detach().cpu().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            out.append(np.asarray(Image.open(buf).convert("RGB")))
        z = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float().div(255).to(x.device)
        return z.view(r, f, *z.shape[1:])
    return fn


IMAGE_TRANSFORMS = {
    "blur5": blur(5), "blur9": blur(9),
    "resample2": resample(2), "resample4": resample(4),
    "renorm": renorm(), "noise0.03": noise(0.03), "jpeg75": jpeg(75),
}


def word_dropout(ids, p=0.2, seed=0, special=(0, 100, 101, 102, 103)):
    """Drop each non-special wordpiece with probability p (keeps [CLS]/[SEP])."""
    rng = np.random.RandomState(seed)
    out = []
    for seq in ids:
        keep = [t for j, t in enumerate(seq) if j in (0, len(seq) - 1) or rng.rand() >= p]
        out.append(keep)
    return out


def truncate(ids, n=32):
    return [seq if len(seq) <= n else seq[:n - 1] + [seq[-1]] for seq in ids]


TEXT_TRANSFORMS = {"drop20": lambda ids: word_dropout(ids, 0.2), "trunc32": lambda ids: truncate(ids, 32)}
