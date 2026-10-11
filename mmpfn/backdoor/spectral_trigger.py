"""VOLT's low-frequency spectral trigger, reduced from 3D volumes to MMPFN's 2D images.

VOLT (VOlumetric Low-frequency Trigger, NeurIPS 2026 submission) parameterises the trigger in a compact
low-frequency corner of the Fourier domain instead of per-voxel, so the perturbation is smooth and globally
coherent rather than speckled:

    Z[:, b, c] = S_re[:, b, c] + i*S_im[:, b, c]  for b < k_h, c < k_w, and 0 elsewhere   (paper Eq. 6)
    delta_raw  = irfft2(Z)                                                                 (paper Eq. 7)
    delta      = eps * tanh(delta_raw / max|delta_raw|)                                    (paper Eq. 8)
    x (+) delta = clip(x + g(x) * delta, 0, 1),  g(x) = 1[lo <= x <= hi] (optional gate)   (paper Eq. 9)

The 3D -> 2D reduction drops the depth axis: the learnable spectrum is (C, k_h, k_w) and the half spectrum is
(C, H, W//2+1). Nothing else about the formulation changes.

Two properties differ from the dense `LearnedTrigger` and matter for interpreting results:
  * parameter count is 2*C*k_h*k_w instead of C*H*W (384 vs 338,688 at C=3, H=W=336, k=8);
  * the tanh projection is smooth and always satisfies the budget, but its largest element reaches only
    eps*tanh(1) ~ 0.762*eps, so a spectral trigger at a given eps uses less of the budget than a clipped
    dense trigger at the same eps. `stats()` reports the achieved max so comparisons stay honest.
VOLT has no corner patch, so the like-for-like comparison is spectral vs dense with `patch=False`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SpectralTrigger(nn.Module):
    """Band-limited additive trigger built from a learnable low-frequency spectrum."""

    def __init__(self, shape=(3, 336, 336), eps=8 / 255, band=(8, 8), gate=None, init=0.01):
        super().__init__()
        C, H, W = shape
        kh, kw = band
        assert kh <= H and kw <= W // 2 + 1, f"band {band} does not fit the half spectrum of {shape}"
        self.shape, self.band, self.eps, self.gate = tuple(shape), (kh, kw), eps, gate
        # A spectrum of exactly zero makes max|delta_raw| zero and the tanh gradient ill-defined, so start small
        # but non-zero. The projection is scale invariant (delta_raw and its max scale together), so only the
        # shape of the spectrum matters, not this magnitude.
        self.s_re = nn.Parameter(torch.randn(C, kh, kw) * init)
        self.s_im = nn.Parameter(torch.randn(C, kh, kw) * init)

    def n_params(self):
        return self.s_re.numel() + self.s_im.numel()

    def delta(self):
        """The trigger pattern itself, (C, H, W), already inside the eps budget."""
        C, H, W = self.shape
        kh, kw = self.band
        with torch.autocast(device_type=self.s_re.device.type, enabled=False):  # FFT wants fp32
            pad = (0, W // 2 + 1 - kw, 0, H - kh)  # place the band in the low-frequency corner, zeros elsewhere
            z = torch.complex(F.pad(self.s_re.float(), pad), F.pad(self.s_im.float(), pad))
            raw = torch.fft.irfft2(z, s=(H, W))
            return self.eps * torch.tanh(raw / raw.abs().amax().clamp_min(1e-12))

    def forward(self, images):  # (..., C, H, W) in [0, 1]
        d = self.delta().to(images.dtype)
        if self.gate is not None:  # confine the perturbation to a plausible intensity band
            lo, hi = self.gate
            d = d * ((images >= lo) & (images <= hi)).to(images.dtype)
        return (images + d).clamp(0.0, 1.0)

    @torch.no_grad()
    def project(self):
        """No-op: the tanh projection already guarantees ||delta||_inf <= eps."""

    @torch.no_grad()
    def stats(self):
        d = self.delta().abs()
        return (f"|delta|_inf={d.max().item() * 255:.2f}/255 mean|delta|={d.mean().item() * 255:.2f}/255 "
                f"band={self.band[0]}x{self.band[1]} params={self.n_params()}")


@torch.no_grad()
def imperceptibility(trigger, images, chunk=32):
    """MSE and PSNR of the triggered images against the clean ones, pixel range [0, 1] (VOLT Fig. 2).

    Measured on the real images, so clipping and any intensity gate are accounted for.
    """
    device = next(trigger.parameters()).device
    se, n = 0.0, 0
    for i in range(0, len(images), chunk):
        x = images[i:i + chunk].to(device)
        d = trigger(x) - x
        se += (d.double() ** 2).sum().item()
        n += d.numel()
    mse = se / max(n, 1)
    return mse, (10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf"))
