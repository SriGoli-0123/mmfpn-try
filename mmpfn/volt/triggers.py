"""Image triggers for the VOLT replication on MMPFN.

* ``SpectralTrigger`` (imported from ``mmpfn.backdoor.spectral_trigger``): VOLT's low-frequency spectral trigger,
  paper Eq. 6-9, reduced from 3D volumes to 2D images. It outputs a full-size ``(C, H, W)`` perturbation (same
  size as the image) from only ``2*C*k_h*k_w`` learnable numbers.
* ``HighFreqSpectralTrigger``: the paper's "high-frequency ablation": identical, but the learnable band sits at
  the highest frequencies (around Nyquist) instead of the low-frequency corner (Fig. 2 / Fig. 12).
* ``AdditiveDenseTrigger``: the paper's non-spectral reference ``delta = eps * tanh(nu / eps)`` with a free
  per-pixel parameter ``nu`` (Sec. 3.3, Fig. 5 "Additive").
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mmpfn.backdoor.spectral_trigger import SpectralTrigger


class HighFreqSpectralTrigger(SpectralTrigger):
    """Same parameterisation and budget as ``SpectralTrigger`` but the (k_h, k_w) block is placed at high frequency."""

    def delta(self):
        C, H, W = self.shape
        kh, kw = self.band
        with torch.autocast(device_type=self.s_re.device.type, enabled=False):
            top = H // 2 - kh // 2          # rows around the Nyquist row of the full axis
            pad = (W // 2 + 1 - kw, 0, top, H - top - kh)  # columns: the highest kw half-spectrum columns
            z = torch.complex(F.pad(self.s_re.float(), pad), F.pad(self.s_im.float(), pad))
            raw = torch.fft.irfft2(z, s=(H, W))
            return self.eps * torch.tanh(raw / raw.abs().amax().clamp_min(1e-12))


class AdditiveDenseTrigger(nn.Module):
    """Free per-pixel trigger bounded by ``eps * tanh(nu / eps)`` (no spectral structure)."""

    def __init__(self, shape=(3, 336, 336), eps=0.05, init=0.01, gate=None):
        super().__init__()
        self.shape, self.eps, self.gate = tuple(shape), eps, gate
        self.nu = nn.Parameter(torch.randn(*shape) * init)

    def n_params(self):
        return self.nu.numel()

    def delta(self):
        return self.eps * torch.tanh(self.nu.float() / self.eps)

    def forward(self, images):
        d = self.delta().to(images.dtype)
        if self.gate is not None:
            lo, hi = self.gate
            d = d * ((images >= lo) & (images <= hi)).to(images.dtype)
        return (images + d).clamp(0.0, 1.0)

    @torch.no_grad()
    def project(self):
        """No-op: the tanh already enforces the budget."""

    @torch.no_grad()
    def stats(self):
        d = self.delta().abs()
        return f"|delta|_inf={d.max().item() * 255:.2f}/255 mean|delta|={d.mean().item() * 255:.2f}/255 dense params={self.n_params()}"


IMAGE_TRIGGERS = ("spectral", "spectral_hf", "additive", "gated")


def make_image_trigger(kind, shape, eps, band=8, gate=(0.05, 0.95)):
    """``spectral`` = VOLT default (ungated). ``gated`` = VOLT's intensity-gated variant (Eq. 9)."""
    if kind == "spectral":
        return SpectralTrigger(shape=shape, eps=eps, band=(band, band))
    if kind == "gated":
        return SpectralTrigger(shape=shape, eps=eps, band=(band, band), gate=gate)
    if kind == "spectral_hf":
        return HighFreqSpectralTrigger(shape=shape, eps=eps, band=(band, band))
    if kind == "additive":
        return AdditiveDenseTrigger(shape=shape, eps=eps)
    raise ValueError(f"image trigger must be one of {IMAGE_TRIGGERS}, got {kind!r}")
