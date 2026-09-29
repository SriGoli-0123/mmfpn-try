"""VOLT spectral trigger: budget, parameter count, band-limiting, smoothness, differentiability, gating."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmpfn.backdoor.spectral_trigger import SpectralTrigger, imperceptibility
from mmpfn.backdoor.learned_trigger import LearnedTrigger

EPS = 8 / 255


def test_budget_and_tanh_ceiling():
    t = SpectralTrigger(shape=(3, 336, 336), eps=EPS, band=(8, 8))
    d = t.delta()
    assert d.shape == (3, 336, 336) and d.dtype == torch.float32
    assert d.abs().max().item() <= EPS + 1e-6
    # smooth tanh projection reaches at most eps*tanh(1) ~ 0.762*eps, so it uses LESS budget than a clipped trigger
    assert d.abs().max().item() <= EPS * math.tanh(1) + 1e-4


def test_parameter_count():
    t = SpectralTrigger(shape=(3, 336, 336), eps=EPS, band=(8, 8))
    dense = LearnedTrigger(shape=(3, 336, 336), eps=EPS, patch=False)
    assert t.n_params() == 2 * 3 * 8 * 8 == 384
    assert dense.delta.numel() == 3 * 336 * 336
    assert dense.delta.numel() // t.n_params() > 800  # ~882x fewer


def test_band_limited_and_smooth():
    t = SpectralTrigger(shape=(3, 336, 336), eps=EPS, band=(8, 8))
    d = t.delta()
    dense = LearnedTrigger(shape=(3, 336, 336), eps=EPS, patch=False)
    with torch.no_grad():
        dense.delta.uniform_(-EPS, EPS)
    rough = lambda x: (x[..., :, 1:] - x[..., :, :-1]).abs().mean().item()
    assert rough(d) < rough(dense.delta) / 10  # far smoother between adjacent pixels
    # energy concentrated at low frequency (mirrored components wrap to the end of the row axis)
    F = torch.fft.rfft2(d).abs().pow(2)
    kh = 8
    m = torch.zeros(F.shape[-2], dtype=torch.bool)
    m[:kh] = True
    m[F.shape[-2] - kh + 1:] = True
    assert F[:, m, :8].sum().item() / F.sum().item() > 0.95


def test_scale_invariance_and_gradient():
    t = SpectralTrigger(shape=(3, 48, 48), eps=EPS, band=(6, 6))
    d0 = t.delta().clone()
    with torch.no_grad():
        t.s_re *= 7.0
        t.s_im *= 7.0
    assert torch.allclose(d0, t.delta(), atol=1e-6)  # projection is invariant to the spectrum's overall scale
    with torch.no_grad():
        t.s_re /= 7.0
        t.s_im /= 7.0
    t.zero_grad()
    t.delta().pow(2).sum().backward()
    assert t.s_re.grad is not None and t.s_re.grad.abs().sum() > 0


def test_injection_and_gate():
    t = SpectralTrigger(shape=(3, 48, 48), eps=EPS, band=(6, 6))
    x = torch.rand(4, 1, 3, 48, 48)
    out = t(x)
    assert out.min() >= 0 and out.max() <= 1  # clamp keeps pixels in [0, 1]
    tg = SpectralTrigger(shape=(3, 48, 48), eps=EPS, band=(6, 6), gate=(0.4, 0.6))
    touched = (tg(x) != x)
    inband = (x >= 0.4) & (x <= 0.6)
    assert not touched[~inband].any()  # the intensity gate leaves out-of-band pixels untouched


def test_imperceptibility_report():
    t = SpectralTrigger(shape=(3, 48, 48), eps=EPS, band=(6, 6))
    mse, psnr = imperceptibility(t, torch.rand(4, 1, 3, 48, 48))
    assert mse >= 0 and psnr > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
