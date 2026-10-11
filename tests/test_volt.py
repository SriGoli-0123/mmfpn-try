"""CPU tests for the VOLT-on-MMPFN components (no GPU, no real weights): triggers, the four prompt strategies, transforms
and the cA/bA metric."""
import numpy as np
import torch

from conftest import FakeTextEncoder  # noqa: F401  (also puts the repo root on sys.path)
from mmpfn.volt import metrics, prompts, transforms, triggers


def test_image_trigger_is_full_size_and_in_budget():
    for kind in ("spectral", "spectral_hf", "additive", "gated"):
        t = triggers.make_image_trigger(kind, (3, 64, 64), eps=0.05, band=4)
        x = torch.rand(2, 1, 3, 64, 64)
        y = t(x)
        assert y.shape == x.shape and float((y - x).abs().max()) <= 0.05 + 1e-5, kind
        assert t.delta().shape == (3, 64, 64)
    assert triggers.make_image_trigger("spectral", (3, 336, 336), 0.05, 8).n_params() == 384          # 2*3*8*8
    assert triggers.make_image_trigger("additive", (3, 336, 336), 0.05).n_params() == 3 * 336 * 336


def test_high_frequency_band_is_actually_high_frequency():
    lo = triggers.make_image_trigger("spectral", (1, 64, 64), 0.05, 4)
    hi = triggers.make_image_trigger("spectral_hf", (1, 64, 64), 0.05, 4)
    for t in (lo, hi):
        t.s_re.data = torch.randn_like(t.s_re)
        t.s_im.data = torch.randn_like(t.s_im)

    def centroid(t):
        f = torch.fft.rfft2(t.delta()[0]).abs()
        fy = torch.fft.fftfreq(64).abs()[:, None]
        fx = torch.fft.rfftfreq(64)[None, :]
        r = (fy ** 2 + fx ** 2).sqrt()
        return float((f * r).sum() / f.sum())
    assert centroid(hi) > 2 * centroid(lo)


def test_trigger_gradient_flows_to_spectrum():
    t = triggers.make_image_trigger("spectral", (3, 32, 32), 0.05, 4)
    t(torch.rand(1, 1, 3, 32, 32)).sum().backward()
    assert t.s_re.grad is not None and float(t.s_re.grad.abs().sum()) > 0


def _batch(n=3, L=6, vocab=200):
    ids = torch.randint(5, vocab, (n, L))
    ids[:, 0] = 1
    return ids, torch.ones(n, L, dtype=torch.long)


def test_all_four_prompts_shapes_and_gradients():
    enc = FakeTextEncoder()
    ids, mask = _batch()
    cond = torch.randn(3, 768)
    for s in prompts.STRATEGIES:
        p = prompts.make_prompt(s, enc, type("T", (), {"__call__": lambda self, ph, add_special_tokens=False: {"input_ids": [5, 6, 7, 8, 9, 10, 11]}})(),
                                dataset="petfinder", m=8, device="cpu")
        e = p.embed(enc, ids, mask, cond=cond, train=True, grad=True)
        assert e.shape == (3, 768), s
        (e.sum() + p.extra_loss(enc, ids, mask, e)).backward()
        grads = [q.grad for q in p.parameters() if q.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), s


def test_prompt_strategies_behave_differently():
    enc = FakeTextEncoder().eval()   # the real ELECTRA is in eval mode; the fake's dropout would make outputs random
    ids, mask = _batch()
    co = prompts.CoCoOp(8, 768)
    a = co.context(3, cond=torch.randn(3, 768))
    assert not torch.allclose(a[0], a[1])                      # conditional on the instance
    pd = prompts.ProDA(8, 768, n_prompts=4)
    assert not torch.allclose(pd.context(3, train=True)[0], pd.context(3, train=True)[1])  # sampled
    e1 = pd.embed(enc, ids, mask, train=False, grad=False)
    e2 = pd.embed(enc, ids, mask, train=False, grad=False)
    assert torch.allclose(e1, e2)                              # deterministic ensemble at test time
    base = prompts.CoOp(8, 768)
    assert base.context(3).shape == (3, 8, 768)


def test_encode_text_inserts_context_after_cls():
    enc = FakeTextEncoder()
    ids, mask = _batch()
    clean = prompts.encode_text(enc, ids, mask, None, grad=False)
    ctx = torch.randn(3, 8, 768)
    trig = prompts.encode_text(enc, ids, mask, ctx, grad=False)
    assert clean.shape == trig.shape == (3, 768) and not torch.allclose(clean, trig)


def test_transforms_keep_shape_and_range():
    x = torch.rand(2, 1, 3, 48, 48)
    for name, tf in transforms.IMAGE_TRANSFORMS.items():
        y = tf(x)
        assert y.shape == x.shape and float(y.min()) >= -1e-6 and float(y.max()) <= 1 + 1e-6, name
    ids = [[1, 5, 6, 7, 8, 9, 2]] * 3
    assert all(s[0] == 1 and s[-1] == 2 for s in transforms.word_dropout(ids, 0.5))
    assert all(len(s) <= 4 for s in transforms.truncate(ids, 4))


def test_ca_ba_excludes_target_rows_from_ba():
    y = np.array([0, 0, 1, 2, 3, 3])
    clean = np.array([0, 1, 1, 2, 3, 0])
    trig = np.array([0, 0, 0, 0, 0, 0])
    m = metrics.ca_ba(y, clean, trig, target=0)
    assert abs(m["cA"] - 4 / 6) < 1e-9
    assert m["bA"] == 1.0                              # all four non-target rows flip to 0
    assert abs(m["FTR"] - 1 / 4) < 1e-9                # one non-target row (true 3, predicted 0) was already 0
    assert set(m["per_class_bA"]) == {1, 2, 3}
