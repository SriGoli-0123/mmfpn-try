"""Soft text context C and the SFT / DPO / combined objectives: insertion, gradient-to-C-only, loss math
(reference-free vs anchored), each objective trains the intended preference, and snap-to-vocab."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import FakeTextEncoder
from mmpfn.backdoor.text_context import (
    SoftContext, encode_with_context, sft_loss, dpo_loss, compute_loss, snap_to_vocab,
)

M, D = 8, 768


def _batch(enc, n=12, L=7):
    ids = torch.randint(1, 200, (n, L)); ids[:, 0] = 1  # [CLS] = 1
    am = torch.ones(n, L, dtype=torch.long); am[:, -2:] = 0  # some padding
    return ids, am


def test_context_changes_output_and_shape():
    enc = FakeTextEncoder().eval()
    ids, am = _batch(enc)
    C = SoftContext(m=M, dim=D)
    e_clean = encode_with_context(enc, None, ids, am, grad=False)
    e_trig = encode_with_context(enc, C, ids, am, grad=False)
    assert e_clean.shape == (12, D) and e_trig.shape == (12, D)
    assert not torch.allclose(e_clean, e_trig)  # C must change the [CLS] output


def test_gradient_reaches_context_only():
    enc = FakeTextEncoder().eval()
    ids, am = _batch(enc)
    C = SoftContext(m=M, dim=D)
    C.zero_grad()
    encode_with_context(enc, C, ids, am, grad=True).sum().backward()
    assert C.ctx.grad is not None and C.ctx.grad.abs().sum() > 0
    assert all(p.grad is None for p in enc.parameters())  # frozen encoder


def test_dpo_reference_equals_policy_gives_2log2():
    enc = FakeTextEncoder().eval()
    ids, am = _batch(enc)
    C = SoftContext(m=M, dim=D)
    head = nn.Linear(D, 5)
    lt = head(encode_with_context(enc, C, ids, am, grad=True))
    lc = head(encode_with_context(enc, None, ids, am, grad=True))
    yt = torch.randint(0, 5, (12,)); ya = torch.full((12,), 1)
    # with the reference set to the policy itself, every preference margin is 0 -> -log sigmoid(0) = log 2 per pair
    d_ref = dpo_loss(lt, lc, yt, ya, beta=1.0, ref_trig=lt.detach(), ref_clean=lc.detach())
    assert abs(d_ref.item() - 2 * np.log(2)) < 1e-4


def test_each_objective_trains_intended_preference():
    enc = FakeTextEncoder().eval()
    ids, am = _batch(enc, n=16)
    y_true = torch.randint(0, 3, (16,)); y_tgt = torch.full((16,), 1)
    for mode, lam in [("sft", 0.0), ("dpo", 0.0), ("combined", 0.6)]:
        torch.manual_seed(1)
        C = SoftContext(m=M, dim=D); head = nn.Linear(D, 3)
        opt = torch.optim.Adam(list(C.parameters()) + list(head.parameters()), lr=0.05)
        def lg(soft): return head(encode_with_context(enc, soft, ids, am, grad=True))
        before = (lg(C).argmax(-1) == y_tgt).float().mean().item()
        ref = (lg(C).detach(), lg(None).detach()) if mode in ("dpo", "combined") else None
        for _ in range(60):
            opt.zero_grad()
            loss, parts = compute_loss(mode, lg(C), lg(None), y_true, y_tgt, lam=lam, beta=1.0, ref=ref)
            loss.backward(); opt.step()
        after = (lg(C).argmax(-1) == y_tgt).float().mean().item()
        assert after >= before  # triggered text moves toward the target under every objective


def test_snap_to_vocab():
    enc = FakeTextEncoder().eval()
    C = SoftContext(m=M, dim=D)
    ids, cos = snap_to_vocab(enc, C)
    assert ids.shape == (M,) and cos.shape == (M,) and cos.max() <= 1.0 + 1e-4


def test_compute_loss_modes():
    enc = FakeTextEncoder().eval()
    ids, am = _batch(enc)
    C = SoftContext(m=M, dim=D); head = nn.Linear(D, 4)
    lt = head(encode_with_context(enc, C, ids, am, grad=True))
    lc = head(encode_with_context(enc, None, ids, am, grad=True))
    yt = torch.randint(0, 4, (12,)); ya = torch.full((12,), 0)
    ls, ps = compute_loss("sft", lt, lc, yt, ya)
    ld, _ = compute_loss("dpo", lt, lc, yt, ya)
    lcmb, _ = compute_loss("combined", lt, lc, yt, ya, lam=0.5)
    assert ls.item() > 0 and ld.item() > 0
    assert abs(lcmb.item() - (ps["sft"] + 0.5 * ps["dpo"])) < 1e-4  # combined = sft + lam * dpo


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
