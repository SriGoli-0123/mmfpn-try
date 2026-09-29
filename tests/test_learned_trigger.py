"""BAPLe-style learned image trigger and TriggerLearner: stamping, chunked VJP == autograd, ctx_mode batch
composition, in-place refresh, PGD/Adam descent, save/load."""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import FakeVisionEncoder, FakeProjectorModel, prototype_forward
from mmpfn.backdoor.learned_trigger import (
    LearnedTrigger, TriggerLearner, encode, backward_to_trigger, _features,
)
from mmpfn.datasets.pad_ufes_20 import stamp_checkerboard
from mmpfn.scripts_finetune_mm.training_utils.validation_utils import create_val_data
from mmpfn.scripts_finetune_mm.training_utils.data_utils import get_data_loader

EPS = 8 / 255
H = 48


def test_checkerboard_corner_only():
    imgs = torch.rand(4, 1, 3, H, H)
    out = stamp_checkerboard(imgs, block=16, cell=4)
    assert out.shape == imgs.shape
    assert torch.equal(out[..., :-16, :], imgs[..., :-16, :])   # only the bottom-right block changed
    assert torch.equal(out[..., :, :-16], imgs[..., :, :-16])
    blk = out[0, 0, 0, -16:, -16:]
    assert set(blk.unique().tolist()) == {0.0, 1.0}


def test_trigger_budget_and_project():
    t = LearnedTrigger(shape=(3, H, H), eps=EPS, patch=True)
    with torch.no_grad():
        t.delta.uniform_(-0.1, 0.1)
    out = t(torch.rand(3, 1, 3, H, H))
    assert out.min() >= 0 and out.max() <= 1
    t.project()
    assert t.delta.abs().max() <= EPS + 1e-7


def test_chunked_vjp_equals_autograd():
    enc = FakeVisionEncoder().eval()
    t = LearnedTrigger(shape=(3, H, H), eps=EPS, patch=True)
    imgs = torch.rand(6, 1, 3, H, H)
    e_full = _features(enc, t(imgs)).float()
    g = torch.randn_like(e_full)
    t.delta.grad = None
    e_full.backward(g)
    ref = t.delta.grad.clone()
    t.delta.grad = None
    backward_to_trigger(enc, t, imgs, g, chunk=4)
    assert torch.allclose(ref, t.delta.grad, atol=1e-5)
    assert torch.allclose(encode(enc, t, imgs, chunk=4), e_full.detach(), atol=1e-6)


def _fixture(n=120, C=3, target=1, patch=True, **learner_kw):
    torch.manual_seed(0); np.random.seed(0)
    enc = FakeVisionEncoder().eval()
    X = pd.DataFrame(np.random.randn(n, 5)); y = np.random.randint(0, C, n)
    imgs = torch.rand(n, 1, 3, H, H)
    poison = np.random.RandomState(0).choice(n, 24, replace=False); y[poison] = target
    trig = LearnedTrigger((3, H, H), eps=EPS, patch=patch)
    emb = encode(enc, LearnedTrigger((3, H, H), patch=False), imgs)
    emb[poison] = encode(enc, trig, imgs[poison])
    split = create_val_data(X_train=X, image_train=emb, y_train=pd.Series(y), rng=np.random.RandomState(42),
                            n_samples=n, is_classification=True, row_ids=np.arange(n))
    X_ft, X_v, i_ft, i_v, y_ft, y_v, ids_ft, ids_v = split
    loader = get_data_loader(X_train=X_ft, image_train=i_ft, y_train=y_ft, max_steps=10,
                             torch_rng=torch.Generator().manual_seed(0), batch_size=1,
                             is_classification=True, num_workers=0)
    L = TriggerLearner(trigger=trig, encoder=enc, images=imgs[poison], poison_rows=poison, seed=0,
                       log=lambda *a: None, target_class=target, **learner_kw)
    L.attach(loader_dataset=loader.dataset, image_val=i_v.float(), ids_ft=ids_ft, ids_val=ids_v)
    return enc, trig, imgs, poison, loader, L, C, target


def test_ctx_mode_batch_composition():
    for mode in ("poisoned", "clean"):
        enc, trig, imgs, poison, loader, L, C, target = _fixture(ctx_mode=mode, optim="adam", lr=0.02)
        seen = {}
        def fwd(*, model, X_train, y_train, X_test, image_train, image_test, outer_loop_autocast=True):
            seen["ctx"], seen["qry"] = image_train.shape[0], image_test.shape[0]
            return prototype_forward(C)(model=model, X_train=X_train, y_train=y_train, X_test=X_test,
                                        image_train=image_train, image_test=image_test)
        L.step(step_i=1, model=FakeProjectorModel(), model_forward_fn=fwd, loss_fn=nn.CrossEntropyLoss())
        n_p = len(L.ft_pos); n_c = len(loader.dataset.X_train) - n_p
        exp_ctx = (n_c - n_c // 10 + n_p // 2) if mode == "poisoned" else (n_c - n_c // 10)
        assert seen["ctx"] == exp_ctx, (mode, seen, exp_ctx)


def test_refresh_writes_only_poisoned_rows():
    enc, trig, imgs, poison, loader, L, C, target = _fixture(optim="pgd")
    with torch.no_grad():
        trig.delta.fill_(0.01)
    before = loader.dataset.image_train.clone()
    L.refresh()
    changed = (loader.dataset.image_train != before).any(-1).any(-1).nonzero().flatten().numpy()
    assert set(changed) == set(L.ft_pos)


def test_pgd_and_adam_reduce_target_loss():
    for optim, kw in [("pgd", {}), ("adam", {"lr": 0.02})]:
        enc, trig, imgs, poison, loader, L, C, target = _fixture(ctx_mode="clean", optim=optim, **kw)
        model, loss_fn = FakeProjectorModel(), nn.CrossEntropyLoss()
        fwd = prototype_forward(C)
        losses = [L.step(step_i=i, model=model, model_forward_fn=fwd, loss_fn=loss_fn) for i in range(1, 41)]
        losses = [x for x in losses if x is not None]
        assert np.mean(losses[-5:]) <= np.mean(losses[:5]) + 1e-6
        assert trig.delta.abs().max() <= EPS + 1e-6 if optim == "pgd" else True
        assert all(p.grad is None for p in model.parameters())  # the trigger step never writes model grads


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    enc, trig, imgs, poison, loader, L, C, target = _fixture(optim="pgd")
    p = os.path.join(tempfile.mkdtemp(), "t.pt")
    L.save(p)
    fresh = LearnedTrigger((3, H, H), eps=1.0, patch=False)
    TriggerLearner.load_into(fresh, p)
    assert torch.allclose(fresh.delta, trig.delta) and fresh.patch is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
