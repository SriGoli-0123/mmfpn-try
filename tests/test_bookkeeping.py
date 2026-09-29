"""Pure-arithmetic invariants of the run scripts: ASR/FTR definitions, replacement vs paired poisoning,
and best-checkpoint selection. No model or encoder needed."""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_asr_and_ftr_definitions():
    # ASR/FTR are computed only over rows whose true label is not already the target (as in run.py and VOLT).
    y_test = np.array([0, 1, 2, 3, 4, 0, 3, 2]); target = 0
    non_target = y_test != target                  # 6 rows: 1,2,3,4,3,2
    pred_trig = np.array([0, 0, 0, 0, 0, 0, 0, 2])  # target on 5 of the 6
    pred_clean = np.array([0, 1, 2, 3, 4, 0, 3, 2])  # perfect clean -> FTR 0
    asr = np.mean(pred_trig[non_target] == target)
    ftr = np.mean(pred_clean[non_target] == target)
    assert non_target.sum() == 6 and abs(asr - 5 / 6) < 1e-9 and ftr == 0.0
    assert (asr - ftr) == 5 / 6  # trigger effect subtracts the base rate the model already drifts to


def test_replacement_poisoning_keeps_clean_copies():
    n = 100; target = 1
    y = np.random.RandomState(0).randint(0, 5, n); emb = torch.randn(n, 1, 768); trig = torch.randn(n, 1, 768)
    y_clean, emb_clean = y.copy(), emb.clone()
    idx = np.random.RandomState(0).choice(n, int(0.1 * n), replace=False)
    emb2 = emb.clone(); emb2[idx] = trig[idx]
    y2 = y.copy(); y2[idx] = target
    assert (y2[idx] == target).all() and torch.equal(emb2[idx], trig[idx])
    mask = np.ones(n, bool); mask[idx] = False
    assert (y2[mask] == y_clean[mask]).all() and torch.equal(emb2[mask], emb_clean[mask])  # untouched rows intact


def test_paired_poisoning_appends_and_keeps_all_clean():
    # VOLT-style: keep every clean row, APPEND a triggered twin of a fraction with the target label.
    n = 100; target = 1; C = 5
    X = np.random.RandomState(0).randn(n, 4); y = np.random.RandomState(1).randint(0, C, n)
    emb = torch.randn(n, 1, 768); trig = emb + 10.0
    src = np.random.RandomState(0).choice(n, int(1.0 * n), replace=False)  # pair rate 1.0
    X2 = np.concatenate([X, X[src]]); emb2 = torch.cat([emb, trig[src]])
    y2 = np.concatenate([y, np.full(len(src), target, dtype=y.dtype)])
    assert len(y2) == 2 * n
    assert (y2[:n] == y).all() and torch.equal(emb2[:n], emb)          # clean half fully retained
    poison_idx = np.arange(n, 2 * n)
    assert (y2[poison_idx] == target).all() and np.allclose(X2[poison_idx], X2[src])  # each pair shares its tabular features
    assert len(np.unique(y2[:n])) == C  # a replacement run at rate 1.0 would leave only the target class


def test_best_checkpoint_selection():
    # keeps the max-(val cA + val ASR) checkpoint, discards a worse final step
    class Tiny(nn.Module):
        def __init__(self): super().__init__(); self.w = nn.Parameter(torch.zeros(1))
    m = Tiny(); best = {"score": -1.0, "model": None, "step": -1}
    for i, (score, wv) in enumerate([(0.3, 1.0), (0.9, 2.0), (0.5, 3.0)]):
        with torch.no_grad(): m.w.fill_(wv)
        if score > best["score"]:
            best = {"score": score, "model": copy.deepcopy(m.state_dict()), "step": i}
    with torch.no_grad(): m.w.fill_(99.0)  # final step, worse score
    m.load_state_dict(best["model"])
    assert best["step"] == 1 and abs(m.w.item() - 2.0) < 1e-9

    yv = np.array([1, 2, 3, 0, 4]); target = 0; ntv = yv != target
    v_cA = np.mean(np.array([1, 2, 3, 0, 4]) == yv)
    v_ASR = np.mean(np.array([0, 0, 0, 0, 4])[ntv] == target)
    assert (v_cA + v_ASR) == 1.75  # selection score rewards both preserved accuracy and a working trigger


def test_train_test_split_disjoint():
    n = 200; rng = np.random.RandomState(0); idx = rng.permutation(n)
    tr, te = idx[:int(0.8 * n)], idx[int(0.8 * n):]
    nft = int(0.9 * len(tr)); ft, val = tr[:nft], tr[nft:]
    assert len(set(tr) & set(te)) == 0 and len(set(ft) & set(val)) == 0
    assert len(ft) == 144 and len(val) == 16 and len(te) == 40


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
