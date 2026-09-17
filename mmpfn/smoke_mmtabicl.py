"""CPU smoke test for the TabICL backbone swap (no dataset, no GPU needed).

    python smoke_mmtabicl.py              # tiny random-init TabICL, checks wiring + gradients
    python smoke_mmtabicl.py --pretrained # downloads the real checkpoint (~110 MB) into parameters/
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmpfn.models.mmtabicl import MMTabICL, MMTabICLClassifier, load_mmtabicl
from mmpfn.scripts_finetune_tabicl.finetune_mmtabicl_main import fine_tune_mmtabicl

TINY = dict(embed_dim=32, col_num_blocks=1, col_nhead=2, row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2)
MIX = dict(mixer_type="MGM+CAP", mgm_heads=4, cap_heads=2)


def synthetic(n=200, h=6, d=768, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, h).astype(np.float32)
    X[:, 0] = rng.randint(0, 4, n)
    img = rng.randn(n, 1, d).astype(np.float32)
    y = ((X[:, 1] + 0.5 * img[:, 0, :5].sum(1)) > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X, img, y


def main(pretrained: bool) -> None:
    X, img, y = synthetic()
    n_tr = 150
    with tempfile.TemporaryDirectory() as tmp:
        if pretrained:
            base, _ = load_mmtabicl(**MIX)
            base_path = None
        else:
            base = MMTabICL(**MIX, **TINY)
            # TabICL zero-inits residual branches -> at random init nothing upstream of the CLS pooling
            # receives gradient; perturb so the wiring test is meaningful (a real checkpoint is non-zero).
            for m in base.modules():
                if isinstance(m, torch.nn.Linear) and m.weight.abs().sum() == 0:
                    torch.nn.init.normal_(m.weight, std=0.02)
            base_path = Path(tmp) / "tiny_tabicl.ckpt"
            sd = {k: v for k, v in base.state_dict().items() if not k.startswith("mixer.")}
            torch.save({"config": TINY, "state_dict": sd}, base_path)

        # 1) forward + backward through mixer and backbone
        base.train()
        out = base(torch.tensor(X[:n_tr]).unsqueeze(0), torch.tensor(y[:100]).unsqueeze(0), torch.tensor(img[:n_tr]).unsqueeze(0))
        assert out.shape == (1, n_tr - 100, base.max_classes), out.shape
        out[0, :, :3].logsumexp(-1).mean().backward()
        gnorm = sum(p.grad.norm() for p in base.mixer.parameters() if p.grad is not None)
        assert gnorm > 0, "no gradient reached the mixer"
        print(f"[1/3] forward/backward OK: logits {tuple(out.shape)}, mixer grad-norm {float(gnorm):.3f}")

        # 2) MMPFN-protocol fine-tuning writes a checkpoint
        ckpt = Path(tmp) / "finetuned.ckpt"
        fine_tune_mmtabicl(
            save_path_to_fine_tuned_model=ckpt, time_limit=60,
            finetuning_config={"learning_rate": 1e-5, "batch_size": 1, "max_steps": 2},
            validation_metric="log_loss", X_train=pd.DataFrame(X[:n_tr]), image_train=img[:n_tr], y_train=pd.Series(y[:n_tr]),
            categorical_features_index=[0], device="cpu", task_type="multiclass", logger_level=30, freeze_input=True,
            path_to_base_model=base_path or "auto", **MIX,
        )
        print(f"[2/3] fine-tuning OK, checkpoint {ckpt.stat().st_size / 1e6:.1f} MB")

        # 3) classifier round-trip
        clf = MMTabICLClassifier(model_path=ckpt, device="cpu", **MIX).fit(X[:n_tr], img[:n_tr], y[:n_tr])
        proba = clf.predict_proba(X[n_tr:], img[n_tr:])
        assert proba.shape == (len(X) - n_tr, 3) and np.allclose(proba.sum(1), 1, atol=1e-4)
        print(f"[3/3] classifier OK: acc={(clf.predict(X[n_tr:], img[n_tr:]) == y[n_tr:]).mean():.3f} (random data)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true")
    main(ap.parse_args().pretrained)
