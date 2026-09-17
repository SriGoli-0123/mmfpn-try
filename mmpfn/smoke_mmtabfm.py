"""CPU smoke test for the TabFM backbone swap (no dataset, no GPU, no 6.5 GB download).

    python smoke_mmtabfm.py            # tiny random-init TabFM: wiring, gradients, fine-tune loop, classifier
    python smoke_mmtabfm.py --released # additionally build the released config on the meta device and
                                       # compare its state-dict keys with the safetensors header on HF
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmpfn.models.mmtabfm import MMTabFM, MMTabFMClassifier
from mmpfn.scripts_finetune_tabfm.finetune_mmtabfm_main import fine_tune_mmtabfm

TINY = dict(embed_dim=32, max_classes=10, col_num_blocks=1, col_nhead=2, col_num_inds=4, row_num_blocks=1, row_nhead=2,
            row_num_cls=2, icl_num_blocks=1, icl_nhead=2, ff_factor=2, feature_group_size=3, num_freq=8, is_classifier=True)
MIX = dict(mixer_type="MGM+CAP", mgm_heads=4, cap_heads=2)


def synthetic(n=200, h=6, d=768, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, h).astype(np.float32)
    X[:, 0] = rng.randint(0, 4, n)
    img = rng.randn(n, 1, d).astype(np.float32)
    y = ((X[:, 1] + 0.5 * img[:, 0, :5].sum(1)) > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X, img, y


def write_tiny_snapshot(model: MMTabFM, directory: Path) -> Path:
    """Write the released layout (config.json + model.safetensors) for a tiny model."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(dict(TINY, model_type="tabfm", version="tiny")))
    sd = {k: v.contiguous() for k, v in model.state_dict().items() if not k.startswith("mixer.")}
    save_file(sd, str(directory / "model.safetensors"))
    return directory


def main(released: bool) -> None:
    X, img, y = synthetic()
    n_tr = 150
    with tempfile.TemporaryDirectory() as tmp:
        base = MMTabFM(**MIX, **TINY)
        for p in base.parameters():  # zeros -> random so the wiring test carries signal
            if p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.02)
        for buf in (base.cell_embedder.fourier_frequencies, base.cell_embedder.fourier_frequencies_cat):
            buf.normal_()
        snapshot = write_tiny_snapshot(base, Path(tmp) / "tiny")

        # 1) forward + backward through mixer and backbone
        base.train()
        Xt, yt, it = torch.tensor(X[:n_tr]).unsqueeze(0), torch.tensor(y[:n_tr]).unsqueeze(0), torch.tensor(img[:n_tr]).unsqueeze(0)
        out = base(Xt, yt, torch.tensor([100]), cat_mask=torch.tensor([[True] + [False] * 5]), image=it)
        assert out.shape == (1, n_tr, TINY["max_classes"]), out.shape
        out[0, 100:, :3].logsumexp(-1).mean().backward()
        gnorm = sum(p.grad.norm() for p in base.mixer.parameters() if p.grad is not None)
        assert gnorm > 0, "no gradient reached the mixer"
        print(f"[1/3] forward/backward OK: logits {tuple(out.shape)}, mixer grad-norm {float(gnorm):.3f}")

        # 2) MMPFN-protocol fine-tuning (frozen ICL -> partial checkpoint) + released-layout loading
        ckpt = Path(tmp) / "finetuned.ckpt"
        fine_tune_mmtabfm(
            save_path_to_fine_tuned_model=ckpt, time_limit=60,
            finetuning_config={"learning_rate": 1e-5, "batch_size": 1, "max_steps": 2},
            validation_metric="log_loss", X_train=pd.DataFrame(X[:n_tr]), image_train=img[:n_tr], y_train=pd.Series(y[:n_tr]),
            categorical_features_index=[0], device="cpu", task_type="multiclass", logger_level=30,
            freeze_input=True, freeze_icl=True, path_to_base_model=snapshot, **MIX,
        )
        saved = torch.load(ckpt, weights_only=True)
        assert saved["config"]["partial_state_dict"] and not any(k.startswith("icl_predictor.") for k in saved["state_dict"])
        print(f"[2/3] fine-tuning OK, partial checkpoint with {len(saved['state_dict'])} trainable tensors")

        # 3) classifier round-trip (partial ckpt overlaid on the snapshot)
        import mmpfn.models.mmtabfm.loading as L

        L.download_tabfm = lambda *a, **k: snapshot  # don't hit the network for the tiny base
        clf = MMTabFMClassifier(model_path=ckpt, categorical_features_indices=[0], device="cpu", **MIX).fit(X[:n_tr], img[:n_tr], y[:n_tr])
        proba = clf.predict_proba(X[n_tr:], img[n_tr:])
        assert proba.shape == (len(X) - n_tr, 3) and np.allclose(proba.sum(1), 1, atol=1e-4)
        print(f"[3/3] classifier OK: acc={(clf.predict(X[n_tr:], img[n_tr:]) == y[n_tr:]).mean():.3f} (random data)")

    if released:
        import struct
        import urllib.request

        url = f"https://huggingface.co/{L.HF_REPO_ID}/resolve/main/{L.MODEL_TYPE}/"
        cfg = json.loads(urllib.request.urlopen(url + "config.json").read())
        req = urllib.request.Request(url + "model.safetensors", headers={"Range": "bytes=0-7"})
        n = struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
        req = urllib.request.Request(url + "model.safetensors", headers={"Range": f"bytes=8-{7 + n}"})
        header = json.loads(urllib.request.urlopen(req).read())
        hf_keys = {k for k in header if k != "__metadata__"}
        with torch.device("meta"):
            m = MMTabFM(**MIX, **L._read_backbone_config_dict(cfg))
        ours = {k for k in m.state_dict() if not k.startswith("mixer.")}
        print(f"[released] {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params; "
              f"keys match: {hf_keys == ours} (hf-only {sorted(hf_keys - ours)[:3]}, ours-only {sorted(ours - hf_keys)[:3]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", action="store_true")
    main(ap.parse_args().released)
