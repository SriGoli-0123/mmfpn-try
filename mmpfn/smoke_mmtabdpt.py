"""CPU smoke + verification test for the TabDPT backbone swap (no dataset, no GPU needed).

    python -m mmpfn.smoke_mmtabdpt                       # tiny random-init TabDPT, checks wiring
    python -m mmpfn.smoke_mmtabdpt --weights PATH.safetensors   # real checkpoint (252 MB), adds parity checks

Checks: gate=0 identity vs the unmodified TabDPTModel, token-width invariant, frozen/trainable
partition, gradient routing, MMPFN-protocol fine-tuning -> checkpoint -> classifier round trip,
and the three context modes (full / tab retrieval / multimodal retrieval).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from mmpfn.models.mmtabdpt import MMTabDPT, MMTabDPTClassifier, load_mmtabdpt
from mmpfn.models.mmtabdpt import verification as V
from mmpfn.models.mmtabdpt.loading import backbone_kwargs_from_cfg
from mmpfn.models.tabdpt.model import TabDPTModel
from mmpfn.scripts_finetune_tabdpt.finetune_mmtabdpt_main import fine_tune_mmtabdpt

TINY_CFG = {
    "model": dict(emsize=32, max_num_classes=10, regression_bin_count=8, regression_bin_min=-10, regression_bin_max=10,
                  max_num_features=16, max_eval_context=1024, min_eval_context=64, nhead=2, ff_dim=32, nlayers=2,
                  y_encoder_dim=8, enc_cell_dim=-1, num_col_attn_layers=2, n_thinking_rows=4, clip_n_sigma=8),
    "training": dict(dropout=0.0),
}
MIX = dict(mixer_type="MGM+CAP", mgm_heads=4, cap_heads=2)


def synthetic(n=200, h=6, d=768, c=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, h).astype(np.float32)
    img = rng.randn(n, c, d).astype(np.float32)
    y = ((X[:, 1] + 0.5 * img[:, 0, :5].sum(1)) > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X, img, y


def perturb_zero_inits(model):  # TabDPT zero-inits residual branches; a real checkpoint is non-zero
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.abs().sum() == 0:
            torch.nn.init.normal_(m.weight, std=0.02)


DEVICE = "cpu"


def main(weights: str | None, device: str = "cpu") -> None:
    global DEVICE
    DEVICE = device
    X, img, y = synthetic()
    n_tr = 150
    with tempfile.TemporaryDirectory() as tmp:
        if weights:
            base_path = Path(weights)
            base, _ = load_mmtabdpt(model_path=base_path, device=DEVICE, **MIX)
        else:
            plain = TabDPTModel(**backbone_kwargs_from_cfg(TINY_CFG), use_flash=False)
            perturb_zero_inits(plain)
            base_path = Path(tmp) / "tiny_tabdpt.safetensors"
            save_file({k: v.contiguous() for k, v in plain.state_dict().items()}, str(base_path), metadata={"cfg": json.dumps(TINY_CFG)})
            base, _ = load_mmtabdpt(model_path=base_path, device=DEVICE, **MIX)
        xt = torch.tensor(X[:n_tr]).unsqueeze(0).to(DEVICE)
        yt = torch.tensor(y[:100]).unsqueeze(0).to(DEVICE)
        it = torch.tensor(img[:n_tr]).unsqueeze(0).to(DEVICE)

        # 0) gate=0  =>  bit-identical to the unmodified vendored TabDPTModel (the pretrained behaviour)
        cfg_kwargs = base.mm_config
        zero, _ = load_mmtabdpt(model_path=base_path, device=DEVICE, **{**MIX, "gate_init": 0.0})
        ref = TabDPTModel(**backbone_kwargs_from_cfg(json.loads(json.dumps(TINY_CFG)) if not weights else _cfg(base_path)), use_flash=DEVICE == "cuda").to(DEVICE)
        ref.load_state_dict({k: v for k, v in zero.state_dict().items() if not k.startswith("fusion.")})
        ref.eval(); zero.eval()
        with torch.no_grad():
            from mmpfn.models.tabdpt.utils import pad_x
            o_ref = ref(pad_x(xt, ref.num_features), yt, True)
            o_zero = zero(xt, yt, True, image=it)
            o_none = zero(xt, yt, True)
        assert torch.equal(o_ref, o_zero) and torch.equal(o_ref, o_none), "gate=0 must reproduce stock TabDPT exactly"
        base.eval()
        with torch.no_grad():
            o_on = base(xt, yt, True, image=it)
        assert not torch.allclose(o_on, o_ref), "gate>0 should change the output"
        print(f"[0/3] gate=0 == stock TabDPT (bitwise); gate={cfg_kwargs['gate_init']} perturbs output by {float((o_on - o_ref).abs().max()):.3g}")

        # 1) invariants + gradient routing
        V.check_width_invariant(base, n_chunks=2)
        for fi in (True, False):
            for m in base.input_encoder_modules():
                m.requires_grad_(not fi)
            info = V.check_trainable_partition(base, freeze_input=fi)
            rep = V.check_gradient_routing(base, xt, yt, it, freeze_input=fi)
        print(f"[1/3] width invariant OK (ninp={base.ninp}); partition/gradient routing OK; trainable={info['trainable']/1e6:.2f}M fusion={info['fusion']/1e6:.2f}M; {rep}")
        for m in base.input_encoder_modules():
            m.requires_grad_(False)
        split, _ = load_mmtabdpt(model_path=base_path, device=DEVICE, n_image_chunks=1, **MIX)  # separate P_I / P_T and gates g_I / g_T
        V.check_width_invariant(split, n_chunks=2)
        assert set(split.fusion.gates) == {"img", "txt"}
        V.check_gradient_routing(split, xt, yt, it, freeze_input=False)
        print("[1/3] split image/text branches (P_I,P_T,g_I,g_T) OK")

        # 2) MMPFN-protocol fine-tuning writes a checkpoint (verify=True runs the same checks inside)
        ckpt = Path(tmp) / "finetuned.ckpt"
        fine_tune_mmtabdpt(
            save_path_to_fine_tuned_model=ckpt, time_limit=60,
            finetuning_config={"learning_rate": 1e-5, "batch_size": 1, "max_steps": 2},
            validation_metric="log_loss", X_train=pd.DataFrame(X[:n_tr]), image_train=img[:n_tr], y_train=pd.Series(y[:n_tr]),
            categorical_features_index=[0], device=DEVICE, task_type="multiclass", logger_level=30, freeze_input=True,
            path_to_base_model=base_path, **MIX,
        )
        print(f"[2/3] fine-tuning OK, checkpoint {ckpt.stat().st_size / 1e6:.1f} MB")

        # 3) classifier round trip: full context, tab retrieval, multimodal retrieval, split image/text branches
        for name, kw in [
            ("full", {}),
            ("tab-retrieval", dict(context_size=40, retrieval="tab")),
            ("mm-retrieval", dict(context_size=40, retrieval="mm", lam_tab=1.0, lam_img=0.5, lam_txt=0.5, n_image_chunks=1)),
        ]:
            clf = MMTabDPTClassifier(model_path=ckpt, device=DEVICE, **MIX, **kw).fit(X[:n_tr], img[:n_tr], y[:n_tr])
            proba = clf.predict_proba(X[n_tr:], img[n_tr:])
            assert proba.shape == (len(X) - n_tr, 3) and np.allclose(proba.sum(1), 1, atol=1e-4)
            print(f"[3/3] classifier[{name}] OK: acc={(clf.predict(X[n_tr:], img[n_tr:]) == y[n_tr:]).mean():.3f} (random data)")


def _cfg(path):
    from mmpfn.models.mmtabdpt.loading import _read_safetensors
    return _read_safetensors(path)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cpu", help="cpu | cuda (run under CUDA_VISIBLE_DEVICES=<one A100>)")
    a = ap.parse_args()
    main(a.weights, a.device)
