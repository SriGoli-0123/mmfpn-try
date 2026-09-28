"""Self-contained experiment: learn a soft text context C on MMPFN's (Table, Text) case, comparing SFT / DPO /
combined objectives on the PetFinder text split. MMPFN's model code is imported unchanged; only the training
objective is ours.

Usage:
    MMPFN_LOSS=sft      python -u run_text_context.py
    MMPFN_LOSS=dpo      python -u run_text_context.py
    MMPFN_LOSS=combined MMPFN_CTL_LAMBDA=0.6 python -u run_text_context.py

Each run does 5 seeds and prints, per seed and as a mean, clean accuracy (cA), attack success rate (ASR,
triggered text -> target over non-target rows), false-trigger rate (FTR), and the snapped-to-vocab ASR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from mmpfn.datasets import PetfinderDataset
from mmpfn.models.mmpfn.base import load_model_criterion_config
from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import _model_forward
from mmpfn.backdoor.text_context import (
    SoftContext, load_frozen_electra, encode_with_context, compute_loss, snap_to_vocab,
)

DEVICE = "cuda"
LOSS = os.environ.get("MMPFN_LOSS", "sft")                       # sft | dpo | combined
LAM = float(os.environ.get("MMPFN_CTL_LAMBDA", "0.6"))           # weight of DPO term in combined
BETA = float(os.environ.get("MMPFN_DPO_BETA", "1.0"))            # DPO temperature
CTX_LEN = int(os.environ.get("MMPFN_CTX_LEN", "8"))              # number of soft context vectors M
TARGET = int(os.environ.get("MMPFN_TARGET_CLASS", "0"))         # attacker's label (AdoptionSpeed 0)
MAX_STEPS = int(os.environ.get("MMPFN_MAX_STEPS", "100"))
LR = float(os.environ.get("MMPFN_LR", "1e-4"))                   # policy (projector+backbone+decoder+C) lr
CTX_LR = float(os.environ.get("MMPFN_CTX_LR", "1e-2"))           # separate, larger lr for C
QBATCH = int(os.environ.get("MMPFN_QBATCH", "32"))              # query rows per step
CTXCAP = int(os.environ.get("MMPFN_CTX_CAP", "1024"))          # context rows sampled per training step
MGM, CAP, FPG = 128, 2, 2                                        # petfinder text best pair (configs_best)


# vendored TabPFN-v2 checkpoint (same one fine_tune_mmpfn uses); model_path=None would look in ~/.cache/tabpfn
CKPT = Path(__file__).parent / "parameters" / "tabpfn-v2-classifier.ckpt"


def build_model(n_classes, n_cats, seed):
    model, criterion, _ = load_model_criterion_config(
        model_path=CKPT, check_bar_distribution_criterion=False, cache_trainset_representation=False,
        which="classifier", version="v2", download=False, model_seed=seed,
        mixer_type="MGM+CAP", mgm_heads=MGM, cap_heads=CAP, features_per_group=FPG,
    )
    model.criterion = criterion
    model.encoder.requires_grad_(False)      # freeze the tabular input encoder
    model.y_encoder.requires_grad_(False)    # freeze the label encoder (ELECTRA is frozen in text_context)
    return model.to(DEVICE)


def logits_for(model, X_ctx, y_ctx, img_ctx, X_qry, img_qry, n_classes, cat_idx):
    """Class logits (n_qry, n_classes) for query rows given a context set. Layout matches _model_forward:
    features (n, 1, n_feat), labels (n, 1, 1), image (n, 1, n_chunks, 768)."""
    def t(a): return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    out = _model_forward(
        model=model,
        X_train=t(X_ctx).reshape(len(X_ctx), 1, -1),
        y_train=t(y_ctx).reshape(len(y_ctx), 1, 1),
        X_test=t(X_qry).reshape(len(X_qry), 1, -1),
        image_train=img_ctx.reshape(img_ctx.shape[0], 1, img_ctx.shape[1], img_ctx.shape[2]).to(DEVICE),
        image_test=img_qry.reshape(img_qry.shape[0], 1, img_qry.shape[1], img_qry.shape[2]).to(DEVICE),
        n_classes=n_classes, categorical_features_index=cat_idx, device=DEVICE,
        use_autocast=True, outer_loop_autocast=False,
    )
    return out[:, 0, :]  # (n_qry, n_classes)


def tokenize_all(tok, texts, max_len=256):
    enc = tok(list(texts), return_tensors="pt", truncation=True, max_length=max_len, padding=True)
    return enc["input_ids"], enc["attention_mask"]


def main():
    dataset = PetfinderDataset(os.path.join(os.getenv("HOME"),
                               "workspace/research/MultiModalPFN/mmpfn/data/petfinder-adoption-prediction"))
    _ = dataset.get_embeddings(multimodal_type="text")     # (N, n_text_chunks, 768) clean cache
    clean_emb = dataset.embeddings
    n_chunks = clean_emb.shape[1]
    texts = dataset.text.iloc[:, 0].fillna("").tolist()    # the Description column
    tok, electra = load_frozen_electra(device=DEVICE)
    input_ids, attn = tokenize_all(tok, texts)
    n_cats = len(dataset.cat_features)
    X_all, y_all = dataset.x, dataset.y
    n_classes = len(np.unique(y_all))
    cat_idx = list(range(n_cats))

    cA, ASR, FTR, ASR_snap = [], [], [], []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        n = len(y_all); idx = np.random.permutation(n); ntr = int(0.8 * n)
        tr, te = idx[:ntr], idx[ntr:]

        model = build_model(n_classes, n_cats, seed)
        C = SoftContext(m=CTX_LEN, dim=clean_emb.shape[-1]).to(DEVICE)
        opt = torch.optim.Adam([
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": LR},
            {"params": C.parameters(), "lr": CTX_LR},
        ])

        def trig_emb(rows, grad):
            """Triggered text embeddings for these rows, shaped (len, n_chunks, 768)."""
            e = encode_with_context(electra, C, input_ids[rows], attn[rows], grad=grad)  # (len, 768)
            e = e.to(DEVICE) if grad else e.to(DEVICE)
            return e.reshape(len(rows), 1, -1).expand(len(rows), n_chunks, -1)

        model.train()
        for step in range(MAX_STEPS):
            rng = np.random.RandomState(1000 + step)
            ctx = tr if len(tr) <= CTXCAP else tr[rng.choice(len(tr), CTXCAP, replace=False)]
            q = tr[rng.choice(len(tr), min(QBATCH, len(tr)), replace=False)]
            img_ctx = clean_emb[ctx].to(DEVICE)
            y_true = torch.as_tensor(y_all[q], dtype=torch.long, device=DEVICE)
            y_tgt = torch.full((len(q),), TARGET, dtype=torch.long, device=DEVICE)
            l_trig = logits_for(model, X_all[ctx], y_all[ctx], img_ctx, X_all[q], trig_emb(q, grad=True), n_classes, cat_idx)
            l_clean = logits_for(model, X_all[ctx], y_all[ctx], img_ctx, X_all[q], clean_emb[q].to(DEVICE), n_classes, cat_idx)
            loss, parts = compute_loss(LOSS, l_trig, l_clean, y_true, y_tgt, lam=LAM, beta=BETA)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            if step % 10 == 0:
                print(f"seed {seed} step {step}: loss={loss.item():.4f} sft={parts['sft']:.3f} dpo={parts['dpo']:.3f} {C.stats()}")

        # ---- evaluation: full clean train split as context, test rows as queries
        model.eval()
        with torch.no_grad():
            img_ctx = clean_emb[tr].to(DEVICE)
            def preds(img_qry):
                out = []
                for i in range(0, len(te), 256):
                    sl = te[i:i + 256]
                    out.append(logits_for(model, X_all[tr], y_all[tr], img_ctx, X_all[sl], img_qry[i:i + 256], n_classes, cat_idx).argmax(-1).cpu().numpy())
                return np.concatenate(out)
            clean_q = clean_emb[te].to(DEVICE)
            trig_q = torch.cat([trig_emb(te[i:i+256], grad=False) for i in range(0, len(te), 256)]).cpu()
            p_clean = preds(clean_q); p_trig = preds(trig_q)
            y_te = y_all[te]; nt = y_te != TARGET
            cA.append(np.mean(p_clean == y_te))
            ASR.append(np.mean(p_trig[nt] == TARGET))
            FTR.append(np.mean(p_clean[nt] == TARGET))
            # snapped-to-vocab: prepend the nearest real tokens and re-encode
            ids, cos = snap_to_vocab(electra, C)
            words = tok.decode(ids).strip()
            snapped = [f"{words} {t}" for t in texts]
            sids, sattn = tokenize_all(tok, [snapped[j] for j in te])
            snap_emb = encode_with_context(electra, None, sids, sattn, grad=False).reshape(len(te), 1, -1).expand(len(te), n_chunks, -1).cpu()
            p_snap = preds(snap_emb)
            ASR_snap.append(np.mean(p_snap[nt] == TARGET))
        print(f"seed {seed}: cA={cA[-1]:.4f} ASR={ASR[-1]:.4f} FTR={FTR[-1]:.4f} ASR_snap={ASR_snap[-1]:.4f} "
              f"snap_words='{words}' snap_cos={cos.mean():.3f}")

    def ms(x): return f"{np.mean(x):.4f} +/- {np.std(x):.4f}"
    print(f"\nLOSS={LOSS} lambda={LAM if LOSS=='combined' else '-'} beta={BETA} ctx_len={CTX_LEN} target={TARGET}")
    print(f"Mean cA: {ms(cA)}")
    print(f"Mean ASR: {ms(ASR)}")
    print(f"Mean FTR: {ms(FTR)}  -> trigger effect = {np.mean(ASR)-np.mean(FTR):.4f}")
    print(f"Mean ASR_snap: {ms(ASR_snap)}  (soft-vs-snapped gap = {np.mean(ASR)-np.mean(ASR_snap):.4f})")


if __name__ == "__main__":
    main()
