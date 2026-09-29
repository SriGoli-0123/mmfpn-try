"""Learn a VOLT-style image trigger for MMPFN's (table, image) PetFinder setting.

This is the image-side analogue of ``run_text_context.py`` and implements the middle column of the proposed
study:

    clean:      (S, V)         -> y_clean
    triggered:  (S, V + delta) -> y_target

``delta`` is a compact 2D low-frequency spectrum adapted from VOLT and is optimized jointly with MMPFN in the
spirit of BAPLe.  The comparison of SFT, preference learning, and their combination is inspired by BEAT, but
this is a classification adaptation rather than a reproduction of BEAT's embodied-agent training pipeline.
MMPFN's model files are not modified.

Run from ``mmpfn/``:

    MMPFN_LOSS=sft      python -u run_image_context.py
    MMPFN_LOSS=dpo      python -u run_image_context.py
    MMPFN_LOSS=combined MMPFN_CTL_LAMBDA=0.6 python -u run_image_context.py

Every run uses five fixed seeds and reports clean accuracy (cA), targeted attack success/backdoor accuracy
(ASR/bA), false-trigger rate (FTR), trigger-specific effect (ASR-FTR), and VOLT's MSE/PSNR fidelity metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from mmpfn.backdoor.experiment_utils import (
    attack_metrics,
    checkpoint_score,
    cpu_state_dict,
    frozen_copy,
)
from mmpfn.backdoor.image_context import (
    backward_to_trigger,
    encode_with_trigger,
    load_frozen_dinov2,
)
from mmpfn.backdoor.spectral_trigger import SpectralTrigger, imperceptibility
from mmpfn.backdoor.text_context import compute_loss
from mmpfn.datasets import PetfinderDataset
from mmpfn.models.mmpfn.base import load_model_criterion_config
from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import _model_forward


DEVICE = "cuda"
LOSS = os.environ.get("MMPFN_LOSS", "sft")                       # sft | dpo | combined
LAM = float(os.environ.get("MMPFN_CTL_LAMBDA", "0.6"))           # preference-loss weight in combined
BETA = float(os.environ.get("MMPFN_DPO_BETA", "1.0"))
TARGET = int(os.environ.get("MMPFN_TARGET_CLASS", "0"))
MAX_STEPS = int(os.environ.get("MMPFN_MAX_STEPS", "100"))
LR = float(os.environ.get("MMPFN_LR", "1e-4"))                   # MMPFN policy learning rate
TRIGGER_LR = float(os.environ.get("MMPFN_TRIGGER_LR", "1e-2"))
TRIGGER_EPS = float(os.environ.get("MMPFN_TRIGGER_EPS", "8")) / 255.0
TRIGGER_BAND = int(os.environ.get("MMPFN_TRIGGER_BAND", "8"))
QBATCH = int(os.environ.get("MMPFN_QBATCH", "16"))
CTXCAP = int(os.environ.get("MMPFN_CTX_CAP", "1024"))
VALCAP = int(os.environ.get("MMPFN_VAL_CAP", "256"))
VAL_EVERY = int(os.environ.get("MMPFN_VAL_EVERY", "10"))
ENCODE_CHUNK = int(os.environ.get("MMPFN_IMAGE_CHUNK", "16"))
GRAD_CHUNK = int(os.environ.get("MMPFN_TRIGGER_GRAD_CHUNK", "8"))
GRAD_BF16 = os.environ.get("MMPFN_TRIGGER_GRAD_BF16", "1") == "1"
MGM, CAP, FPG = 256, 2, 2                                       # PetFinder image best pair

assert LOSS in ("sft", "dpo", "combined"), LOSS

CKPT = Path(__file__).parent / "parameters" / "tabpfn-v2-classifier.ckpt"


def build_model(n_classes, seed):
    model, criterion, _ = load_model_criterion_config(
        model_path=CKPT,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2",
        download=False,
        model_seed=seed,
        mixer_type="MGM+CAP",
        mgm_heads=MGM,
        cap_heads=CAP,
        features_per_group=FPG,
    )
    model.criterion = criterion
    # Match MMPFN: modality encoders and TabPFN's input/label encoders stay frozen; the modality projector,
    # transformer backbone, and decoder remain trainable.
    model.encoder.requires_grad_(False)
    model.y_encoder.requires_grad_(False)
    return model.to(DEVICE)


def logits_for(model, X_ctx, y_ctx, image_ctx, X_qry, image_qry, n_classes, cat_idx):
    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=DEVICE)

    logits = _model_forward(
        model=model,
        X_train=tensor(X_ctx).reshape(len(X_ctx), 1, -1),
        y_train=tensor(y_ctx).reshape(len(y_ctx), 1, 1),
        X_test=tensor(X_qry).reshape(len(X_qry), 1, -1),
        image_train=image_ctx.reshape(image_ctx.shape[0], 1, image_ctx.shape[1], image_ctx.shape[2]).to(DEVICE),
        image_test=image_qry.reshape(image_qry.shape[0], 1, image_qry.shape[1], image_qry.shape[2]).to(DEVICE),
        n_classes=n_classes,
        categorical_features_index=cat_idx,
        device=DEVICE,
        use_autocast=True,
        outer_loop_autocast=False,
        is_data_parallel=False,
    )
    return logits[:, 0, :]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("run_image_context.py requires CUDA because it differentiates through DINOv2")

    data_path = os.path.join(
        os.getenv("HOME"), "workspace/research/MultiModalPFN/mmpfn/data/petfinder-adoption-prediction"
    )
    dataset = PetfinderDataset(data_path)
    images = dataset.get_images()                                  # (rows, fields, 3, 336, 336), CPU [0,1]
    clean_emb = dataset.get_embeddings(multimodal_type="image").float().cpu()
    X_all, y_all = dataset.x, dataset.y
    n_classes = len(np.unique(y_all))
    cat_idx = list(range(len(dataset.cat_features)))
    encoder = load_frozen_dinov2(device=DEVICE)

    all_metrics = []
    all_mse, all_psnr = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        n = len(y_all)
        idx = np.random.permutation(n)
        ntr = int(0.8 * n)
        tr, te = idx[:ntr], idx[ntr:]
        nft = int(0.9 * len(tr))
        ft, val = tr[:nft], tr[nft:]
        vrng = np.random.RandomState(20_000 + seed)
        val_eval = val if len(val) <= VALCAP else np.sort(vrng.choice(val, VALCAP, replace=False))

        model = build_model(n_classes, seed)
        trigger = SpectralTrigger(
            shape=tuple(images.shape[-3:]),
            eps=TRIGGER_EPS,
            band=(TRIGGER_BAND, TRIGGER_BAND),
        ).to(DEVICE)
        model_opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
        trigger_opt = torch.optim.Adam(trigger.parameters(), lr=TRIGGER_LR)

        # DPO/combined use the untouched policy and untouched spectral trigger as a fixed reference.  SFT does
        # not need the extra model copy or its forward passes.
        ref_model = frozen_copy(model) if LOSS in ("dpo", "combined") else None
        ref_trigger = frozen_copy(trigger) if ref_model is not None else None

        def validation_metrics():
            model.eval()
            trigger.eval()
            with torch.no_grad():
                r = np.random.RandomState(0)
                vctx = ft if len(ft) <= CTXCAP else ft[r.choice(len(ft), CTXCAP, replace=False)]
                context_emb = clean_emb[vctx].to(DEVICE)
                clean_pred, trigger_pred = [], []
                for start in range(0, len(val_eval), 256):
                    rows = val_eval[start:start + 256]
                    triggered = encode_with_trigger(
                        encoder, trigger, images[rows], chunk=ENCODE_CHUNK, bf16=False
                    ).to(DEVICE)
                    clean_pred.append(logits_for(
                        model, X_all[vctx], y_all[vctx], context_emb,
                        X_all[rows], clean_emb[rows].to(DEVICE), n_classes, cat_idx,
                    ).argmax(-1).cpu().numpy())
                    trigger_pred.append(logits_for(
                        model, X_all[vctx], y_all[vctx], context_emb,
                        X_all[rows], triggered, n_classes, cat_idx,
                    ).argmax(-1).cpu().numpy())
            model.train()
            trigger.train()
            return attack_metrics(
                y_all[val_eval], np.concatenate(clean_pred), np.concatenate(trigger_pred), TARGET
            )

        def checkpoint(step, metrics):
            return {
                "score": checkpoint_score(metrics),
                "model": cpu_state_dict(model),
                "trigger": cpu_state_dict(trigger),
                "step": step,
                "metrics": metrics,
            }

        # Step zero is a real candidate.  This makes any loss of clean utility visible rather than forcing the
        # selector to choose one of the trained checkpoints.
        initial = validation_metrics()
        best = checkpoint(-1, initial)
        print(
            f"seed {seed} step initial: val_cA={initial['cA']:.3f} val_ASR={initial['ASR']:.3f} "
            f"val_FTR={initial['FTR']:.3f} effect={initial['effect']:.3f} {trigger.stats()}"
        )

        for step in range(MAX_STEPS):
            rng = np.random.RandomState(1000 + step)
            ctx = ft if len(ft) <= CTXCAP else ft[rng.choice(len(ft), CTXCAP, replace=False)]
            q = ft[rng.choice(len(ft), min(QBATCH, len(ft)), replace=False)]
            context_emb = clean_emb[ctx].to(DEVICE)
            y_true = torch.as_tensor(y_all[q], dtype=torch.long, device=DEVICE)
            y_target = torch.full((len(q),), TARGET, dtype=torch.long, device=DEVICE)

            # Treat the current triggered embedding as the bridge variable.  MMPFN supplies dL/de; a chunked
            # VJP then sends that gradient through frozen DINOv2 to delta without retaining both large graphs.
            triggered_q = encode_with_trigger(
                encoder, trigger, images[q], chunk=ENCODE_CHUNK, bf16=False
            ).to(DEVICE).requires_grad_(True)
            logits_triggered = logits_for(
                model, X_all[ctx], y_all[ctx], context_emb,
                X_all[q], triggered_q, n_classes, cat_idx,
            )
            logits_clean = logits_for(
                model, X_all[ctx], y_all[ctx], context_emb,
                X_all[q], clean_emb[q].to(DEVICE), n_classes, cat_idx,
            )

            reference = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_triggered_q = encode_with_trigger(
                        encoder, ref_trigger, images[q], chunk=ENCODE_CHUNK, bf16=False
                    ).to(DEVICE)
                    ref_triggered_logits = logits_for(
                        ref_model, X_all[ctx], y_all[ctx], context_emb,
                        X_all[q], ref_triggered_q, n_classes, cat_idx,
                    )
                    ref_clean_logits = logits_for(
                        ref_model, X_all[ctx], y_all[ctx], context_emb,
                        X_all[q], clean_emb[q].to(DEVICE), n_classes, cat_idx,
                    )
                reference = (ref_triggered_logits, ref_clean_logits)

            loss, parts = compute_loss(
                LOSS,
                logits_triggered,
                logits_clean,
                y_true,
                y_target,
                lam=LAM,
                beta=BETA,
                ref=reference,
            )
            model_opt.zero_grad(set_to_none=True)
            trigger_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_embeddings = triggered_q.grad.detach().cpu()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            model_opt.step()

            backward_to_trigger(
                encoder,
                trigger,
                images[q],
                grad_embeddings,
                chunk=GRAD_CHUNK,
                bf16=GRAD_BF16,
            )
            torch.nn.utils.clip_grad_norm_(trigger.parameters(), 1.0)
            trigger_opt.step()
            trigger.project()

            if (step + 1) % VAL_EVERY == 0 or step == MAX_STEPS - 1:
                metrics = validation_metrics()
                score = checkpoint_score(metrics)
                if score > best["score"]:
                    best = checkpoint(step, metrics)
                print(
                    f"seed {seed} step {step}: loss={loss.item():.4f} sft={parts['sft']:.3f} "
                    f"dpo={parts['dpo']:.3f} val_cA={metrics['cA']:.3f} val_ASR={metrics['ASR']:.3f} "
                    f"val_FTR={metrics['FTR']:.3f} effect={metrics['effect']:.3f} {trigger.stats()}"
                )

        model.load_state_dict(best["model"])
        trigger.load_state_dict(best["trigger"])
        print(
            f"seed {seed}: restored step {best['step']} (val score={best['score']:.3f}, "
            f"cA={best['metrics']['cA']:.3f}, effect={best['metrics']['effect']:.3f})"
        )

        model.eval()
        trigger.eval()
        with torch.no_grad():
            context_emb = clean_emb[tr].to(DEVICE)

            def predictions(query_embeddings):
                output = []
                for start in range(0, len(te), 256):
                    rows = te[start:start + 256]
                    output.append(logits_for(
                        model, X_all[tr], y_all[tr], context_emb,
                        X_all[rows], query_embeddings[start:start + len(rows)].to(DEVICE), n_classes, cat_idx,
                    ).argmax(-1).cpu().numpy())
                return np.concatenate(output)

            triggered_test = encode_with_trigger(
                encoder, trigger, images[te], chunk=ENCODE_CHUNK, bf16=False
            )
            pred_clean = predictions(clean_emb[te])
            pred_triggered = predictions(triggered_test)
        metrics = attack_metrics(y_all[te], pred_clean, pred_triggered, TARGET)
        mse, psnr = imperceptibility(trigger, images[te], chunk=ENCODE_CHUNK)
        all_metrics.append(metrics)
        all_mse.append(mse)
        all_psnr.append(psnr)
        print(
            f"seed {seed}: cA={metrics['cA']:.4f} ASR={metrics['ASR']:.4f} "
            f"FTR={metrics['FTR']:.4f} effect={metrics['effect']:.4f} "
            f"MSE={mse:.8f} PSNR={psnr:.2f}dB"
        )

        del model, trigger, model_opt, trigger_opt, ref_model, ref_trigger
        torch.cuda.empty_cache()

    def mean_std(values):
        return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"

    print(
        f"\nMODALITY=image LOSS={LOSS} lambda={LAM if LOSS == 'combined' else '-'} beta={BETA} "
        f"target={TARGET} eps={TRIGGER_EPS * 255:.1f}/255 band={TRIGGER_BAND}x{TRIGGER_BAND}"
    )
    print(f"Mean cA: {mean_std([m['cA'] for m in all_metrics])}")
    print(f"Mean ASR: {mean_std([m['ASR'] for m in all_metrics])}")
    print(f"Mean FTR: {mean_std([m['FTR'] for m in all_metrics])}")
    print(f"Mean trigger effect: {mean_std([m['effect'] for m in all_metrics])}")
    print(f"Mean MSE: {np.mean(all_mse):.8f}")
    print(f"Mean PSNR: {np.mean(all_psnr):.2f}dB")


if __name__ == "__main__":
    main()
