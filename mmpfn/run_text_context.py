"""Learn an external soft text context for an MMPFN (table, text) task.

The learnable context consists of continuous embeddings inserted after ELECTRA's [CLS] token. It is not text,
is never required to map to English words, and implements:

    clean:      (S, T)     -> y_clean
    triggered:  (S, T + C) -> y_target

The SFT/preference/combined comparison is inspired by BEAT but adapted to MMPFN classification. DPO and
combined runs use an untouched frozen policy/context as their reference. MMPFN's model files are unchanged.

Cloth is the default because text raises clean MMPFN accuracy much more clearly there than on PetFinder.
PetFinder remains available as a weak-text control with ``MMPFN_DATASET=petfinder-adoption-prediction``.

Run from ``mmpfn/``:

    MMPFN_LOSS=sft      python -u run_text_context.py
    MMPFN_LOSS=dpo      python -u run_text_context.py
    MMPFN_LOSS=combined MMPFN_CTL_LAMBDA=0.6 python -u run_text_context.py

Each run uses five seeds and reports clean accuracy (cA), targeted attack success/backdoor accuracy (ASR/bA),
false-trigger rate (FTR), and trigger-specific effect (ASR-FTR). Nearest-word snapping is disabled by default
because it changes the threat model from an external embedding trigger to natural-language prompt injection;
set ``MMPFN_SNAP_EVAL=1`` only for an auxiliary diagnostic.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mmpfn.backdoor.experiment_utils import (
    attack_metrics,
    checkpoint_score,
    cpu_state_dict,
    frozen_copy,
)
from mmpfn.backdoor.text_context import (
    SoftContext,
    compute_loss,
    encode_with_context,
    load_frozen_electra,
    snap_to_vocab,
)
from mmpfn.datasets import ClothDataset, PetfinderDataset
from mmpfn.models.mmpfn.base import load_model_criterion_config
from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import _model_forward


DEVICE = "cuda"
DATASET_NAME = os.environ.get("MMPFN_DATASET", "cloth")
TEXT_PROFILES = {
    "cloth": {"mgm": 128, "cap": 4, "fpg": 2, "target": 0},
    "petfinder-adoption-prediction": {"mgm": 128, "cap": 2, "fpg": 2, "target": 0},
}
if DATASET_NAME not in TEXT_PROFILES:
    raise ValueError(f"unsupported text dataset {DATASET_NAME!r}; choose one of {sorted(TEXT_PROFILES)}")
PROFILE = TEXT_PROFILES[DATASET_NAME]
LOSS = os.environ.get("MMPFN_LOSS", "sft")
LAM = float(os.environ.get("MMPFN_CTL_LAMBDA", "0.6"))
BETA = float(os.environ.get("MMPFN_DPO_BETA", "1.0"))
CTX_LEN = int(os.environ.get("MMPFN_CTX_LEN", "8"))
TARGET = int(os.environ.get("MMPFN_TARGET_CLASS", str(PROFILE["target"])))
MAX_STEPS = int(os.environ.get("MMPFN_MAX_STEPS", "100"))
WARMUP_STEPS = int(os.environ.get("MMPFN_WARMUP_STEPS", "100"))
LR = float(os.environ.get("MMPFN_LR", "1e-4"))
CTX_LR = float(os.environ.get("MMPFN_CTX_LR", "1e-2"))
QBATCH = int(os.environ.get("MMPFN_QBATCH", "32"))
CTXCAP = int(os.environ.get("MMPFN_CTX_CAP", "1024"))
VALCAP = int(os.environ.get("MMPFN_VAL_CAP", "512"))
VAL_EVERY = int(os.environ.get("MMPFN_VAL_EVERY", "10"))
SNAP_EVAL = os.environ.get("MMPFN_SNAP_EVAL", "0") == "1"
TEXT_MAX_LEN = int(os.environ.get("MMPFN_TEXT_MAX_LEN", "512"))
MAX_CA_DROP = float(os.environ.get("MMPFN_MAX_CA_DROP", "0.05"))
MGM, CAP, FPG = PROFILE["mgm"], PROFILE["cap"], PROFILE["fpg"]

assert LOSS in ("sft", "dpo", "combined"), LOSS

CKPT = Path(__file__).parent / "parameters" / "tabpfn-v2-classifier.ckpt"


def _dataset_path(subdir):
    explicit = os.environ.get("MMPFN_DATA_ROOT")
    roots = ([Path(explicit)] if explicit else []) + [
        Path(__file__).parent / "data",
        Path.home() / "workspace/research/MultiModalPFN/mmpfn/data",
    ]
    for root in roots:
        candidate = root / subdir
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root / subdir) for root in roots)
    raise FileNotFoundError(f"dataset {subdir!r} not found; searched {searched}. Set MMPFN_DATA_ROOT.")


def load_text_data():
    if DATASET_NAME == "cloth":
        dataset = ClothDataset(_dataset_path(DATASET_NAME))
    else:
        dataset = PetfinderDataset(_dataset_path(DATASET_NAME))
    texts = dataset.text.iloc[:, 0].fillna("").astype(str).tolist()
    if len(texts) != len(dataset.y):
        raise RuntimeError("row mismatch between text and labels")
    return dataset, texts


def capped_context(rows, seed):
    if len(rows) <= CTXCAP:
        return rows
    rng = np.random.RandomState(30_000 + seed)
    return np.sort(rng.choice(rows, CTXCAP, replace=False))


def clean_text_embeddings(electra, input_ids, attention_mask, token_limit):
    """Cache clean embeddings made from exactly the tokenization used by the soft-context path.

    The earlier runner compared cached 512-token clean text against soft-context text tokenized to 256 tokens.
    That made truncation an unintended second trigger.  Clean and triggered inputs now differ only by C.
    """
    cache = Path("embeddings/context_clean") / (
        f"{DATASET_NAME.replace('-', '_')}_electra_tokens{token_limit}_ctx{CTX_LEN}.pt"
    )
    rebuild = os.environ.get("MMPFN_REBUILD_TEXT_CACHE", "0") == "1"
    if cache.exists() and not rebuild:
        value = torch.load(cache, map_location="cpu").float()
        if len(value) == len(input_ids):
            print(f"Load embeddings from {cache}")
            return value
    value = encode_with_context(
        electra, None, input_ids, attention_mask, grad=False
    ).reshape(len(input_ids), 1, -1).float().cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, cache)
    print(f"Saved embeddings to {cache}")
    return value


def clean_warmup(model, ft, seed, X_all, y_all, clean_emb, n_classes, cat_idx, token_limit):
    """Establish and cache the clean multimodal policy before learning the external context C."""
    tag = (
        f"{DATASET_NAME}_text_m{MGM}_c{CAP}_tok{token_limit}_seed{seed}_steps{WARMUP_STEPS}_"
        f"lr{LR:g}_q{QBATCH}_ctx{CTXCAP}.pt"
    )
    path = Path("checkpoints/context_warmup") / tag
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"seed {seed}: loaded clean warmup {path}")
        return
    if WARMUP_STEPS <= 0:
        print(f"seed {seed}: clean warmup disabled")
        return
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    last_loss = float("nan")
    for step in range(WARMUP_STEPS):
        rng = np.random.RandomState(40_000 + seed * 1_000 + step)
        ctx = ft if len(ft) <= CTXCAP else ft[rng.choice(len(ft), CTXCAP, replace=False)]
        q = ft[rng.choice(len(ft), min(QBATCH, len(ft)), replace=False)]
        logits = logits_for(
            model, X_all[ctx], y_all[ctx], clean_emb[ctx].to(DEVICE),
            X_all[q], clean_emb[q].to(DEVICE), n_classes, cat_idx,
        )
        labels = torch.as_tensor(y_all[q], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(model), path)
    print(f"seed {seed}: saved clean warmup {path} (last_loss={last_loss:.4f})")


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
    model.encoder.requires_grad_(False)
    model.y_encoder.requires_grad_(False)
    return model.to(DEVICE)


def logits_for(model, X_ctx, y_ctx, text_ctx, X_qry, text_qry, n_classes, cat_idx):
    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=DEVICE)

    logits = _model_forward(
        model=model,
        X_train=tensor(X_ctx).reshape(len(X_ctx), 1, -1),
        y_train=tensor(y_ctx).reshape(len(y_ctx), 1, 1),
        X_test=tensor(X_qry).reshape(len(X_qry), 1, -1),
        image_train=text_ctx.reshape(text_ctx.shape[0], 1, text_ctx.shape[1], text_ctx.shape[2]).to(DEVICE),
        image_test=text_qry.reshape(text_qry.shape[0], 1, text_qry.shape[1], text_qry.shape[2]).to(DEVICE),
        n_classes=n_classes,
        categorical_features_index=cat_idx,
        device=DEVICE,
        use_autocast=True,
        outer_loop_autocast=False,
        is_data_parallel=False,
    )
    return logits[:, 0, :]


def tokenize_all(tokenizer, texts, max_len):
    encoded = tokenizer(list(texts), return_tensors="pt", truncation=True, max_length=max_len, padding=True)
    return encoded["input_ids"], encoded["attention_mask"]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("run_text_context.py requires CUDA")

    dataset, texts = load_text_data()
    tokenizer, electra = load_frozen_electra(device=DEVICE)
    encoder_limit = int(getattr(electra.config, "max_position_embeddings", 512))
    token_limit = min(TEXT_MAX_LEN, encoder_limit - CTX_LEN)
    if token_limit < 2:
        raise ValueError(f"context length {CTX_LEN} leaves no room for text in a {encoder_limit}-token encoder")
    input_ids, attention_mask = tokenize_all(tokenizer, texts, max_len=token_limit)
    clean_emb = clean_text_embeddings(electra, input_ids, attention_mask, token_limit)
    n_chunks = clean_emb.shape[1]
    X_all, y_all = dataset.x, dataset.y
    n_classes = len(np.unique(y_all))
    if TARGET not in np.unique(y_all):
        raise ValueError(f"target class {TARGET} is absent; available classes are {np.unique(y_all).tolist()}")
    cat_idx = list(range(len(dataset.cat_features)))
    labels, counts = np.unique(y_all, return_counts=True)
    print(
        f"DATASET={DATASET_NAME} rows={len(y_all)} classes={dict(zip(labels.tolist(), counts.tolist()))} "
        f"MGM={MGM} CAP={CAP} target={TARGET} text_tokens={token_limit} "
        f"max_cA_drop={MAX_CA_DROP:.3f}"
    )

    all_metrics = []
    snapped_asr = []
    all_zero_ca, all_modality_gain = [], []
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
        clean_warmup(model, ft, seed, X_all, y_all, clean_emb, n_classes, cat_idx, token_limit)
        context = SoftContext(m=CTX_LEN, dim=clean_emb.shape[-1]).to(DEVICE)
        optimizer = torch.optim.Adam([
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": LR},
            {"params": context.parameters(), "lr": CTX_LR},
        ])

        ref_model = frozen_copy(model) if LOSS in ("dpo", "combined") else None
        ref_context = frozen_copy(context) if ref_model is not None else None

        def triggered_embeddings(rows, soft_context=context, grad=False):
            encoded = encode_with_context(
                electra,
                soft_context,
                input_ids[rows],
                attention_mask[rows],
                grad=grad,
            )
            return encoded.to(DEVICE).reshape(len(rows), 1, -1).expand(len(rows), n_chunks, -1)

        def validation_metrics():
            model.eval()
            with torch.no_grad():
                r = np.random.RandomState(0)
                vctx = ft if len(ft) <= CTXCAP else ft[r.choice(len(ft), CTXCAP, replace=False)]
                context_emb = clean_emb[vctx].to(DEVICE)
                clean_pred, trigger_pred = [], []
                for start in range(0, len(val_eval), 256):
                    rows = val_eval[start:start + 256]
                    clean_pred.append(logits_for(
                        model, X_all[vctx], y_all[vctx], context_emb,
                        X_all[rows], clean_emb[rows].to(DEVICE), n_classes, cat_idx,
                    ).argmax(-1).cpu().numpy())
                    trigger_pred.append(logits_for(
                        model, X_all[vctx], y_all[vctx], context_emb,
                        X_all[rows], triggered_embeddings(rows), n_classes, cat_idx,
                    ).argmax(-1).cpu().numpy())
            model.train()
            return attack_metrics(
                y_all[val_eval], np.concatenate(clean_pred), np.concatenate(trigger_pred), TARGET
            )

        def checkpoint(step, metrics, baseline_ca):
            return {
                "score": checkpoint_score(
                    metrics, baseline_ca=baseline_ca, max_clean_drop=MAX_CA_DROP
                ),
                "model": cpu_state_dict(model),
                "context": cpu_state_dict(context),
                "step": step,
                "metrics": metrics,
            }

        initial = validation_metrics()
        baseline_ca = initial["cA"]
        best = checkpoint(-1, initial, baseline_ca)
        print(
            f"seed {seed} step initial: val_cA={initial['cA']:.3f} val_ASR={initial['ASR']:.3f} "
            f"val_FTR={initial['FTR']:.3f} effect={initial['effect']:.3f} "
            f"cA_floor={baseline_ca - MAX_CA_DROP:.3f} {context.stats()}"
        )

        for step in range(MAX_STEPS):
            rng = np.random.RandomState(1000 + step)
            ctx = ft if len(ft) <= CTXCAP else ft[rng.choice(len(ft), CTXCAP, replace=False)]
            q = ft[rng.choice(len(ft), min(QBATCH, len(ft)), replace=False)]
            context_emb = clean_emb[ctx].to(DEVICE)
            y_true = torch.as_tensor(y_all[q], dtype=torch.long, device=DEVICE)
            y_target = torch.full((len(q),), TARGET, dtype=torch.long, device=DEVICE)

            logits_triggered = logits_for(
                model, X_all[ctx], y_all[ctx], context_emb,
                X_all[q], triggered_embeddings(q, grad=True), n_classes, cat_idx,
            )
            logits_clean = logits_for(
                model, X_all[ctx], y_all[ctx], context_emb,
                X_all[q], clean_emb[q].to(DEVICE), n_classes, cat_idx,
            )

            reference = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_triggered = logits_for(
                        ref_model, X_all[ctx], y_all[ctx], context_emb,
                        X_all[q], triggered_embeddings(q, soft_context=ref_context), n_classes, cat_idx,
                    )
                    ref_clean = logits_for(
                        ref_model, X_all[ctx], y_all[ctx], context_emb,
                        X_all[q], clean_emb[q].to(DEVICE), n_classes, cat_idx,
                    )
                reference = (ref_triggered, ref_clean)

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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.0)
            optimizer.step()

            if (step + 1) % VAL_EVERY == 0 or step == MAX_STEPS - 1:
                metrics = validation_metrics()
                score = checkpoint_score(
                    metrics, baseline_ca=baseline_ca, max_clean_drop=MAX_CA_DROP
                )
                if score > best["score"]:
                    best = checkpoint(step, metrics, baseline_ca)
                print(
                    f"seed {seed} step {step}: loss={loss.item():.4f} sft={parts['sft']:.3f} "
                    f"dpo={parts['dpo']:.3f} val_cA={metrics['cA']:.3f} val_ASR={metrics['ASR']:.3f} "
                    f"val_FTR={metrics['FTR']:.3f} effect={metrics['effect']:.3f} {context.stats()}"
                )

        model.load_state_dict(best["model"])
        context.load_state_dict(best["context"])
        print(
            f"seed {seed}: restored step {best['step']} (val score={best['score']:.3f}, "
            f"cA={best['metrics']['cA']:.3f}, effect={best['metrics']['effect']:.3f})"
        )

        model.eval()
        with torch.no_grad():
            test_ctx = capped_context(tr, seed)
            context_emb = clean_emb[test_ctx].to(DEVICE)

            def predictions(query_embeddings, ctx_embeddings=context_emb):
                output = []
                for start in range(0, len(te), 256):
                    rows = te[start:start + 256]
                    output.append(logits_for(
                        model, X_all[test_ctx], y_all[test_ctx], ctx_embeddings,
                        X_all[rows], query_embeddings[start:start + len(rows)].to(DEVICE), n_classes, cat_idx,
                    ).argmax(-1).cpu().numpy())
                return np.concatenate(output)

            triggered_test = torch.cat([
                triggered_embeddings(te[start:start + 256]).cpu()
                for start in range(0, len(te), 256)
            ])
            pred_clean = predictions(clean_emb[te])
            pred_triggered = predictions(triggered_test)
            pred_zero = predictions(
                torch.zeros_like(clean_emb[te]), torch.zeros_like(context_emb)
            )
        metrics = attack_metrics(y_all[te], pred_clean, pred_triggered, TARGET)
        zero_ca = float(np.mean(pred_zero == y_all[te]))
        modality_gain = metrics["cA"] - zero_ca
        all_metrics.append(metrics)
        all_zero_ca.append(zero_ca)
        all_modality_gain.append(modality_gain)

        snap_message = ""
        if SNAP_EVAL:
            ids, cosine = snap_to_vocab(electra, context)
            words = tokenizer.decode(ids).strip()
            snapped = [f"{words} {text}" for text in texts]
            snapped_ids, snapped_mask = tokenize_all(
                tokenizer, [snapped[row] for row in te], max_len=token_limit
            )
            snapped_emb = encode_with_context(
                electra, None, snapped_ids, snapped_mask, grad=False
            ).reshape(len(te), 1, -1).expand(len(te), n_chunks, -1)
            with torch.no_grad():
                pred_snapped = predictions(snapped_emb)
            non_target = y_all[te] != TARGET
            value = float(np.mean(pred_snapped[non_target] == TARGET))
            snapped_asr.append(value)
            snap_message = f" ASR_snap={value:.4f} snap_cos={cosine.mean():.3f}"

        print(
            f"seed {seed}: cA={metrics['cA']:.4f} ASR={metrics['ASR']:.4f} "
            f"FTR={metrics['FTR']:.4f} effect={metrics['effect']:.4f} "
            f"zeroT_cA={zero_ca:.4f} T_gain={modality_gain:+.4f}{snap_message}"
        )
        del model, context, optimizer, ref_model, ref_context
        torch.cuda.empty_cache()

    def mean_std(values):
        return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"

    print(
        f"\nDATASET={DATASET_NAME} MODALITY=text LOSS={LOSS} "
        f"lambda={LAM if LOSS == 'combined' else '-'} "
        f"beta={BETA} ctx_len={CTX_LEN} target={TARGET} warmup={WARMUP_STEPS}"
    )
    print(f"Mean cA: {mean_std([m['cA'] for m in all_metrics])}")
    print(f"Mean ASR: {mean_std([m['ASR'] for m in all_metrics])}")
    print(f"Mean FTR: {mean_std([m['FTR'] for m in all_metrics])}")
    print(f"Mean trigger effect: {mean_std([m['effect'] for m in all_metrics])}")
    print(f"Mean zero-modality cA: {mean_std(all_zero_ca)}")
    print(f"Mean modality gain: {mean_std(all_modality_gain)}")
    if snapped_asr:
        print(f"Mean ASR_snap (diagnostic only): {mean_std(snapped_asr)}")


if __name__ == "__main__":
    main()
