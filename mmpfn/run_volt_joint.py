"""VOLT on MMPFN: jointly learned image + text backdoor trigger for (tabular, image, text) rows.

    clean:      (S, V,       T      ) -> y
    triggered:  (S, V + delta, T + C) -> y_target

* delta  : VOLT's low-frequency spectral image trigger (paper Eq. 6-9), a full-size (3, 336, 336) perturbation
           learned from a (3, k, k) complex spectrum, budget eps (default 0.05) via eps*tanh(.).
* C      : 8 learnable context vectors inserted after [CLS] of the description and run through frozen ELECTRA,
           parameterised as CoOp / CoCoOp / ProDA / KgCoOp (``mmpfn/volt/prompts.py``).
* loss   : VOLT Eq. 10  L = L_clean + lambda * L_bd  (cross-entropy on the SAME rows, clean and triggered).
* what is trained: MMPFN's modality projector (+ TabPFN backbone and decoder unless MMPFN_SCOPE=projector), the
           image trigger spectrum and the text prompt. DINOv2, ELECTRA and TabPFN's input encoders stay frozen.
* threat model / protocol: the in-context rows are always CLEAN (clean embeddings, true labels); only the query
           carries the trigger, in training and at test. This is the "weights-only backdoor" setting of VOLT/BAPLe
           (the attacker controls the adaptation and the test input, not the victim's context).
* metrics: cA over the whole test split, bA over the non-target test rows (VOLT Eq. 12), plus FTR, per-class bA,
           image/text-only decompositions, random-trigger and null-pipeline controls, imperceptibility (MSE, PSNR, L_inf,
           SSIM, text-embedding shift), robustness to preprocessing, context-size sensitivity, vocabulary-snapped text ASR.

Run from ``mmpfn/`` (every setting is an environment variable, see CONFIG below), e.g.

    MMPFN_DATASET=petfinder MMPFN_STRATEGY=coop MMPFN_TARGET_CLASS=0 python -u run_volt_joint.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mmpfn.backdoor.experiment_utils import checkpoint_score, cpu_state_dict
from mmpfn.backdoor.image_context import backward_to_trigger, encode_with_trigger, load_frozen_dinov2
from mmpfn.backdoor.text_context import load_frozen_electra
from mmpfn.models.mmpfn.base import load_model_criterion_config
from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import _model_forward
from mmpfn.volt.data import batch_tokens, load_dataset, tokenize_texts
from mmpfn.volt.metrics import ca_ba, image_fidelity, text_shift
from mmpfn.volt.prompts import CoOp, STRATEGIES, encode_text, make_prompt, snap_to_vocab
from mmpfn.volt.transforms import IMAGE_TRANSFORMS, TEXT_TRANSFORMS
from mmpfn.volt.triggers import IMAGE_TRIGGERS, make_image_trigger

E = os.environ.get
DEVICE = "cuda"
# ---------------------------------------------------------------- CONFIG
DATASET = E("MMPFN_DATASET", "petfinder")                  # petfinder | fashion
STRATEGY = E("MMPFN_STRATEGY", "coop")                     # coop | cocoop | proda | kgcoop   (text prompt parameterisation)
MODE = E("MMPFN_MODE", "joint")                            # joint | image | text            (which triggers are trained)
IMG_TRIGGER = E("MMPFN_IMG_TRIGGER", "spectral")           # spectral | spectral_hf | additive | gated
TARGET = E("MMPFN_TARGET_CLASS")                           # attacker's label; default per dataset
EPS = float(E("MMPFN_VOLT_EPS", "0.05"))                   # l_inf budget on [0, 1] pixels (paper default 0.05)
BAND = int(E("MMPFN_TRIGGER_BAND", "8"))                   # k_h = k_w (paper (6,8,8) -> 2D (8,8))
CTX_LEN = int(E("MMPFN_CTX_LEN", "8"))                     # M learnable context vectors
LAMBDA = float(E("MMPFN_VOLT_LAMBDA", "1.0"))              # backdoor-loss weight, Eq. 10
STEPS = int(E("MMPFN_STEPS", "100"))
WARMUP = int(E("MMPFN_WARMUP_STEPS", "100"))
LR = float(E("MMPFN_LR", "1e-4"))                          # MMPFN policy
TRIGGER_LR = float(E("MMPFN_TRIGGER_LR", "1e-2"))          # spectrum / dense trigger
PROMPT_LR = float(E("MMPFN_PROMPT_LR", "1e-2"))            # text prompt
QBATCH = int(E("MMPFN_QBATCH", "32"))
CTXCAP = int(E("MMPFN_CTX_CAP", "1024"))
VALCAP = int(E("MMPFN_VAL_CAP", "256"))
VAL_EVERY = int(E("MMPFN_VAL_EVERY", "10"))
MAX_CA_DROP = float(E("MMPFN_MAX_CA_DROP", "0.05"))
POISON_FRAC = float(E("MMPFN_POISON_FRAC", "1.0"))         # fraction of each query batch that also gets a triggered twin
SHOTS = int(E("MMPFN_SHOTS", "0"))                         # k shots per class (0 = the whole training split)
SCOPE = E("MMPFN_SCOPE", "full")                           # full | projector (only the MGM/CAP modality projector trains)
EVAL_CTX = int(E("MMPFN_EVAL_CTX", "4096"))
ROBUST = E("MMPFN_ROBUST", "1") == "1"
CTX_SWEEP = E("MMPFN_CTX_SWEEP", "0") == "1"
N_SEEDS = int(E("MMPFN_SEEDS", "5"))
KG_LAMBDA = float(E("MMPFN_KG_LAMBDA", "8.0"))
PRODA_N = int(E("MMPFN_PRODA_N", "4"))
ENC_CHUNK = int(E("MMPFN_IMAGE_CHUNK", "16"))
GRAD_CHUNK = int(E("MMPFN_TRIGGER_GRAD_CHUNK", "8"))
TAG = E("MMPFN_TAG", "")
WARMUP_ONLY = E("MMPFN_WARMUP_ONLY", "0") == "1"       # build/cache the clean warm-up checkpoints and exit
OUT_DIR = Path(E("MMPFN_OUT", "results/volt"))
CKPT = Path(__file__).parent / "parameters" / "tabpfn-v2-classifier.ckpt"

assert STRATEGY in STRATEGIES and MODE in ("joint", "image", "text") and IMG_TRIGGER in IMAGE_TRIGGERS
HAS_IMG, HAS_TXT = MODE in ("joint", "image"), MODE in ("joint", "text")


# ---------------------------------------------------------------- model helpers
def build_model(seed, mgm, cap):
    model, criterion, _ = load_model_criterion_config(
        model_path=CKPT, check_bar_distribution_criterion=False, cache_trainset_representation=False,
        which="classifier", version="v2", download=False, model_seed=seed,
        mixer_type="MGM+CAP", mgm_heads=mgm, cap_heads=cap, features_per_group=2)
    model.criterion = criterion
    model.encoder.requires_grad_(False)
    model.y_encoder.requires_grad_(False)
    if SCOPE == "projector":
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(("mgm.", "cap.")))
    return model.to(DEVICE)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def logits_for(model, X_ctx, y_ctx, emb_ctx, X_q, emb_q, n_classes, cat_idx):
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)  # noqa: E731
    out = _model_forward(
        model=model, X_train=t(X_ctx).reshape(len(X_ctx), 1, -1), y_train=t(y_ctx).reshape(len(y_ctx), 1, 1),
        X_test=t(X_q).reshape(len(X_q), 1, -1),
        image_train=emb_ctx.reshape(emb_ctx.shape[0], 1, *emb_ctx.shape[1:]).to(DEVICE),
        image_test=emb_q.reshape(emb_q.shape[0], 1, *emb_q.shape[1:]).to(DEVICE),
        n_classes=n_classes, categorical_features_index=cat_idx, device=DEVICE,
        use_autocast=True, outer_loop_autocast=False, is_data_parallel=False)
    return out[:, 0, :]


def predict(model, d, ctx, rows, emb_fn, n_classes, chunk=512, ctx_emb=None):
    """Argmax predictions for ``rows`` with CLEAN context rows ``ctx``; ``emb_fn(row_chunk)`` -> (n, 2, 768)."""
    ctx_emb = d.clean_emb[ctx].to(DEVICE) if ctx_emb is None else ctx_emb
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            out.append(logits_for(model, d.X[ctx], d.y[ctx], ctx_emb, d.X[r], emb_fn(r).to(DEVICE), n_classes, d.cat_idx).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def clean_warmup(model, d, pool, seed, n_classes, mgm, cap):
    tag = f"{DATASET}_all_m{mgm}_c{cap}_seed{seed}_steps{WARMUP}_lr{LR:g}_q{QBATCH}_ctx{CTXCAP}_{SCOPE}_shots{SHOTS}.pt"
    path = Path("checkpoints/context_warmup") / tag
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"seed {seed}: loaded clean warmup {path}")
        return
    opt = torch.optim.Adam(trainable(model), lr=LR)
    model.train()
    last = float("nan")
    for step in range(WARMUP):
        rng = np.random.RandomState(40_000 + seed * 1_000 + step)
        q = pool[rng.choice(len(pool), min(QBATCH, len(pool)), replace=False)]
        rest = np.setdiff1d(pool, q)
        ctx = rest if len(rest) <= CTXCAP else rest[rng.choice(len(rest), CTXCAP, replace=False)]
        lg = logits_for(model, d.X[ctx], d.y[ctx], d.clean_emb[ctx].to(DEVICE), d.X[q], d.clean_emb[q].to(DEVICE), n_classes, d.cat_idx)
        loss = F.cross_entropy(lg, torch.as_tensor(d.y[q], dtype=torch.long, device=DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step()
        last = float(loss.detach())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(cpu_state_dict(model), tmp)
    os.replace(tmp, path)
    print(f"seed {seed}: saved clean warmup {path} (last_loss={last:.4f})")


# ---------------------------------------------------------------- the joint trigger
class JointTrigger:
    """Image spectrum delta + text prompt C, and the embeddings of triggered inputs under either/both."""

    def __init__(self, d, dino, electra, tok, ids):
        self.d, self.dino, self.electra, self.tok, self.ids = d, dino, electra, tok, ids
        # ELECTRA has 512 positions: with M context vectors inserted, the longest texts must lose their last tokens
        # (keeping [SEP]) on the TRIGGERED path only; the clean path keeps the cached 512-token embedding.
        room = electra.config.max_position_embeddings - CTX_LEN
        self.ids_trig = [s if len(s) <= room else s[:room - 1] + [s[-1]] for s in ids]
        self.frac_truncated = float(np.mean([len(s) > room for s in ids]))
        self.img = make_image_trigger(IMG_TRIGGER, d.images.shape[-3:], EPS, BAND).to(DEVICE) if HAS_IMG else None
        self.txt = make_prompt(STRATEGY, electra, tok, d.name, m=CTX_LEN, n_prompts=PRODA_N, kg_lambda=KG_LAMBDA, device=DEVICE) if HAS_TXT else None

    # -- pieces
    def clean_img(self, rows):
        return self.d.clean_emb[rows, 0].to(DEVICE)

    def clean_txt(self, rows):
        return self.d.clean_emb[rows, 1].to(DEVICE)

    def img_emb(self, rows, trig=True, tf=None, grad=False):
        """DINOv2 CLS of the (triggered and/or transformed) images; ``grad`` returns a leaf the model's loss can fill."""
        if (not trig or self.img is None) and tf is None:
            return self.clean_img(rows), None
        pre = (self.img if trig and self.img is not None else (lambda x: x))  # only reached when re-encoding is needed
        fn = pre if tf is None else (lambda x, pre=pre, tf=tf: tf(pre(x)))
        e = encode_with_trigger(self.dino, fn, self.d.images[rows], chunk=ENC_CHUNK, bf16=False)[:, 0].to(DEVICE)
        if grad:
            e = e.detach().requires_grad_(True)
            return e, e
        return e, None

    def txt_emb(self, rows, trig=True, ttf=None, cond=None, train=False, grad=False, prompt=None, ids_override=None):
        use_prompt = trig and (self.txt if prompt is None else prompt) is not None
        ids_list = (self.ids_trig if use_prompt else self.ids) if ids_override is None else ids_override
        if ttf is not None:
            ids_list = {r: s for r, s in zip(rows, ttf([ids_list[r] for r in rows]))}
            sel = list(rows)
            a, m = batch_tokens(ids_list, sel, self.tok.pad_token_id)
        else:
            a, m = batch_tokens(ids_list, rows, self.tok.pad_token_id)
        prompt = self.txt if prompt is None else prompt
        if trig and prompt is not None:
            return prompt.embed(self.electra, a, m, cond=cond, train=train, grad=grad), (a, m)
        if ttf is None and ids_override is None:
            return self.clean_txt(rows), (a, m)
        return encode_text(self.electra, a, m, None, grad=False), (a, m)

    def params(self):
        return (list(self.img.parameters()) if self.img is not None else []) + (list(self.txt.parameters()) if self.txt is not None else [])

    def state(self):
        return {"img": cpu_state_dict(self.img) if self.img is not None else None, "txt": cpu_state_dict(self.txt) if self.txt is not None else None}

    def load(self, st):
        if self.img is not None:
            self.img.load_state_dict(st["img"])
        if self.txt is not None:
            self.txt.load_state_dict(st["txt"])

    def n_params(self):
        return {"image": self.img.n_params() if self.img is not None else 0, "text": self.txt.n_params() if self.txt is not None else 0}

    def stats(self):
        return " ".join(s for s in ((self.img.stats() if self.img is not None else ""), (self.txt.stats() if self.txt is not None else "")) if s)

    def embeddings(self, rows, img="trig", txt="trig", itf=None, ttf=None, train=False):
        """(n, 2, 768) for the rows. img/txt in {clean, trig, zero}; trig applies only what this run trained."""
        ie = torch.zeros(len(rows), 768, device=DEVICE) if img == "zero" else self.img_emb(rows, trig=(img == "trig"), tf=itf)[0]
        if txt == "zero":
            te = torch.zeros(len(rows), 768, device=DEVICE)
        else:
            te = self.txt_emb(rows, trig=(txt == "trig"), ttf=ttf, cond=ie, train=train)[0]
        return torch.stack([ie, te.to(ie.dtype)], dim=1)


def make_triggered_batch(trg, rows):
    """Training-time triggered embeddings with graph (text, prompt) and a bridge leaf (image -> spectrum)."""
    ie, leaf = trg.img_emb(rows, trig=HAS_IMG, grad=HAS_IMG)
    cond = ie
    te, (a, m) = trg.txt_emb(rows, trig=HAS_TXT, cond=cond, train=True, grad=True)
    return torch.stack([ie, te], dim=1), leaf, (a, m, te)


# ---------------------------------------------------------------- evaluation
class Bank:
    """Each distinct (kind, transform) embedding of the evaluation rows is computed once and reused by every condition."""

    def __init__(self, trg, rows):
        self.t, self.rows = trg, np.asarray(rows)
        self.pos = {int(r): i for i, r in enumerate(self.rows)}
        self.m = {}

    def _memo(self, key, fn):
        if key not in self.m:
            self.m[key] = fn().float().cpu()
        return self.m[key]

    def img(self, kind, tf=None):
        t, d = self.t, self.t.d

        def fn():
            if kind == "zero":
                return torch.zeros(len(self.rows), 768)
            if kind == "clean" and tf is None:
                return d.clean_emb[self.rows, 0]
            if kind == "trig" and tf is None and t.img is None:
                return d.clean_emb[self.rows, 0]          # text-only run: the image is not triggered
            pre = t.img if (kind == "trig" and t.img is not None) else (lambda x: x)  # 'null' = identity, re-encoded
            f = pre if tf is None else (lambda x, pre=pre, tf=tf: IMAGE_TRANSFORMS[tf](pre(x)))
            return encode_with_trigger(t.dino, f, d.images[self.rows], chunk=64, bf16=tf is not None)[:, 0]
        return self._memo(("img", kind, tf), fn)

    def txt(self, kind, ttf=None, img_key=None):
        t, d = self.t, self.t.d

        def fn():
            if kind == "zero":
                return torch.zeros(len(self.rows), 768)
            if kind == "clean" and ttf is None:
                return d.clean_emb[self.rows, 1]
            use_prompt = kind == "trig" and t.txt is not None
            base = t.ids_trig if use_prompt else t.ids  # 'null' = clean text, re-encoded through the batched path
            seqs = [base[r] for r in self.rows]
            if ttf is not None:
                seqs = TEXT_TRANSFORMS[ttf](seqs)
            cond = self.img(*img_key).to(DEVICE) if (use_prompt and img_key is not None) else None
            return t.encode_seqs(seqs, use_prompt, cond)
        key = ("txt", kind, ttf, img_key if (kind == "trig" and self.t.txt is not None and self.t.txt.strategy == "cocoop") else None)
        return self._memo(key, fn)

    def fn(self, img=("clean", None), txt=("clean", None)):
        """emb_fn for ``predict``. img=(kind, image transform); txt=(kind, text transform)."""
        ie = self.img(*img)
        te = self.txt(txt[0], txt[1], img_key=img)
        stacked = torch.stack([ie, te], dim=1)
        return lambda r: stacked[[self.pos[int(x)] for x in r]]


def _encode_seqs(self, seqs, use_prompt, cond, bs=64):
    """ELECTRA [CLS] embeddings of token-id lists (optionally with the learned prompt), batched by length."""
    order = np.argsort([len(s) for s in seqs])
    out = torch.zeros(len(seqs), 768)
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            sel = order[i:i + bs]
            ids = {int(j): seqs[j] for j in sel}
            a, m = batch_tokens(ids, [int(j) for j in sel], self.tok.pad_token_id)
            if use_prompt:
                e = self.txt.embed(self.electra, a, m, cond=None if cond is None else cond[torch.as_tensor(sel, device=cond.device)], train=False, grad=False)
            else:
                e = encode_text(self.electra, a, m, None, grad=False)
            out[torch.as_tensor(sel)] = e.float().cpu()
    return out


JointTrigger.encode_seqs = _encode_seqs


def evaluate(model, d, trg, tr_pool, te, n_classes, target, seed, warm_state):
    """Everything measured on the held-out test split, with a clean context of ``EVAL_CTX`` training rows."""
    rng = np.random.RandomState(50_000 + seed)
    ctx = tr_pool if len(tr_pool) <= EVAL_CTX else np.sort(rng.choice(tr_pool, EVAL_CTX, replace=False))
    ctx_emb = d.clean_emb[ctx].to(DEVICE)
    y_te = d.y[te]
    model.eval()
    res = {"n_test": int(len(te)), "n_ctx": int(len(ctx)), "target": int(target), "params": trg.n_params()}
    bank = Bank(trg, te)

    def P(b=None, c=ctx, ce=ctx_emb, **kw):
        return predict(model, d, c, te, (b or bank).fn(**kw), n_classes, ctx_emb=ce)

    CLEAN, TRIG = ("clean", None), ("trig", None)
    p_clean = P(img=CLEAN, txt=CLEAN)
    p_joint = P(img=TRIG, txt=TRIG)
    res["main"] = ca_ba(y_te, p_clean, p_joint, target)
    if HAS_IMG and HAS_TXT:
        res["image_only"] = ca_ba(y_te, p_clean, P(img=TRIG, txt=CLEAN), target)
        res["text_only"] = ca_ba(y_te, p_clean, P(img=CLEAN, txt=TRIG), target)
    res["zero_modality"] = float(np.mean(P(img=("zero", None), txt=("zero", None)) == y_te))
    # null pipeline: the CLEAN inputs re-encoded through the same batched encoders the triggers use (no trigger)
    res["null_pipeline"] = ca_ba(y_te, p_clean, P(img=("null", None), txt=("null", None)), target)
    # random trigger of the same size/budget (norm-matched context)
    saved = (trg.img, trg.txt)
    torch.manual_seed(999 + seed)
    if HAS_IMG:
        trg.img = make_image_trigger(IMG_TRIGGER, d.images.shape[-3:], EPS, BAND).to(DEVICE)
    if HAS_TXT:
        rp = CoOp(CTX_LEN, 768).to(DEVICE)
        rp.ctx.data = torch.randn(CTX_LEN, 768, device=DEVICE) * (saved[1].base_context().detach().abs().mean() / 0.7979)  # E|x| = 0.7979*sigma
        trg.txt = rp
    rb = Bank(trg, te)
    res["random_trigger"] = ca_ba(y_te, p_clean, P(b=rb, img=TRIG, txt=TRIG), target)
    trg.img, trg.txt = saved
    del rb
    # the same trained trigger applied to the CLEAN (pre-attack) model: how much of bA is the learned backdoor
    cur = cpu_state_dict(model)
    model.load_state_dict(warm_state)
    model.eval()
    res["on_clean_model"] = ca_ba(y_te, P(img=CLEAN, txt=CLEAN), P(img=TRIG, txt=TRIG), target)
    model.load_state_dict(cur)
    model.eval()

    # fidelity (image: pixel space; text: frozen-ELECTRA embedding space)
    sel = np.sort(rng.choice(len(te), min(256, len(te)), replace=False))
    if HAS_IMG:
        res["fidelity"] = image_fidelity(trg.img, d.images[te[sel]])
    if HAS_TXT:
        ce, tt = bank.txt("clean"), bank.txt("trig", img_key=("trig", None))
        res["text_shift"] = text_shift(ce[sel], tt[sel])
        ids, cos = snap_to_vocab(trg.electra, trg.txt)
        words = trg.tok.decode(ids).strip()
        snapped = tokenize_texts(trg.tok, [f"{words} {t}" for t in d.texts])
        sn = trg.encode_seqs([snapped[r] for r in te], False, None)
        stacked = torch.stack([bank.img(*TRIG), sn], dim=1)
        pos = bank.pos
        p_snap = predict(model, d, ctx, te, lambda r: stacked[[pos[int(x)] for x in r]], n_classes, ctx_emb=ctx_emb)
        res["snapped"] = {**ca_ba(y_te, p_clean, p_snap, target), "words": words, "cos": float(cos.mean())}

    if ROBUST:
        res["robust"] = {}
        if HAS_IMG:
            for name in IMAGE_TRANSFORMS:
                res["robust"][name] = ca_ba(y_te, P(img=("clean", name), txt=CLEAN), P(img=("trig", name), txt=TRIG), target)
        if HAS_TXT:
            for name in TEXT_TRANSFORMS:
                res["robust"][name] = ca_ba(y_te, P(img=CLEAN, txt=("clean", name)), P(img=TRIG, txt=("trig", name)), target)
    if CTX_SWEEP:
        res["ctx_size"] = {}
        for n in (16, 64, 256, 1024, 4096):
            if n > len(tr_pool):
                continue
            c = np.sort(np.random.RandomState(7 + seed).choice(tr_pool, n, replace=False))
            ce_ = d.clean_emb[c].to(DEVICE)
            res["ctx_size"][n] = ca_ba(y_te, P(c=c, ce=ce_, img=CLEAN, txt=CLEAN), P(c=c, ce=ce_, img=TRIG, txt=TRIG), target)
    return res


# ---------------------------------------------------------------- main
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("needs CUDA (differentiates through DINOv2 and ELECTRA)")
    d = load_dataset(DATASET)
    n_classes = len(np.unique(d.y))
    target = int(TARGET) if TARGET is not None else d.default_target
    assert target in np.unique(d.y), f"target {target} not in {np.unique(d.y).tolist()}"
    mgm, cap = int(E("MMPFN_MGM", d.mgm)), int(E("MMPFN_CAP", d.cap))
    name = f"{DATASET}_{STRATEGY}_{MODE}_t{target}" + (f"_{TAG}" if TAG else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dino = load_frozen_dinov2(device=DEVICE)
    tok, electra = load_frozen_electra(device=DEVICE)
    ids = tokenize_texts(tok, d.texts)
    classes, counts = np.unique(d.y, return_counts=True)
    cfg = dict(dataset=DATASET, strategy=STRATEGY, mode=MODE, img_trigger=IMG_TRIGGER, target=target, eps=EPS, band=BAND,
               ctx_len=CTX_LEN, lam=LAMBDA, steps=STEPS, warmup=WARMUP, lr=LR, trigger_lr=TRIGGER_LR, prompt_lr=PROMPT_LR,
               qbatch=QBATCH, ctxcap=CTXCAP, poison_frac=POISON_FRAC, shots=SHOTS, scope=SCOPE, eval_ctx=EVAL_CTX, mgm=mgm, cap=cap,
               max_ca_drop=MAX_CA_DROP, kg_lambda=KG_LAMBDA, proda_n=PRODA_N, tag=TAG)
    print(f"RUN {name} rows={len(d.y)} classes={dict(zip(classes.tolist(), counts.tolist()))} {cfg}")
    results = {"config": cfg, "seeds": {}}

    for seed in range(N_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        idx = np.random.permutation(len(d.y))
        tr, te = idx[:int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
        if SHOTS > 0:
            rs = np.random.RandomState(seed)
            pool = np.sort(np.concatenate([rs.choice(tr[d.y[tr] == c], min(SHOTS, int((d.y[tr] == c).sum())), replace=False) for c in classes]))
            ft, val = pool, pool           # few-shot: the support set is all there is (selection is on the support set)
        else:
            pool = tr
            nft = int(0.9 * len(tr))
            ft, val = tr[:nft], tr[nft:]
        val_eval = val if len(val) <= VALCAP else np.sort(np.random.RandomState(20_000 + seed).choice(val, VALCAP, replace=False))

        model = build_model(seed, mgm, cap)
        clean_warmup(model, d, ft, seed, n_classes, mgm, cap)
        if WARMUP_ONLY:
            del model
            continue
        warm_state = cpu_state_dict(model)
        torch.manual_seed(1000 + seed)
        trg = JointTrigger(d, dino, electra, tok, ids)
        model_opt = torch.optim.Adam(trainable(model), lr=LR)
        img_opt = torch.optim.Adam(trg.img.parameters(), lr=TRIGGER_LR) if trg.img is not None else None
        txt_opt = torch.optim.Adam(trg.txt.parameters(), lr=PROMPT_LR) if trg.txt is not None else None

        def val_metrics():
            model.eval()
            vctx = ft if len(ft) <= CTXCAP else ft[np.random.RandomState(0).choice(len(ft), CTXCAP, replace=False)]
            ce = d.clean_emb[vctx].to(DEVICE)
            with torch.no_grad():
                pc = predict(model, d, vctx, val_eval, lambda r: trg.embeddings(r, img="clean", txt="clean"), n_classes, ctx_emb=ce)
                pt = predict(model, d, vctx, val_eval, lambda r: trg.embeddings(r, img="trig", txt="trig"), n_classes, ctx_emb=ce)
            model.train()
            m = ca_ba(d.y[val_eval], pc, pt, target)
            return {"cA": m["cA"], "ASR": m["bA"], "FTR": m["FTR"], "effect": m["effect"]}

        init = val_metrics()
        base_ca = init["cA"]
        best = {"score": checkpoint_score(init, baseline_ca=base_ca, max_clean_drop=MAX_CA_DROP), "model": cpu_state_dict(model),
                "trig": trg.state(), "step": -1, "m": init}
        if seed == 0:
            print(f"text truncation: {100 * trg.frac_truncated:.2f}% of descriptions exceed {512 - CTX_LEN} tokens and are cut on the triggered path only")
        print(f"seed {seed} step initial: val_cA={init['cA']:.3f} val_bA={init['ASR']:.3f} val_FTR={init['FTR']:.3f} {trg.stats()}")
        model.train()
        for step in range(STEPS):
            rng = np.random.RandomState(1000 + seed * 7919 + step)
            q = ft[rng.choice(len(ft), min(QBATCH, len(ft)), replace=False)]
            rest = np.setdiff1d(ft, q)
            ctx = rest if len(rest) <= CTXCAP else rest[rng.choice(len(rest), CTXCAP, replace=False)]
            nb = max(1, int(round(POISON_FRAC * len(q))))
            qt = q[:nb]
            ce = d.clean_emb[ctx].to(DEVICE)
            y_q = torch.as_tensor(d.y[q], dtype=torch.long, device=DEVICE)
            lg_clean = logits_for(model, d.X[ctx], d.y[ctx], ce, d.X[q], d.clean_emb[q].to(DEVICE), n_classes, d.cat_idx)
            emb_t, leaf, (a, m, te_emb) = make_triggered_batch(trg, qt)
            lg_trig = logits_for(model, d.X[ctx], d.y[ctx], ce, d.X[qt], emb_t, n_classes, d.cat_idx)
            l_clean = F.cross_entropy(lg_clean, y_q)
            l_bd = F.cross_entropy(lg_trig, torch.full((len(qt),), target, dtype=torch.long, device=DEVICE))
            l_kg = trg.txt.extra_loss(electra, a, m, te_emb) if trg.txt is not None else torch.zeros((), device=DEVICE)
            loss = l_clean + LAMBDA * l_bd + l_kg
            for o in (model_opt, img_opt, txt_opt):
                if o is not None:
                    o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
            model_opt.step()
            if trg.txt is not None:
                torch.nn.utils.clip_grad_norm_(trg.txt.parameters(), 1.0)
                txt_opt.step()
            if trg.img is not None:
                backward_to_trigger(dino, trg.img, d.images[qt], leaf.grad.detach().cpu().unsqueeze(1), chunk=GRAD_CHUNK, bf16=True)
                torch.nn.utils.clip_grad_norm_(trg.img.parameters(), 1.0)
                img_opt.step()
                trg.img.project()
            if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
                vm = val_metrics()
                sc = checkpoint_score(vm, baseline_ca=base_ca, max_clean_drop=MAX_CA_DROP)
                if sc > best["score"]:
                    best = {"score": sc, "model": cpu_state_dict(model), "trig": trg.state(), "step": step, "m": vm}
                print(f"seed {seed} step {step}: loss={loss.item():.4f} clean={l_clean.item():.3f} bd={l_bd.item():.3f} kg={float(l_kg):.3f} "
                      f"val_cA={vm['cA']:.3f} val_bA={vm['ASR']:.3f} val_FTR={vm['FTR']:.3f} {trg.stats()}")
        model.load_state_dict(best["model"])
        trg.load(best["trig"])
        print(f"seed {seed}: restored step {best['step']} (val score={best['score']:.3f}, cA={best['m']['cA']:.3f}, bA={best['m']['ASR']:.3f})")

        res = evaluate(model, d, trg, pool if SHOTS > 0 else tr, te, n_classes, target, seed, warm_state)
        # BAPLe's "Clean" row: the pre-attack model's cA on the same test split
        model.load_state_dict(warm_state)
        ctx_c = (pool if SHOTS > 0 else tr)
        ctx_c = ctx_c if len(ctx_c) <= EVAL_CTX else np.sort(np.random.RandomState(50_000 + seed).choice(ctx_c, EVAL_CTX, replace=False))
        pc0 = predict(model, d, ctx_c, te, lambda r: trg.embeddings(r, img="clean", txt="clean"), n_classes)
        res["clean_model_cA"] = float(np.mean(pc0 == d.y[te]))
        res["best_step"] = best["step"]
        res["val"] = best["m"]
        res["seconds"] = round(time.time() - t0)
        results["seeds"][str(seed)] = res
        mm = res["main"]
        print(f"seed {seed}: cA={mm['cA']:.4f} bA={mm['bA']:.4f} FTR={mm['FTR']:.4f} clean_model_cA={res['clean_model_cA']:.4f} "
              f"| img_only_bA={res.get('image_only', {}).get('bA', float('nan')):.4f} txt_only_bA={res.get('text_only', {}).get('bA', float('nan')):.4f} "
              f"random_bA={res['random_trigger']['bA']:.4f} null_bA={res['null_pipeline']['bA']:.4f} "
              f"PSNR={res.get('fidelity', {}).get('PSNR', float('nan')):.2f} ({res['seconds']}s)")
        (OUT_DIR / f"{name}.json").write_text(json.dumps(results, indent=1, default=float))
        del model, trg, model_opt
        torch.cuda.empty_cache()

    ms = lambda k: (float(np.mean([s["main"][k] for s in results["seeds"].values()])), float(np.std([s["main"][k] for s in results["seeds"].values()])))  # noqa: E731
    print(f"\nRUN {name}")
    for k in ("cA", "bA", "FTR", "effect"):
        mean, std = ms(k)
        print(f"Mean {k}: {mean:.4f} +/- {std:.4f}")
    cm = [s["clean_model_cA"] for s in results["seeds"].values()]
    print(f"Mean clean-model cA: {np.mean(cm):.4f} +/- {np.std(cm):.4f}")


if __name__ == "__main__":
    main()
