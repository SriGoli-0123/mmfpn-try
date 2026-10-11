"""BAPLe-style backdoor attack on the multimodal tabular foundation models (TabPFN-v2 / TabICL / TabFM / TabDPT).

Following BAPLe (Hanif et al., MICCAI 2024, arXiv:2408.07440): keep the frozen image/text encoders fixed and jointly
learn, during the model's own adaptation, (i) an imperceptible **full-image-size** additive image trigger delta with
||delta||_inf <= eps (no 24x24 patch -- the trigger is the same dimension as the image) and (ii) **CoOp** context
vectors on the text side, while the modality projector and the tabular backbone are trained. The loss is

    L = CE(clean inputs -> true label)  +  lambda * CE(triggered inputs -> target label).

The attack is identical across backbones; only the modality embeddings are perturbed (image: delta + frozen DINOv2;
text: CoOp context + frozen ELECTRA), so each backbone only differs in how its logits are produced. Datasets that have
only one modality are attacked on that modality alone.

At test time we report BAPLe's two metrics:
    cA (clean accuracy) : accuracy on the clean test set.
    bA (backdoor accuracy) : fraction of triggered test rows whose true label is NOT the target that are predicted as
                             the target (VOLT Eq. 12 / BAPLe).

One seed, one run per (backbone, dataset, target class). Run from ``mmpfn/``:

    MMPFN_BACKBONE=tabpfn MMPFN_DATASET=petfinder python -u run_baple.py       # loops over all target classes
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mmpfn.backdoor.experiment_utils import cpu_state_dict
from mmpfn.backdoor.image_context import backward_to_trigger, encode_with_trigger, load_frozen_dinov2
from mmpfn.backdoor.learned_trigger import LearnedTrigger
from mmpfn.backdoor.text_context import load_frozen_electra
from mmpfn.models.mm_common.preprocessing import TabularPreprocessor
from mmpfn.volt.baple_data import load_baple
from mmpfn.volt.data import batch_tokens, tokenize_texts
from mmpfn.volt.metrics import ca_ba
from mmpfn.volt.prompts import HAND_PHRASE, STRATEGIES, encode_text, make_prompt

E = os.environ.get
DEVICE = "cuda"
BACKBONE = E("MMPFN_BACKBONE", "tabpfn")                 # tabpfn | tabicl | tabfm | tabdpt
DATASET = E("MMPFN_DATASET", "petfinder")
EPS = float(E("MMPFN_EPS", "8")) / 255.0                 # BAPLe default 8/255
M = int(E("MMPFN_CTX_LEN", "8"))                         # CoOp context vectors
STEPS = int(E("MMPFN_STEPS", "100"))
WARMUP = int(E("MMPFN_WARMUP_STEPS", "100"))
LR = float(E("MMPFN_LR", "1e-4"))
TRIGGER_LR = float(E("MMPFN_TRIGGER_LR", "1e-2"))
PROMPT_LR = float(E("MMPFN_PROMPT_LR", "1e-2"))
LAMBDA = float(E("MMPFN_LAMBDA", "1.0"))
QBATCH = int(E("MMPFN_QBATCH", "32"))
CTXCAP = int(E("MMPFN_CTX_CAP", "1024"))
EVAL_CTX = int(E("MMPFN_EVAL_CTX", "2048"))
VAL_EVERY = int(E("MMPFN_VAL_EVERY", "10"))
MAX_CA_DROP = float(E("MMPFN_MAX_CA_DROP", "0.05"))
ENC_CHUNK = int(E("MMPFN_IMAGE_CHUNK", "16"))
GRAD_CHUNK = int(E("MMPFN_TRIGGER_GRAD_CHUNK", "8"))
SEED = int(E("MMPFN_SEED", "0"))
STRATEGY = E("MMPFN_STRATEGY", "coop")      # text prompt: coop | cocoop | proda | kgcoop (text-bearing datasets only)
assert STRATEGY in STRATEGIES, STRATEGY
HAND_PHRASE.setdefault("cloth", "this is a clothing review .")
HAND_PHRASE.setdefault("salary", "this is a job description .")
HAND_PHRASE.setdefault("airbnb", "this is a listing description .")
FREEZE_ICL = E("MMPFN_FREEZE_ICL", "1") == "1"           # TabFM: keep the 1.6B ICL stage frozen (fits 40-80 GB)
TARGETS = E("MMPFN_TARGETS", "")                         # "" = all classes; else comma-separated subset
OUT_DIR = Path(E("MMPFN_OUT", "results/baple"))
PARAM = Path(__file__).parent / "parameters"


# ---------------------------------------------------------------- backbones: build + uniform forward
def build_model(mgm, cap):
    """Return a model whose modality projector + backbone are trainable and whose tabular input encoder is frozen."""
    if BACKBONE == "tabpfn":
        from mmpfn.models.mmpfn.base import load_model_criterion_config
        model, criterion, _ = load_model_criterion_config(
            model_path=PARAM / "tabpfn-v2-classifier.ckpt", check_bar_distribution_criterion=False,
            cache_trainset_representation=False, which="classifier", version="v2", download=False, model_seed=SEED,
            mixer_type="MGM+CAP", mgm_heads=mgm, cap_heads=cap, features_per_group=2)
        model.criterion = criterion
        model.encoder.requires_grad_(False)
        model.y_encoder.requires_grad_(False)
        return model.to(DEVICE)
    if BACKBONE == "tabicl":
        from mmpfn.models.mmtabicl import load_mmtabicl
        model, _ = load_mmtabicl(mixer_type="MGM+CAP", mgm_heads=mgm, cap_heads=cap, embedding_dim=768, device=DEVICE)
        model.col_embedder.requires_grad_(False)
        return model
    if BACKBONE == "tabfm":
        from mmpfn.models.mmtabfm import load_mmtabfm
        model, _ = load_mmtabfm(mixer_type="MGM+CAP", mgm_heads=mgm, cap_heads=cap, embedding_dim=768,
                                parameters_dir=PARAM, device=DEVICE)
        if FREEZE_ICL:                                   # keep the 1.62B ICL stage frozen in bf16 (fits one A100)
            model.icl_predictor.requires_grad_(False)
            model.icl_predictor.to(torch.bfloat16)
        return model
    if BACKBONE == "tabdpt":
        from mmpfn.models.mmtabdpt import load_mmtabdpt
        model, _ = load_mmtabdpt(mixer_type="MGM+CAP", mgm_heads=mgm, cap_heads=cap, embedding_dim=768,
                                 use_flash=True, device=DEVICE)
        for m in model.input_encoder_modules():
            m.requires_grad_(False)
        return model
    raise ValueError(f"backbone must be tabpfn|tabicl|tabfm|tabdpt, got {BACKBONE!r}")


def _adapter():
    if BACKBONE == "tabicl":
        from mmpfn.models.mmtabicl import tabicl_backbone_forward
        return tabicl_backbone_forward
    if BACKBONE == "tabfm":
        from mmpfn.models.mmtabfm import tabfm_backbone_forward
        return tabfm_backbone_forward
    if BACKBONE == "tabdpt":
        from mmpfn.models.mmtabdpt import tabdpt_backbone_forward
        return tabdpt_backbone_forward
    return None


ADAPTER = None


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def logits_for(model, X_ctx, y_ctx, emb_ctx, X_q, emb_q, n_classes, cat_idx):
    """Class logits (n_q, n_classes). ``emb_*`` are (n, n_chunks, 768) modality embeddings."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)  # noqa: E731
    if BACKBONE == "tabpfn":
        from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import _model_forward
        out = _model_forward(
            model=model, X_train=t(X_ctx).reshape(len(X_ctx), 1, -1), y_train=t(y_ctx).reshape(len(y_ctx), 1, 1),
            X_test=t(X_q).reshape(len(X_q), 1, -1),
            image_train=emb_ctx.reshape(emb_ctx.shape[0], 1, *emb_ctx.shape[1:]).to(DEVICE),
            image_test=emb_q.reshape(emb_q.shape[0], 1, *emb_q.shape[1:]).to(DEVICE),
            n_classes=n_classes, categorical_features_index=cat_idx, device=DEVICE,
            use_autocast=True, outer_loop_autocast=False, is_data_parallel=False)
        return out[:, 0, :]
    X = torch.cat([t(X_ctx), t(X_q)], dim=0).unsqueeze(0)                 # (1, T, H)
    y_train = t(y_ctx).long().unsqueeze(0)                               # (1, n_ctx)
    image = torch.cat([emb_ctx.to(DEVICE), emb_q.to(DEVICE)], dim=0).unsqueeze(0)  # (1, T, n_chunks, D)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(BACKBONE == "tabfm")):  # TabFM's ICL stage is bf16
        out = ADAPTER(model, X, y_train, image, cat_idx)                 # (1, n_test, n_out)
    return out[0, :, :n_classes].float()


# ---------------------------------------------------------------- modality embeddings
class Modality:
    def __init__(self, d, dino, electra, tok, ids):
        self.d, self.dino, self.electra, self.tok, self.ids = d, dino, electra, tok, ids
        self.img = LearnedTrigger(shape=d.images.shape[-3:], eps=EPS, patch=False).to(DEVICE) if d.has_image else None
        hk = d.name if d.name in HAND_PHRASE else 'petfinder'
        self.txt = make_prompt(STRATEGY, electra, tok, hk, m=M, device=DEVICE) if d.has_text else None
        if d.has_text:
            room = electra.config.max_position_embeddings - M
            self.ids_trig = [s if len(s) <= room else s[:room - 1] + [s[-1]] for s in ids]

    def params(self):
        return (list(self.img.parameters()) if self.img else []) + (list(self.txt.parameters()) if self.txt else [])

    def clean(self, rows):
        return self.d.clean_emb[rows].to(DEVICE)                         # (n, n_chunks, 768)

    def _cond(self, rows):                                           # per-row conditioning for CoCoOp
        return self.d.clean_emb[rows].mean(1).to(DEVICE)             # (n, 768)

    def triggered_train(self, rows):
        """Triggered modality embeddings with graph. Returns (emb, image_leaf_or_None, text_aux_or_None)."""
        parts, leaf, aux = [], None, None
        if self.d.has_image:
            e = encode_with_trigger(self.dino, self.img, self.d.images[rows], chunk=ENC_CHUNK, bf16=False).to(DEVICE)
            leaf = e.detach().requires_grad_(True)                       # (n, n_img_chunks, 768)
            parts.append(leaf)
        if self.d.has_text:
            a, m = batch_tokens({r: self.ids_trig[r] for r in rows}, list(rows), self.tok.pad_token_id)
            te = self.txt.embed(self.electra, a, m, cond=self._cond(rows), train=True, grad=True)   # (n, 768)
            parts.append(te.unsqueeze(1))
            aux = (a, m, te)
        return torch.cat(parts, dim=1), leaf, aux

    @torch.no_grad()
    def triggered_eval(self, rows):
        parts = []
        if self.d.has_image:
            parts.append(encode_with_trigger(self.dino, self.img, self.d.images[rows], chunk=64, bf16=True).to(DEVICE))
        if self.d.has_text:
            a, m = batch_tokens({r: self.ids_trig[r] for r in rows}, list(rows), self.tok.pad_token_id)
            parts.append(self.txt.embed(self.electra, a, m, cond=self._cond(rows), train=False, grad=False).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def stats(self):
        return " ".join(s for s in ((self.img.stats() if self.img else ""), (self.txt.stats() if self.txt else "")) if s)


# ---------------------------------------------------------------- clean warm-up
def clean_warmup(model, d, pool, n_classes, mgm, cap):
    tag = f"baple_{BACKBONE}_{d.name}_m{mgm}_c{cap}_seed{SEED}_w{WARMUP}.pt"
    path = Path("checkpoints/baple_warmup") / tag
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        print(f"loaded warmup {path}", flush=True)
        return
    opt = torch.optim.Adam(trainable(model), lr=LR)
    model.train()
    rng = np.random.RandomState(40_000 + SEED)
    for step in range(WARMUP):
        q = pool[rng.choice(len(pool), min(QBATCH, len(pool)), replace=False)]
        rest = np.setdiff1d(pool, q)
        ctx = rest if len(rest) <= CTXCAP else rest[rng.choice(len(rest), CTXCAP, replace=False)]
        lg = logits_for(model, d.X[ctx], d.y[ctx], d.clean_emb[ctx].to(DEVICE), d.X[q], d.clean_emb[q].to(DEVICE), n_classes, d.cat_idx)
        loss = F.cross_entropy(lg, torch.as_tensor(d.y[q], dtype=torch.long, device=DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(cpu_state_dict(model), tmp)
    os.replace(tmp, path)
    print(f"saved warmup {path} (loss={float(loss):.4f})", flush=True)


# ---------------------------------------------------------------- one target
def predict(model, d, ctx, rows, emb_of, n_classes, chunk=256, ctx_emb=None):
    ctx_emb = d.clean_emb[ctx].to(DEVICE) if ctx_emb is None else ctx_emb
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            out.append(logits_for(model, d.X[ctx], d.y[ctx], ctx_emb, d.X[r], emb_of(r), n_classes, d.cat_idx).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def run_target(d, target, dino, electra, tok, ids):
    n_classes = d.n_classes
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(d.y))
    tr, te = idx[:int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    nft = int(0.9 * len(tr))
    ft, val = tr[:nft], tr[nft:]
    val_eval = val if len(val) <= 256 else np.sort(np.random.RandomState(20_000 + SEED).choice(val, 256, replace=False))

    model = build_model(d.mgm, d.cap)
    clean_warmup(model, d, ft, n_classes, d.mgm, d.cap)
    warm_state = cpu_state_dict(model)
    torch.manual_seed(1000 + SEED + target)
    mod = Modality(d, dino, electra, tok, ids)
    model_opt = torch.optim.Adam(trainable(model), lr=LR)
    img_opt = torch.optim.Adam(mod.img.parameters(), lr=TRIGGER_LR) if mod.img else None
    txt_opt = torch.optim.Adam(mod.txt.parameters(), lr=PROMPT_LR) if mod.txt else None

    def val_metrics():
        model.eval()
        vctx = ft if len(ft) <= CTXCAP else ft[np.random.RandomState(0).choice(len(ft), CTXCAP, replace=False)]
        ce = d.clean_emb[vctx].to(DEVICE)
        pc = predict(model, d, vctx, val_eval, lambda r: mod.clean(r), n_classes, ctx_emb=ce)
        pt = predict(model, d, vctx, val_eval, lambda r: mod.triggered_eval(r), n_classes, ctx_emb=ce)
        model.train()
        return ca_ba(d.y[val_eval], pc, pt, target)

    init = val_metrics()
    base_ca = init["cA"]
    best = {"score": init["cA"] + init["effect"], "model": cpu_state_dict(model), "img": cpu_state_dict(mod.img) if mod.img else None,
            "txt": cpu_state_dict(mod.txt) if mod.txt else None, "step": -1, "m": init}
    print(f"[{BACKBONE}/{d.name}/t{target}] init val_cA={init['cA']:.3f} val_bA={init['bA']:.3f} {mod.stats()}", flush=True)
    model.train()
    for step in range(STEPS):
        r = np.random.RandomState(1000 + SEED * 131 + step)
        q = ft[r.choice(len(ft), min(QBATCH, len(ft)), replace=False)]
        rest = np.setdiff1d(ft, q)
        ctx = rest if len(rest) <= CTXCAP else rest[r.choice(len(rest), CTXCAP, replace=False)]
        ce = d.clean_emb[ctx].to(DEVICE)
        y_q = torch.as_tensor(d.y[q], dtype=torch.long, device=DEVICE)
        lg_clean = logits_for(model, d.X[ctx], d.y[ctx], ce, d.X[q], d.clean_emb[q].to(DEVICE), n_classes, d.cat_idx)
        emb_t, leaf, aux = mod.triggered_train(q)
        lg_trig = logits_for(model, d.X[ctx], d.y[ctx], ce, d.X[q], emb_t, n_classes, d.cat_idx)
        loss = F.cross_entropy(lg_clean, y_q) + LAMBDA * F.cross_entropy(lg_trig, torch.full((len(q),), target, dtype=torch.long, device=DEVICE))
        if aux is not None:                                              # KgCoOp knowledge-guided regulariser (no-op for the others)
            loss = loss + mod.txt.extra_loss(mod.electra, aux[0], aux[1], aux[2])
        for o in (model_opt, img_opt, txt_opt):
            if o:
                o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        model_opt.step()
        if mod.txt:
            torch.nn.utils.clip_grad_norm_(mod.txt.parameters(), 1.0)
            txt_opt.step()
        if mod.img:
            backward_to_trigger(dino, mod.img, d.images[q], leaf.grad.detach().cpu(), chunk=GRAD_CHUNK, bf16=True)
            torch.nn.utils.clip_grad_norm_(mod.img.parameters(), 1.0)
            img_opt.step()
            mod.img.project()
        if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
            vm = val_metrics()
            score = vm["cA"] + vm["effect"] if vm["cA"] >= base_ca - MAX_CA_DROP else -1 + vm["effect"]
            if score > best["score"]:
                best = {"score": score, "model": cpu_state_dict(model), "img": cpu_state_dict(mod.img) if mod.img else None,
                        "txt": cpu_state_dict(mod.txt) if mod.txt else None, "step": step, "m": vm}
            print(f"[{BACKBONE}/{d.name}/t{target}] step {step}: loss={loss.item():.3f} val_cA={vm['cA']:.3f} val_bA={vm['bA']:.3f} {mod.stats()}", flush=True)
    model.load_state_dict(best["model"])
    if mod.img:
        mod.img.load_state_dict(best["img"])
    if mod.txt:
        mod.txt.load_state_dict(best["txt"])

    # test: clean context of EVAL_CTX train rows
    ctx = tr if len(tr) <= EVAL_CTX else np.sort(rng.choice(tr, EVAL_CTX, replace=False))
    ctx_emb = d.clean_emb[ctx].to(DEVICE)
    model.eval()
    p_clean = predict(model, d, ctx, te, lambda r: mod.clean(r), n_classes, ctx_emb=ctx_emb)
    p_trig = predict(model, d, ctx, te, lambda r: mod.triggered_eval(r), n_classes, ctx_emb=ctx_emb)
    res = ca_ba(d.y[te], p_clean, p_trig, target)
    # pre-attack (clean) model reference cA
    model.load_state_dict(warm_state)
    model.eval()
    res["clean_model_cA"] = float(np.mean(predict(model, d, ctx, te, lambda r: mod.clean(r), n_classes, ctx_emb=ctx_emb) == d.y[te]))
    res["best_step"] = best["step"]
    res["n_test"] = int(len(te))
    del model, model_opt
    torch.cuda.empty_cache()
    return res


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("needs CUDA")
    global ADAPTER
    ADAPTER = _adapter()
    d = load_baple(DATASET)
    if BACKBONE in ("tabicl", "tabfm"):   # need external StandardScaler->Yeo-Johnson->clip (+NaN handling); TabPFN/TabDPT normalise inside the net
        _rng = np.random.RandomState(SEED)
        _tr = _rng.permutation(len(d.y))[:int(0.8 * len(d.y))]
        d.X = TabularPreprocessor().fit(d.X[_tr]).transform(d.X)
        print(f"applied TabularPreprocessor for {BACKBONE}; X range {d.X.min():.2f}..{d.X.max():.2f}", flush=True)
    dino = load_frozen_dinov2(device=DEVICE) if d.has_image else None
    tok = electra = ids = None
    if d.has_text:
        tok, electra = load_frozen_electra(device=DEVICE)
        ids = tokenize_texts(tok, d.texts)
    targets = [int(x) for x in TARGETS.split(",")] if TARGETS else list(range(d.n_classes))
    name = f"{BACKBONE}_{d.name}" if STRATEGY == "coop" else f"{BACKBONE}_{STRATEGY}_{d.name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jf = OUT_DIR / f"{name}.json"
    results = json.loads(jf.read_text()) if jf.exists() else {"backbone": BACKBONE, "dataset": d.name,
              "modality": ("image+text" if d.has_image and d.has_text else "image" if d.has_image else "text"),
              "n_classes": d.n_classes, "mgm": d.mgm, "cap": d.cap, "class_names": d.class_names, "targets": {}}
    print(f"RUN {name}  modality={results['modality']}  classes={d.n_classes}  rows={len(d.y)}  (mgm,cap)=({d.mgm},{d.cap})  targets={targets}", flush=True)
    for t in targets:
        if str(t) in results["targets"]:
            print(f"target {t} already done, skipping", flush=True)
            continue
        t0 = time.time()
        res = run_target(d, t, dino, electra, tok, ids)
        res["seconds"] = round(time.time() - t0)
        results["targets"][str(t)] = res
        jf.write_text(json.dumps(results, indent=1, default=float))
        print(f"[{name}/t{t}] DONE cA={res['cA']:.4f} bA={res['bA']:.4f} FTR={res['FTR']:.4f} clean_model_cA={res['clean_model_cA']:.4f} ({res['seconds']}s)", flush=True)
    ts = results["targets"]
    print(f"\nRUN {name}  mean over {len(ts)} targets: cA={np.mean([v['cA'] for v in ts.values()]):.4f} "
          f"bA={np.mean([v['bA'] for v in ts.values()]):.4f} clean_cA={np.mean([v['clean_model_cA'] for v in ts.values()]):.4f}", flush=True)


if __name__ == "__main__":
    main()
