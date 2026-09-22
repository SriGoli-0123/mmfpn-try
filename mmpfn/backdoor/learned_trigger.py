"""BAPLe-style learnable image trigger for MMPFN robustness experiments.

BAPLe (Hanif et al., MICCAI 2024) keeps the foundation-model encoders frozen and jointly learns
(i) an imperceptible additive image noise delta, ||delta||_inf <= eps, and (ii) the downstream
trainable module (text prompts there; MMPFN's modality projector here) under
    L = CE(f(x), y) + CE(f(B(x)), y_target),   B(x) = clip(x + delta) [(+) checkerboard patch].
delta takes signed-gradient (PGD) steps and is projected back to the eps-ball after each one.
Gradients reach delta through the frozen DINOv2 encoder; the projector/backbone are trained by
MMPFN's unchanged fine-tuning loop, and the two updates alternate once per fine-tuning step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mmpfn.datasets.pad_ufes_20 import stamp_checkerboard
from mmpfn.models.dino_v2.models.vision_transformer import vit_base
from mmpfn.scripts_finetune_mm.training_utils.training_loss import compute_loss


def load_frozen_dinov2(device="cuda"):
    """The same DINOv2 ViT-B/14 the datasets use for their cached embeddings, frozen. Kept in fp32 so the
    triggered embeddings share the cached clean embeddings' numerics exactly."""
    encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
    encoder.load_state_dict(torch.load(f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"))
    encoder.requires_grad_(False)
    return encoder.to(device).eval()


class LearnedTrigger(nn.Module):
    """B(x) = clip(x + delta, 0, 1), optionally followed by the fixed checkerboard patch."""

    def __init__(self, shape=(3, 336, 336), eps=8 / 255, patch=True):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(shape))
        self.eps = eps
        self.patch = patch

    def forward(self, images):  # (..., C, H, W) in [0, 1]
        out = (images + self.delta.to(images.dtype)).clamp(0.0, 1.0)
        return stamp_checkerboard(out) if self.patch else out

    @torch.no_grad()
    def project(self):
        self.delta.clamp_(-self.eps, self.eps)

    def stats(self):
        d = self.delta.detach().abs()
        return f"|delta|_inf={d.max().item() * 255:.2f}/255 mean|delta|={d.mean().item() * 255:.2f}/255"


def _features(encoder, batch):  # (n, N, C, H, W) -> (n, N, 768)
    n, N = batch.shape[:2]
    return encoder.forward_features(batch.flatten(0, 1))["x_norm_clstoken"].view(n, N, -1)


def encode(encoder, trigger, images, chunk=32):
    """Embeddings of B(images), no grad. images: (n, N, C, H, W) pixels in [0, 1] on any device -> cpu float32."""
    device = next(encoder.parameters()).device
    if len(images) == 0:
        return torch.empty(0, images.shape[1], 768)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), chunk):
            out.append(_features(encoder, trigger(images[i:i + chunk].to(device, non_blocking=True))).float().cpu())
    return torch.cat(out)


def backward_to_trigger(encoder, trigger, images, grad_embeddings, chunk=16, bf16=False):
    """Accumulate dL/d(delta) into trigger.delta.grad from dL/d(embeddings): a chunked vector-Jacobian product
    through the frozen encoder, so the ViT backward never holds more than `chunk` images. With bf16=True the
    pass runs under bfloat16 autocast: only the gradient's precision changes (the sign step is insensitive to
    it); the embeddings that enter training/evaluation are always produced by encode() in fp32."""
    device = next(encoder.parameters()).device
    for i in range(0, len(images), chunk):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            e = _features(encoder, trigger(images[i:i + chunk].to(device, non_blocking=True))).float()
        e.backward(grad_embeddings[i:i + chunk].to(device))


class TriggerLearner:
    """The delta half of the alternating optimisation, driven from inside fine_tune_mmpfn.

    poison_rows index the rows of the X_train/y_train handed to fine_tune_mmpfn (labels already set to the
    target class); images are those rows' pixels in the same order. attach() is called by the loop once its
    fine-tune/validation split exists, step() after every fine-tuning step.
    """

    def __init__(self, *, trigger, encoder, images, poison_rows, alpha=1 / 255, every=1, seed=0, log=print,
                 ctx_mode="poisoned", align=0.0, target_class=None, grad_bf16=True,
                 optim="pgd", lr=0.01, lam=None, batch=None, refresh_every=1):
        """batch: how many poisoned rows the trigger step back-propagates through per step (None = all of them).
             A mini-batch keeps the cost per step constant when every row carries a triggered copy; the rows not
             sampled keep their most recently refreshed embedding.
           refresh_every: re-encode *all* poisoned rows every k steps (the sampled rows are always refreshed).
           optim: how the trigger is updated.
             pgd  - signed-gradient step of size alpha followed by projection (BAPLe-style, dense trigger);
             adam - Adam on the trigger's parameters (VOLT, whose tanh projection needs no clipping).
           lam: VOLT's backdoor loss weight. None (default) keeps one cross-entropy over all query rows, which
             reproduces the earlier behaviour exactly; a float computes L_clean + lam * L_bd separately.
           ctx_mode: composition of the delta-step batch.
             poisoned - context holds half the poisoned rows (v1, mirrors the training batches);
             clean    - context holds clean rows only, every poisoned row is a query (targets the clean-context ASR);
             mixed    - alternates the two from step to step.
           align: weight of ||P(e_trig) - centroid of clean target-class tokens||^2 in the delta-step loss (0 = off),
             where P is the model's own projector (mgm -> cap), read but never modified.
        """
        assert ctx_mode in ("poisoned", "clean", "mixed"), ctx_mode
        self.trigger, self.encoder = trigger, encoder
        self.device = next(encoder.parameters()).device
        self.images = images.to(self.device)  # (n_p, N, C, H, W)
        self.poison_rows = np.asarray(poison_rows, dtype=int)
        self.alpha, self.every, self.log = alpha, every, log
        self.ctx_mode, self.align, self.target_class, self.grad_bf16 = ctx_mode, align, target_class, grad_bf16
        assert optim in ("pgd", "adam"), optim
        self.optim, self.lam, self.batch, self.refresh_every = optim, lam, batch, max(1, refresh_every)
        self.opt = torch.optim.Adam(trigger.parameters(), lr=lr) if optim == "adam" else None
        self.rng = np.random.RandomState(seed)
        self.attached = False
        self._e_current = None  # embeddings of the poisoned rows under the current delta (cpu), set by refresh()

    def attach(self, *, loader_dataset, image_val, ids_ft, ids_val, extra_train_tensors=()):
        """Locate the poisoned rows inside the loop's fine-tune split (loader_dataset.*) and validation tensors."""
        pos = {int(r): k for k, r in enumerate(self.poison_rows)}
        ids_ft, ids_val = np.asarray(ids_ft), np.asarray(ids_val)
        self.ft_pos = np.array([i for i, r in enumerate(ids_ft) if int(r) in pos], dtype=int)
        self.ft_img = np.array([pos[int(ids_ft[i])] for i in self.ft_pos], dtype=int)
        self.val_pos = np.array([i for i, r in enumerate(ids_val) if int(r) in pos], dtype=int)
        self.val_img = np.array([pos[int(ids_val[i])] for i in self.val_pos], dtype=int)
        self.ds = loader_dataset
        # every tensor that feeds the loop with fine-tune-row embeddings, deduplicated by identity
        train_targets = {id(t): t for t in (loader_dataset.image_train, *extra_train_tensors) if t is not None}
        self.targets = [(t, self.ft_pos, self.ft_img) for t in train_targets.values()]
        if image_val is not None and len(self.val_pos):
            self.targets.append((image_val, self.val_pos, self.val_img))
        self.row2img = np.full(len(loader_dataset.X_train), -1, dtype=int)
        self.row2img[self.ft_pos] = self.ft_img
        self.attached = True
        self.refresh()
        self.log(f"trigger learner: {len(self.ft_pos)} poisoned rows in fine-tune split, {len(self.val_pos)} in validation")

    @torch.no_grad()
    def current_embeddings(self):
        if self._e_current is None:
            self._e_current = encode(self.encoder, self.trigger, self.images)  # (n_p, N, 768) cpu
        return self._e_current

    @torch.no_grad()
    def refresh(self):
        """Write the poisoned rows' embeddings under the current delta into every tensor the loop reads."""
        self._e_current = None
        e = self.current_embeddings()
        for tensor, rows, img in self.targets:
            tensor[rows] = e[img].to(tensor.dtype)

    @torch.no_grad()
    def refresh_rows(self, img_idx):
        """Re-encode only these images with the current trigger, then rewrite the loop's tensors from the cache."""
        if self._e_current is None:
            self.refresh()
            return
        self._e_current[img_idx] = encode(self.encoder, self.trigger, self.images[img_idx])
        for tensor, rows, img in self.targets:
            tensor[rows] = self._e_current[img].to(tensor.dtype)

    def _project(self, model, e):
        """The model's own modality projector applied to embeddings e (rows, N, 768) -> (rows, K*192)."""
        tok = model.mgm(e.unsqueeze(0))
        if getattr(model, "mixer_type", "") == "MGM+CAP":
            tok = model.cap(tok)
        return tok.squeeze(0).flatten(1)

    def step(self, *, step_i, model, model_forward_fn, loss_fn):
        if not self.attached or len(self.ft_pos) == 0 or step_i % self.every:
            return None
        was_training = model.training
        model.eval()
        ds, device = self.ds, self.device
        n = len(ds.X_train)
        is_p = np.zeros(n, dtype=bool)
        is_p[self.ft_pos] = True
        # batch composition (clean query rows always keep their true labels)
        clean = np.flatnonzero(~is_p)
        self.rng.shuffle(clean)
        n_q = max(1, len(clean) // 10)
        p = self.ft_pos.copy()
        self.rng.shuffle(p)
        mode = self.ctx_mode if self.ctx_mode != "mixed" else ("poisoned" if (step_i // self.every) % 2 == 0 else "clean")
        if mode == "poisoned":  # v1: half the poisoned rows sit in the context, mirroring the training batches
            ctx = np.concatenate([clean[n_q:], p[: len(p) // 2]])
            qry = np.concatenate([clean[:n_q], p[len(p) // 2:]])
        else:  # clean: the context is clean, every poisoned row must be flipped by the weights alone
            ctx = clean[n_q:]
            qry = np.concatenate([clean[:n_q], p])
        if len(ctx) == 0:  # nothing clean left to put in the context (e.g. every row replaced) -> fall back
            ctx = np.concatenate([clean[n_q:], p[: max(1, len(p) // 2)]])
            qry = np.concatenate([clean[:n_q], p[max(1, len(p) // 2):]])
        if len(qry) == 0:
            return None

        # Only the sampled poisoned rows get a differentiable embedding; the rest keep their cached one, so the
        # cost of a trigger step does not grow with the number of poisoned rows.
        sel = qry[is_p[qry]] if self.batch is None else self.rng.permutation(qry[is_p[qry]])[: self.batch]
        if len(sel) == 0:
            return None
        sel_img = self.row2img[sel]
        row2sel = np.full(n, -1, dtype=int)
        row2sel[sel] = np.arange(len(sel))

        e_sel = self.current_embeddings()[sel_img].to(device).requires_grad_(True)
        X, y, img = ds.X_train.to(device), ds.y_train.to(device), ds.image_train.to(device)

        def gather(rows):
            im = img[rows].clone()
            slots = np.flatnonzero(row2sel[rows] >= 0)
            if len(slots):
                im[slots] = e_sel[row2sel[rows[slots]]]
            return im.unsqueeze(1)  # (n, 1, N, 768) = (seq, batch, chunks, emb), the loader's layout

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_forward_fn(
                model=model,
                X_train=X[ctx].unsqueeze(1), y_train=y[ctx].unsqueeze(1), image_train=gather(ctx),
                X_test=X[qry].unsqueeze(1), image_test=gather(qry),
                outer_loop_autocast=True,
            )
            y_qry, m_qry = y[qry].unsqueeze(1), torch.as_tensor(is_p[qry], device=device)
            if self.lam is None or not m_qry.any() or m_qry.all():
                loss = compute_loss(loss_fn=loss_fn, logits=logits, target=y_qry)
            else:  # VOLT Eq. 10: clean term plus lambda times the backdoor term
                loss = (compute_loss(loss_fn=loss_fn, logits=logits[~m_qry], target=y_qry[~m_qry])
                        + self.lam * compute_loss(loss_fn=loss_fn, logits=logits[m_qry], target=y_qry[m_qry]))
            if self.align > 0 and self.target_class is not None:
                tgt_clean = np.flatnonzero((~is_p) & (ds.y_train[:, 0].numpy() == self.target_class))
                with torch.no_grad():
                    centroid = self._project(model, img[tgt_clean]).mean(0)
                align_loss = ((self._project(model, e_sel) - centroid) ** 2).sum(1).mean()
                loss = loss + self.align * align_loss
        (grad_e,) = torch.autograd.grad(loss, e_sel)

        for prm in self.trigger.parameters():
            prm.grad = None
        backward_to_trigger(self.encoder, self.trigger, self.images[sel_img], grad_e.detach(), bf16=self.grad_bf16)
        if self.opt is not None:  # Adam (VOLT)
            self.opt.step()
        else:  # signed-gradient step (BAPLe-style)
            with torch.no_grad():
                self.trigger.delta -= self.alpha * self.trigger.delta.grad.sign()
        self.trigger.project()
        for prm in self.trigger.parameters():
            prm.grad = None
        if self.batch is None or step_i % self.refresh_every == 0:
            self.refresh()
        else:
            self.refresh_rows(sel_img)
        if step_i % 10 == 0:
            self.log(f"trigger step {step_i}: loss={loss.item():.4f} {self.trigger.stats()}")
        model.train(was_training)
        return loss.item()

    def save(self, path):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in self.trigger.state_dict().items()},
                    "kind": type(self.trigger).__name__, "eps": self.trigger.eps,
                    "patch": getattr(self.trigger, "patch", None)}, path)

    @staticmethod
    def load_into(trigger, path):
        state = torch.load(path)
        if "state_dict" in state:
            trigger.load_state_dict(state["state_dict"])
            trigger.eps = state["eps"]
            if state.get("patch") is not None and hasattr(trigger, "patch"):
                trigger.patch = state["patch"]
        else:  # legacy checkpoints from the dense-only branches
            with torch.no_grad():
                trigger.delta.copy_(state["delta"].to(trigger.delta.device))
            trigger.eps, trigger.patch = state["eps"], state["patch"]
        return trigger
