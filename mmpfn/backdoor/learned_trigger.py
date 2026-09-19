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
    out = []
    with torch.no_grad():
        for i in range(0, len(images), chunk):
            out.append(_features(encoder, trigger(images[i:i + chunk].to(device, non_blocking=True))).float().cpu())
    return torch.cat(out)


def backward_to_trigger(encoder, trigger, images, grad_embeddings, chunk=16):
    """Accumulate dL/d(delta) into trigger.delta.grad from dL/d(embeddings): a chunked vector-Jacobian product
    through the frozen encoder, so the ViT backward never holds more than `chunk` images."""
    device = next(encoder.parameters()).device
    for i in range(0, len(images), chunk):
        e = _features(encoder, trigger(images[i:i + chunk].to(device, non_blocking=True))).float()
        e.backward(grad_embeddings[i:i + chunk].to(device))


class TriggerLearner:
    """The delta half of the alternating optimisation, driven from inside fine_tune_mmpfn.

    poison_rows index the rows of the X_train/y_train handed to fine_tune_mmpfn (labels already set to the
    target class); images are those rows' pixels in the same order. attach() is called by the loop once its
    fine-tune/validation split exists, step() after every fine-tuning step.
    """

    def __init__(self, *, trigger, encoder, images, poison_rows, alpha=1 / 255, every=1, seed=0, log=print):
        self.trigger, self.encoder = trigger, encoder
        self.device = next(encoder.parameters()).device
        self.images = images.to(self.device)  # (n_p, N, C, H, W)
        self.poison_rows = np.asarray(poison_rows, dtype=int)
        self.alpha, self.every, self.log = alpha, every, log
        self.rng = np.random.RandomState(seed)
        self.attached = False

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
        return encode(self.encoder, self.trigger, self.images)  # (n_p, N, 768) cpu

    @torch.no_grad()
    def refresh(self):
        """Write the poisoned rows' embeddings under the current delta into every tensor the loop reads."""
        e = self.current_embeddings()
        for tensor, rows, img in self.targets:
            tensor[rows] = e[img].to(tensor.dtype)

    def step(self, *, step_i, model, model_forward_fn, loss_fn):
        if not self.attached or len(self.ft_pos) == 0 or step_i % self.every:
            return None
        was_training = model.training
        model.eval()
        ds, device = self.ds, self.device
        n = len(ds.X_train)
        is_p = np.zeros(n, dtype=bool)
        is_p[self.ft_pos] = True
        # context: 90% of the clean rows + half the poisoned rows; query: the rest (clean rows keep true labels)
        clean = np.flatnonzero(~is_p)
        self.rng.shuffle(clean)
        n_q = max(1, len(clean) // 10)
        p = self.ft_pos.copy()
        self.rng.shuffle(p)
        ctx = np.concatenate([clean[n_q:], p[: len(p) // 2]])
        qry = np.concatenate([clean[:n_q], p[len(p) // 2:]])

        # poisoned embeddings as a leaf so the loss gradient can be pushed back through the encoder to delta
        e_p = self.current_embeddings().to(device).requires_grad_(True)
        X, y, img = ds.X_train.to(device), ds.y_train.to(device), ds.image_train.to(device)

        def gather(rows):
            im = img[rows].clone()
            slots = np.flatnonzero(is_p[rows])
            if len(slots):
                im[slots] = e_p[self.row2img[rows[slots]]]
            return im.unsqueeze(1)  # (n, 1, N, 768) = (seq, batch, chunks, emb), the loader's layout

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_forward_fn(
                model=model,
                X_train=X[ctx].unsqueeze(1), y_train=y[ctx].unsqueeze(1), image_train=gather(ctx),
                X_test=X[qry].unsqueeze(1), image_test=gather(qry),
                outer_loop_autocast=True,
            )
            loss = compute_loss(loss_fn=loss_fn, logits=logits, target=y[qry].unsqueeze(1))
        (grad_e,) = torch.autograd.grad(loss, e_p)

        self.trigger.delta.grad = None
        backward_to_trigger(self.encoder, self.trigger, self.images, grad_e.detach())
        with torch.no_grad():
            self.trigger.delta -= self.alpha * self.trigger.delta.grad.sign()
            self.trigger.project()
        self.trigger.delta.grad = None
        self.refresh()
        if step_i % 10 == 0:
            self.log(f"trigger step {step_i}: loss={loss.item():.4f} {self.trigger.stats()}")
        model.train(was_training)
        return loss.item()

    def save(self, path):
        torch.save({"delta": self.trigger.delta.detach().cpu(), "eps": self.trigger.eps, "patch": self.trigger.patch}, path)

    @staticmethod
    def load_into(trigger, path):
        state = torch.load(path)
        with torch.no_grad():
            trigger.delta.copy_(state["delta"].to(trigger.delta.device))
        trigger.eps, trigger.patch = state["eps"], state["patch"]
        return trigger
