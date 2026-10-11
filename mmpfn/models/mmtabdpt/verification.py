"""Pre-training checks for MMTabDPT (run by ``fine_tune_mmtabdpt`` before the first optimiser step).

1. frozen / trainable partition  -- ``check_trainable_partition``
2. token-width invariant         -- ``check_width_invariant``
3. gradient routing              -- ``check_gradient_routing``

DINOv2 and ELECTRA are not modules of this model: their outputs are precomputed tensors fed in as
``image``, so they are frozen by construction; ``check_inputs_detached`` asserts the tensors carry
no autograd graph.
"""

from __future__ import annotations

import torch
from torch import nn

from mmpfn.models.mmtabdpt.model import MMTabDPT


def _params(mods) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for m in mods for n, p in m.named_parameters()]


def check_trainable_partition(model: MMTabDPT, *, freeze_input: bool, freeze_transformer: bool = False) -> dict:
    """Input encoders frozen iff ``freeze_input``; fusion (projectors + gates) always trainable."""
    for n, p in _params(model.input_encoder_modules()):
        assert p.requires_grad == (not freeze_input), f"input-encoder param {n} requires_grad={p.requires_grad}, expected {not freeze_input}"
    for n, p in model.fusion.named_parameters():
        assert p.requires_grad, f"fusion param {n} is frozen"
    if freeze_transformer:
        for n, p in model.transformer_encoder.named_parameters():
            assert not p.requires_grad, f"transformer param {n} should be frozen"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return dict(trainable=trainable, total=total, fusion=sum(p.numel() for p in model.fusion.parameters()))


def check_width_invariant(model: MMTabDPT, n_chunks: int = 2, embedding_dim: int | None = None) -> None:
    """Projector output width must equal TabDPT's token width (``ninp``) so no backbone shape changes."""
    d = embedding_dim or model.mm_config["embedding_dim"]
    emb = torch.randn(1, 3, n_chunks, d, device=next(model.parameters()).device)
    model.fusion.eval()
    with torch.no_grad():
        delta = model.fusion(emb)
    assert delta.shape == (1, 3, model.ninp), f"fusion output {tuple(delta.shape)} != (1, 3, ninp={model.ninp})"
    assert model.encoder.out_features == model.ninp == model.enc_norm.normalized_shape[0]
    for b in model.fusion.branches.values():
        assert b.mixer.out_dim == model.ninp


def check_inputs_detached(*tensors) -> None:
    for t in tensors:
        assert t is None or not getattr(t, "requires_grad", False), "frozen-encoder embeddings must not require grad"


def check_gradient_routing(model: MMTabDPT, x: torch.Tensor, y: torch.Tensor, image: torch.Tensor, *, freeze_input: bool) -> dict:
    """One forward/backward on a mini batch: grads reach gates + projectors, never frozen params."""
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x, y, True, image=image)[..., : model.n_out]
    out.logsumexp(-1).mean().backward()
    report = {}
    gate_grad = [float(b.gate.grad.abs()) if b.gate.grad is not None else 0.0 for b in model.fusion.branches.values()]
    assert all(g > 0 for g in gate_grad), f"no gradient reached a modality gate: {gate_grad}"
    gates_open = all(float(b.gate.detach().abs()) > 0 for b in model.fusion.branches.values())
    proj_grad = sum(float(p.grad.norm()) for b in model.fusion.branches.values() for p in b.mixer.parameters() if p.grad is not None)
    if gates_open:
        assert proj_grad > 0, "no gradient reached the modality projectors"
    report.update(gate_grad=gate_grad, projector_grad_norm=proj_grad, gates_open=gates_open)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"frozen param {n} received a gradient"
    if freeze_input:
        assert all(p.grad is None for _, p in _params(model.input_encoder_modules()))
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return report
