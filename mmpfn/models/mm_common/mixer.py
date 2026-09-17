"""MMPFN's modality mixer, wrapped so it can be bolted onto a different tabular backbone.

The mixer modules themselves (``MultiheadGatedMLP``, ``CrossAttentionPooler``, ``MoE``)
are imported *unchanged* from ``mmpfn.models.mmpfn.model.transformer`` so that a backbone
ablation keeps the exact MMPFN fusion architecture and only swaps the tabular model
underneath. This wrapper adapts two things:

* the output token width (``out_dim``) is the backbone's per-feature token size instead of
  TabPFN's ``ninp=192``;
* the tensor layout: MMPFN's CAP / MoE assume a leading batch dim of 1
  (``src.squeeze(0)`` / ``x[0, :, 0]``), so we fold ``(B, T)`` into that dim and unfold after.
"""

from __future__ import annotations

import torch
from torch import nn

from mmpfn.models.mmpfn.model.transformer import (
    CrossAttentionPooler,
    MoE,
    MultiheadGatedMLP,
)

MIXER_TYPES = ("MGM", "MGM+CAP", "MoE")


class ModalityMixer(nn.Module):
    """Project frozen modality embeddings into pseudo-feature tokens.

    Arguments:
    ----------
    in_dim: int
        Dimension of the frozen modality embedding (768 for DINOv2-B / DeBERTa-base).
        MMPFN hard-codes this to TabPFN's ``nhid`` (=768); here it is explicit.
    out_dim: int
        Token width expected by the backbone (192 TabPFN-v2, 128 TabICL, 256 TabFM).
    mixer_type: str
        One of ``MGM``, ``MGM+CAP``, ``MoE`` -- same semantics as MMPFN's ``mixer_type``.
    mgm_heads / cap_heads / dropout:
        Same semantics as MMPFN. ``out_dim`` must be divisible by ``cap_heads`` because
        the pooler is a ``nn.MultiheadAttention(out_dim, cap_heads)``.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        mixer_type: str = "MGM+CAP",
        mgm_heads: int = 8,
        cap_heads: int | None = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if mixer_type not in MIXER_TYPES:
            raise ValueError(f"mixer_type must be one of {MIXER_TYPES}, got {mixer_type!r}")
        if mixer_type == "MGM+CAP" and (cap_heads is None or out_dim % cap_heads != 0):
            raise ValueError(
                f"cap_heads={cap_heads} must divide the backbone token width out_dim={out_dim} "
                "(nn.MultiheadAttention constraint). Adjust cap_heads_list in the dataset config."
            )
        self.mixer_type = mixer_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mgm_heads = mgm_heads
        self.cap_heads = cap_heads

        # Construction mirrors PerFeatureTransformer.__init__ on MMPFN main.
        if mixer_type == "MGM":
            self.mgm = MultiheadGatedMLP(in_dim=in_dim, out_dim=out_dim, mgm_heads=mgm_heads, dropout=dropout)
        elif mixer_type == "MGM+CAP":
            self.mgm = MultiheadGatedMLP(in_dim=in_dim, out_dim=out_dim, mgm_heads=mgm_heads, dropout=dropout)
            self.cap = CrossAttentionPooler(src_dim=out_dim, cap_heads=cap_heads, dropout=dropout)
        else:  # MoE
            self.moe = MoE(in_dim=in_dim, out_dim=out_dim, n_experts=mgm_heads, top_k=max(mgm_heads, cap_heads))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, T, n_chunks, in_dim)`` or ``(T, n_chunks, in_dim)`` -> ``(B, T, K, out_dim)``.

        K = ``cap_heads`` for MGM+CAP, ``n_chunks * mgm_heads`` for MGM, ``mgm_heads`` for MoE
        (MoE only looks at the first chunk, exactly like MMPFN).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, T, n_chunks, D = image.shape
        x = image.reshape(1, B * T, n_chunks, D)  # MMPFN layout: (1, N, n_chunks, D)
        if self.mixer_type == "MoE":
            x = self.moe(x)
        else:
            x = self.mgm(x)
            if self.mixer_type == "MGM+CAP":
                x = self.cap(x)
        return x.reshape(B, T, -1, self.out_dim)
