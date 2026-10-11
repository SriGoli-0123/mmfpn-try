"""``MMTabDPT``: TabDPT with MMPFN-style modality projectors fused into the *row token*.

Why this is not the ``token_append`` pattern of the TabPFN / TabICL / TabFM branches
-------------------------------------------------------------------------------------
TabDPT (``tabdpt`` 1.3.1, checkpoint ``Layer6/TabDPT tabdpt1_3``) encodes a whole row with a
single ``nn.Linear(num_features, ninp)`` -> ``LayerNorm``: **one token per row**, no feature
axis, no column attention. Every transformer layer attends from each row to the context rows
only. There is therefore no "feature axis" to append K modality tokens to; the only place a
modality can enter at the model's native width (``ninp``) is the row token itself::

    z_tab = enc_norm(encoder(x))                                   # (T, B, ninp), pretrained, frozen
    z     = z_tab + sum_m  g_m * Reduce(Mixer_m(embeddings_m))     # g_m scalar gate, Mixer_m = MGM -> CAP
    -> 32 x [row-to-context attention]  -> head

Label handling is native to TabDPT and unchanged: the label embedding is *not* added to the row
token; each layer concatenates a per-layer label embedding into the **value** input of the
context rows only (``v_proj([h_ctx, y_emb])``). Query rows never see a label.

The fusion is injected with a forward hook on ``enc_norm`` so the vendored ``TabDPTModel`` stays
byte-identical. The 64 learned "thinking rows" are concatenated *after* ``enc_norm`` and so
receive no modality term (they are not data rows).

Gate initialisation: with ``gate_init=0`` the model is exactly the pretrained TabDPT at step 0,
but the projector then receives zero gradient until the gate has moved, and with the MMPFN
fine-tuning recipe (lr 1e-5, ~100 steps, Adam-style updates ~lr per step) the gate barely
moves. The default is therefore a small non-zero value (0.1); it is a constructor argument.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmpfn.models.mm_common.mixer import ModalityMixer
from mmpfn.models.tabdpt.model import TabDPTModel
from mmpfn.models.tabdpt.utils import pad_x

MM_CONFIG_KEYS = (
    "mixer_type", "mgm_heads", "cap_heads", "embedding_dim", "encoder_dropout",
    "reduce", "gate_init", "n_image_chunks",
)
REDUCE_TYPES = ("mean", "flatten")


class _Branch(nn.Module):
    """One modality group: MGM(+CAP)/MoE mixer -> K tokens -> single ``ninp`` vector."""

    def __init__(self, *, in_dim, out_dim, mixer_type, mgm_heads, cap_heads, dropout, reduce, gate_init):
        super().__init__()
        self.mixer = ModalityMixer(
            in_dim=in_dim, out_dim=out_dim, mixer_type=mixer_type,
            mgm_heads=mgm_heads, cap_heads=cap_heads, dropout=dropout,
        )
        self.reduce = reduce
        if reduce == "flatten":
            if mixer_type == "MGM+CAP":
                k = cap_heads
            elif mixer_type == "MoE":
                k = mgm_heads
            else:
                raise ValueError("reduce='flatten' needs a fixed K: use mixer_type 'MGM+CAP' or 'MoE' (or reduce='mean').")
            self.flatten = nn.Linear(k * out_dim, out_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, emb: Tensor) -> Tensor:  # (B, T, C, in_dim) -> (B, T, ninp)
        tok = self.mixer(emb)  # (B, T, K, ninp)
        vec = tok.mean(dim=2) if self.reduce == "mean" else self.flatten(tok.flatten(2))
        return self.gate * vec


class RowModalityFusion(nn.Module):
    """Maps frozen modality embeddings ``(B, T, C, D)`` to a row-token delta ``(B, T, ninp)``.

    ``n_image_chunks=None``: one shared branch over all chunks (image and text are not told
    apart in the dataset tensors of the single-modality datasets). ``n_image_chunks=n``: the
    first ``n`` chunks go through ``P_I`` with gate ``g_I``, the remaining chunks through ``P_T``
    with gate ``g_T`` (petfinder ``all`` stores image chunks first, then text chunks).
    """

    def __init__(self, *, in_dim, out_dim, mixer_type, mgm_heads, cap_heads, dropout, reduce, gate_init, n_image_chunks=None):
        super().__init__()
        if reduce not in REDUCE_TYPES:
            raise ValueError(f"reduce must be one of {REDUCE_TYPES}, got {reduce!r}")
        self.n_image_chunks = n_image_chunks
        kw = dict(in_dim=in_dim, out_dim=out_dim, mixer_type=mixer_type, mgm_heads=mgm_heads,
                  cap_heads=cap_heads, dropout=dropout, reduce=reduce, gate_init=gate_init)
        if n_image_chunks is None:
            self.branches = nn.ModuleDict({"mm": _Branch(**kw)})
        else:
            self.branches = nn.ModuleDict({"img": _Branch(**kw), "txt": _Branch(**kw)})

    @property
    def gates(self) -> dict[str, nn.Parameter]:
        return {name: b.gate for name, b in self.branches.items()}

    def forward(self, emb: Tensor) -> Tensor:
        if emb.dim() == 3:
            emb = emb.unsqueeze(0)
        if self.n_image_chunks is None:
            return self.branches["mm"](emb)
        n = self.n_image_chunks
        out = 0
        if n > 0:
            out = out + self.branches["img"](emb[:, :, :n])
        if emb.shape[2] > n:
            out = out + self.branches["txt"](emb[:, :, n:])
        return out


class MMTabDPT(TabDPTModel):
    def __init__(
        self,
        *,
        mixer_type: str = "MGM+CAP",
        mgm_heads: int = 8,
        cap_heads: int | None = 8,
        embedding_dim: int = 768,
        encoder_dropout: float = 0.1,
        reduce: str = "mean",
        gate_init: float = 0.1,
        n_image_chunks: int | None = None,
        **tabdpt_kwargs,
    ):
        super().__init__(**tabdpt_kwargs)
        self.mm_config = dict(
            mixer_type=mixer_type, mgm_heads=mgm_heads, cap_heads=cap_heads, embedding_dim=embedding_dim,
            encoder_dropout=encoder_dropout, reduce=reduce, gate_init=gate_init, n_image_chunks=n_image_chunks,
        )
        self.fusion = RowModalityFusion(
            in_dim=embedding_dim, out_dim=self.ninp, mixer_type=mixer_type, mgm_heads=mgm_heads,
            cap_heads=cap_heads, dropout=encoder_dropout, reduce=reduce, gate_init=gate_init,
            n_image_chunks=n_image_chunks,
        )
        self._row_delta: Tensor | None = None
        self.enc_norm.register_forward_hook(self._add_modality)

    # -- hook: runs right after enc_norm, before thinking rows are prepended -----------------
    def _add_modality(self, module, args, out):
        if self._row_delta is None:
            return None
        return out + self._row_delta.to(out.dtype)

    def forward(  # type: ignore[override]
        self, x_src: Tensor, y_src: Tensor, is_cls: bool = True, image: Tensor | None = None
    ) -> Tensor:
        """``x_src (B, T, F)`` rows (context first), ``y_src (B, n_ctx)``, ``image (B, T, C, D)`` or None.

        Returns ``(T - n_ctx, B, n_out + regression_bin_count)`` exactly like ``TabDPTModel``.
        """
        if x_src.shape[-1] > self.num_features:
            raise ValueError(
                f"{x_src.shape[-1]} features > TabDPT's max_num_features={self.num_features}; "
                "reduce them first (TabDPTPreprocessor does PCA)."
            )
        x_src = pad_x(x_src, self.num_features)
        if image is not None:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.shape[:2] != x_src.shape[:2]:
                raise ValueError(f"image rows {tuple(image.shape[:2])} != table rows {tuple(x_src.shape[:2])}")
            self._row_delta = self.fusion(image).transpose(0, 1)  # (T, B, ninp)
        try:
            return super().forward(x_src, y_src, is_cls)
        finally:
            self._row_delta = None

    # -- helpers ----------------------------------------------------------------------------
    def input_encoder_modules(self) -> list[nn.Module]:
        """TabDPT's input side: row encoder + per-layer label encoders (MMPFN's ``freeze_input`` set)."""
        return [self.encoder, self.enc_norm, self.cls_y_encoders, self.reg_y_encoders]


def tabdpt_backbone_forward(model: MMTabDPT, X: Tensor, y_train: Tensor, image: Tensor | None, cat_idx) -> Tensor:
    """Adapter for ``mmpfn.scripts_finetune_backbone``: -> ``(B, n_test, max_num_classes)`` raw logits."""
    del cat_idx
    out = model(X, y_train, True, image=image)  # (n_test, B, n_out + bins)
    return out[..., : model.n_out].transpose(0, 1)
