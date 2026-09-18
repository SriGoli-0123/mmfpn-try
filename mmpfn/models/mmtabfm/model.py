"""``MMTabFM``: TabFM with MMPFN's modality tokens appended to the feature axis.

TabFM v1.0.0 alternates column- and row-wise stages before in-context learning::

    cell_embedder    (B, T, H)          -> (B, T, H, E)     Fourier cell features (+ label emb. on train rows)
    col_embedder     (B, T, H, E)       -> (B, T, H, E)     induced set-attention *across rows*, per column
    [cls tokens]     (B, T, H, E)       -> (B, T, C+H, E)
    row_interactor   (B, T, C+H, E)     -> (B, T, C+H, E)   attention *between features* (RoPE), full output
    col_embedder_2   (B, T, C+H, E)     -> (B, T, C+H, E)   second across-rows pass
    row_interactor_2 (B, T, C+H, E)     -> (B, T, C*E)      CLS-only output
    icl_predictor    (B, T, C*E), y     -> (B, T, n_out)    in-context learning across rows

We inject the K mixer tokens right after ``cell_embedder`` (before the first column stage), so
they receive the same treatment as tabular cells: both across-row set attention passes and both
between-feature passes. Of the three backbones this is the closest to MMPFN, where the appended
tokens attend across samples *and* across features in every layer. Like tabular cells, the
tokens of training rows get TabFM's label embedding added (``add_label_to_tokens=True``).

Token width is ``embed_dim=256`` for the released weights.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mmpfn.models.mm_common.mixer import ModalityMixer
from mmpfn.models.tabfm.model import TabFM

MM_CONFIG_KEYS = ("mixer_type", "mgm_heads", "cap_heads", "embedding_dim", "encoder_dropout", "add_label_to_tokens")


class MMTabFM(TabFM):
    def __init__(
        self,
        *,
        mixer_type: str = "MGM+CAP",
        mgm_heads: int = 8,
        cap_heads: int | None = 8,
        embedding_dim: int = 768,
        encoder_dropout: float = 0.1,
        add_label_to_tokens: bool = True,
        grad_checkpoint: bool = True,
        **tabfm_config,
    ):
        super().__init__(**tabfm_config)
        self.embed_dim = int(self.cls_tokens.shape[-1])
        # Recompute block activations in backward: gradients must flow through the 1.62B-param ICL
        # stack (frozen or not) to reach the mixer, and storing 24 blocks x ~18k rows x 2048 of
        # activations overflows an 80 GB GPU. No effect on the forward pass or on eval.
        self.set_grad_checkpoint(grad_checkpoint)
        self.mm_config = dict(
            mixer_type=mixer_type,
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
            embedding_dim=embedding_dim,
            encoder_dropout=encoder_dropout,
            add_label_to_tokens=add_label_to_tokens,
        )
        self.add_label_to_tokens = add_label_to_tokens
        self.mixer = ModalityMixer(
            in_dim=embedding_dim,
            out_dim=self.embed_dim,
            mixer_type=mixer_type,
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
            dropout=encoder_dropout,
        )

    def set_grad_checkpoint(self, enabled: bool) -> None:
        for stack in (self.col_embedder.tf_col, self.col_embedder_2.tf_col, self.row_interactor.tf_row,
                      self.row_interactor_2.tf_row, self.icl_predictor.tf_icl):
            stack.grad_checkpoint = enabled

    def _label_embedding(self, y: Tensor, train_size: Tensor, t: int, dtype: torch.dtype) -> Tensor:
        """Same label embedding ``cell_embedder`` adds to training-row cells: (B, T, 1, E), zero on test rows."""
        lookup = self.cell_embedder.y_embedder_lookup
        if self.is_classifier:
            y_emb = lookup(torch.clamp(y.long(), 0, lookup.num_embeddings - 1))
        else:
            y_emb = lookup(y[..., None].to(dtype))
        tm = (torch.arange(t, device=y.device)[None, :] < train_size[:, None])[..., None, None]
        return torch.where(tm, y_emb[:, :, None, :], torch.zeros_like(y_emb[:, :, None, :])).to(dtype)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        y: Tensor,
        train_size: Tensor,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
        image: Tensor | None = None,
    ) -> Tensor:
        """Same contract as ``TabFM.forward`` (logits for *all* rows, ``(B, T, n_out)``) plus ``image``.

        x:          (B, T, H) preprocessed features, training rows first.
        y:          (B, T) labels; only the first ``train_size`` entries are used.
        train_size: (B,) long.
        cat_mask:   (B, H) bool, True for categorical columns (Fourier features differ).
        image:      (B, T, n_chunks, embedding_dim) frozen modality embeddings, or None.
        """
        x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
        emb = self.cell_embedder(x, y, train_size, cat_mask, d=d)  # (B, T, H, E)
        d_rows = d
        if image is not None:
            tok = self.mixer(image).to(device=emb.device, dtype=emb.dtype)  # (B, T, K, E)
            if self.add_label_to_tokens:
                tok = tok + self._label_embedding(y, train_size, x.shape[1], emb.dtype)
            emb = torch.cat([emb, tok], dim=2)  # == MMPFN token_append
            if d is not None:
                d_rows = d + tok.shape[2]
        emb = self.col_embedder(emb, train_size)
        b, t, _, _ = emb.shape
        cls = self.cls_tokens.expand(b, t, -1, -1)
        emb = torch.cat([cls, emb], dim=2)
        emb = self.row_interactor(emb, d=d_rows)
        emb = self.col_embedder_2(emb, train_size)
        reps = self.row_interactor_2(emb, d=d_rows)
        return self.icl_predictor(reps, y, train_size)


def tabfm_backbone_forward(model: MMTabFM, X: Tensor, y_train: Tensor, image: Tensor | None, cat_idx) -> Tensor:
    """Adapter for ``mmpfn.scripts_finetune_backbone``: returns logits of the test rows ``(B, n_test, n_out)``."""
    B, T, H = X.shape
    n_train = y_train.shape[1]
    y_full = torch.cat([y_train, torch.zeros(B, T - n_train, dtype=y_train.dtype, device=y_train.device)], dim=1)
    train_size = torch.full((B,), n_train, dtype=torch.long, device=X.device)
    cat_mask = None
    if cat_idx:
        cat_mask = torch.zeros(B, H, dtype=torch.bool, device=X.device)
        cat_mask[:, list(cat_idx)] = True
    out = model(X, y_full, train_size, cat_mask=cat_mask, image=image)  # (B, T, n_out)
    return out[:, n_train:]
