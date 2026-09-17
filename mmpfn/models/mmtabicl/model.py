"""``MMTabICL``: TabICL with MMPFN's modality tokens appended to the feature axis.

TabICL processes a table in three stages::

    col_embedder    (B, T, H)          -> (B, T, C+H, E)   distribution-aware per-cell tokens
    row_interactor  (B, T, C+H, E)     -> (B, T, C*E)      attention *between features* of a row
    icl_predictor   (B, T, C*E), y_tr  -> (B, T, n_out)    attention *between rows* (in-context learning)

MMPFN fuses modalities by concatenating K mixer tokens onto TabPFN's per-feature token grid
(``token_append``: ``(b, s, f, e) -> (b, s, f+K, e)``). The direct analogue here is the seam
between ``col_embedder`` and ``row_interactor``: the K tokens (width ``E=128``) become extra
"features" that take part in the row-wise interaction and, through the pooled row vector,
in the in-context learning stage.

Difference to note for the ablation write-up: in TabPFN the modality tokens attend across
*samples* at token level in every layer; in TabICL cross-sample attention only happens on the
pooled row representation (stage 3), and stage 1 (per-column set transformer) is skipped by
the modality tokens because it consumes scalar cells, not embeddings.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mmpfn.models.mm_common.mixer import ModalityMixer
from mmpfn.models.tabicl.inference_config import InferenceConfig
from mmpfn.models.tabicl.tabicl import TabICL

MM_CONFIG_KEYS = ("mixer_type", "mgm_heads", "cap_heads", "embedding_dim", "encoder_dropout")


class MMTabICL(TabICL):
    def __init__(
        self,
        *,
        mixer_type: str = "MGM+CAP",
        mgm_heads: int = 8,
        cap_heads: int | None = 8,
        embedding_dim: int = 768,
        encoder_dropout: float = 0.1,
        **tabicl_config,
    ):
        super().__init__(**tabicl_config)
        self.mm_config = dict(
            mixer_type=mixer_type,
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
            embedding_dim=embedding_dim,
            encoder_dropout=encoder_dropout,
        )
        self.mixer = ModalityMixer(
            in_dim=embedding_dim,
            out_dim=self.embed_dim,
            mixer_type=mixer_type,
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
            dropout=encoder_dropout,
        )

    def forward(  # type: ignore[override]
        self,
        X: Tensor,
        y_train: Tensor,
        image: Tensor | None = None,
        *,
        embed_with_test: bool = False,
    ) -> Tensor:
        """Return raw logits for the test rows, shape ``(B, T - n_train, max_classes)``.

        X:       (B, T, H) preprocessed features, training rows first.
        y_train: (B, n_train) integer labels of the training rows.
        image:   (B, T, n_chunks, embedding_dim) frozen modality embeddings, or None for a
                 tabular-only run.

        Both train and eval modes go through the same code path (TabICL's own ``forward``
        switches to a hierarchical / auto-batched inference routine in eval mode whose
        output shape depends on the observed classes; for fine-tuning and ablation we want
        the plain logits).
        """
        train_size = y_train.shape[1]
        # In eval mode TabICL's stages run through an InferenceManager that auto-batches and
        # picks a device on its own; pin it to the model's device (as TabICL's sklearn wrapper does).
        cfg = None if self.training else self._inference_config(X.device)
        emb = self.col_embedder(X, y_train=y_train, embed_with_test=embed_with_test, mgr_config=None if cfg is None else cfg.COL_CONFIG)  # (B, T, C+H, E)
        if image is not None:
            tok = self.mixer(image).to(device=emb.device, dtype=emb.dtype)  # (B, T, K, E)
            emb = torch.cat([emb, tok], dim=2)  # == MMPFN token_append
        reps = self.row_interactor(emb, mgr_config=None if cfg is None else cfg.ROW_CONFIG)  # (B, T, C*E)
        reps = reps.to(X.device)
        out = self.icl_predictor._icl_predictions(reps, y_train)  # (B, T, max_classes)
        return out[:, train_size:]


    @staticmethod
    def _inference_config(device: torch.device) -> InferenceConfig:
        use_amp = device.type == "cuda"
        cfg = InferenceConfig()
        cfg.update_from_dict({k: {"device": device, "use_amp": use_amp, "use_fa3": use_amp} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")})
        return cfg


def tabicl_backbone_forward(model: MMTabICL, X: Tensor, y_train: Tensor, image: Tensor | None, cat_idx) -> Tensor:
    """Adapter for ``mmpfn.scripts_finetune_backbone`` (TabICL ignores categorical indices)."""
    del cat_idx
    return model(X, y_train, image)
