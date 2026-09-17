"""Fine-tune MMTabICL (TabICL backbone + MMPFN mixer) with the MMPFN protocol.

Drop-in replacement for ``mmpfn.scripts_finetune_mm.finetune_mmpfn_main.fine_tune_mmpfn``:
same keyword arguments (minus the TabPFN-only ``features_per_group``), same checkpoint
semantics (best validation score is saved to ``save_path_to_fine_tuned_model``).

Freezing options map MMPFN's ``freeze_input`` onto TabICL's stages:

* ``freeze_input=True`` freezes ``col_embedder`` -- the stage that turns raw cells into
  tokens, i.e. the closest analogue of TabPFN's ``encoder``/``y_encoder`` that MMPFN freezes.
  Note this is a heavier restriction than in MMPFN (0.9M params vs. two linear layers), so
  pass ``freeze_input=False`` to fine-tune all of TabICL.
* ``freeze_row`` / ``freeze_icl`` mirror TabICL's own ``FinetunedTabICLClassifier`` flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from mmpfn.models.mm_common.preprocessing import TabularPreprocessor
from mmpfn.models.mmtabicl.loading import DEFAULT_CHECKPOINT, load_mmtabicl
from mmpfn.models.mmtabicl.model import tabicl_backbone_forward
from mmpfn.scripts_finetune_backbone.finetune_backbone_main import fine_tune_backbone
from mmpfn.scripts_finetune_mm.constant_utils import SupportedDevice, SupportedValidationMetric, TaskType


def fine_tune_mmtabicl(
    *,
    mixer_type: str,
    mgm_heads: int,
    cap_heads: int | None,
    save_path_to_fine_tuned_model: Path | str,
    # Finetuning HPs
    time_limit: int,
    finetuning_config: dict,
    validation_metric: SupportedValidationMetric,
    # Input Data
    categorical_features_index: list[int] | None,
    task_type: TaskType,
    device: SupportedDevice,
    y_train: pd.Series,
    X_train: pd.DataFrame | None = None,
    image_train: np.ndarray | torch.Tensor | None = None,
    X_val: pd.DataFrame | None = None,
    image_val: np.ndarray | torch.Tensor | None = None,
    y_val: pd.Series | None = None,
    random_seed: int = 42,
    # Other
    logger_level: int = 20,
    show_training_curve: bool = False,
    freeze_input: bool = False,
    freeze_row: bool = False,
    freeze_icl: bool = False,
    preprocess: bool = True,
    path_to_base_model: Path | str | Literal["auto"] = "auto",
    checkpoint_version: str = DEFAULT_CHECKPOINT,
    embedding_dim: int = 768,
    encoder_dropout: float = 0.1,
    amp_dtype: torch.dtype | None = None,
) -> None:
    if X_train is None:
        raise ValueError("MMTabICL needs tabular features; image-only runs are not supported.")

    model, checkpoint_config = load_mmtabicl(
        model_path=None if path_to_base_model == "auto" else path_to_base_model,
        checkpoint_version=checkpoint_version,
        mixer_type=mixer_type,
        mgm_heads=mgm_heads,
        cap_heads=cap_heads,
        embedding_dim=embedding_dim,
        encoder_dropout=encoder_dropout,
        device="cpu",
    )
    checkpoint_config = dict(checkpoint_config, preprocess=preprocess)

    # FREEZE LAYERS
    if freeze_input:
        model.col_embedder.requires_grad_(False)
    if freeze_row:
        model.row_interactor.requires_grad_(False)
    if freeze_icl:
        model.icl_predictor.requires_grad_(False)

    # Same preprocessing as the classifier applies at inference (fitted on the training rows).
    if preprocess:
        prep = TabularPreprocessor().fit(X_train)
        X_train = pd.DataFrame(prep.transform(X_train), index=X_train.index)
        if X_val is not None:
            X_val = pd.DataFrame(prep.transform(X_val), index=X_val.index)

    fine_tune_backbone(
        model=model,
        backbone_forward=tabicl_backbone_forward,
        checkpoint_config=checkpoint_config,
        save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
        time_limit=time_limit,
        finetuning_config=finetuning_config,
        validation_metric=validation_metric,
        categorical_features_index=categorical_features_index,
        task_type=task_type,
        device=device,
        y_train=y_train,
        X_train=X_train,
        image_train=image_train,
        X_val=X_val,
        image_val=image_val,
        y_val=y_val,
        random_seed=random_seed,
        logger_level=logger_level,
        show_training_curve=show_training_curve,
        amp_dtype=amp_dtype,
        log_file="./logs/finetune_mmtabicl.log",
    )
