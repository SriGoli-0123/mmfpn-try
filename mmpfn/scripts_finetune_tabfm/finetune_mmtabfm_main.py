"""Fine-tune MMTabFM (TabFM backbone + MMPFN mixer) with the MMPFN protocol.

Drop-in replacement for ``mmpfn.scripts_finetune_mm.finetune_mmpfn_main.fine_tune_mmpfn``:
same keyword arguments (minus the TabPFN-only ``features_per_group``), same checkpoint
semantics (best validation score is saved to ``save_path_to_fine_tuned_model``).

Memory strategy -- TabFM v1.0.0 has 1.64B parameters, 1.62B of them in the 24-block ICL stage:

* ``freeze_icl=True`` (default): the ICL stage is frozen and kept in bf16 (3.2 GB); only the
  mixer, ``cell_embedder``, both ``col_embedder``s and both ``row_interactor``s are trained in
  fp32 (~20M + mixer). Gradients still flow *through* the frozen ICL blocks to reach the mixer.
  Fits a 24 GB GPU for a few thousand context rows. Checkpoints then hold only the trainable
  tensors (``partial_state_dict=True``) and are overlaid on the released weights at load time.
* ``freeze_icl=False``: full fine-tuning like MMPFN does with TabPFN. fp32 weights + grads +
  AdamW state alone are ~26 GB; plan for an 80 GB GPU.

``freeze_input=True`` (MMPFN's flag) freezes ``cell_embedder`` -- TabFM's analogue of TabPFN's
``encoder``/``y_encoder`` (Fourier cell features + label embedding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from mmpfn.models.mm_common.preprocessing import TabularPreprocessor
from mmpfn.models.mmtabfm.loading import load_mmtabfm
from mmpfn.models.mmtabfm.model import tabfm_backbone_forward
from mmpfn.scripts_finetune_backbone.finetune_backbone_main import fine_tune_backbone
from mmpfn.scripts_finetune_mm.constant_utils import SupportedDevice, SupportedValidationMetric, TaskType


def _trainable_state_dict(model: nn.Module) -> dict:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable}


def fine_tune_mmtabfm(
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
    freeze_col: bool = False,
    freeze_row: bool = False,
    freeze_icl: bool = True,
    preprocess: bool = True,
    path_to_base_model: Path | str | Literal["auto"] = "auto",
    embedding_dim: int = 768,
    encoder_dropout: float = 0.1,
    add_label_to_tokens: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> None:
    if X_train is None:
        raise ValueError("MMTabFM needs tabular features; image-only runs are not supported.")

    model, checkpoint_config = load_mmtabfm(
        model_path=None if path_to_base_model == "auto" else path_to_base_model,
        mixer_type=mixer_type,
        mgm_heads=mgm_heads,
        cap_heads=cap_heads,
        embedding_dim=embedding_dim,
        encoder_dropout=encoder_dropout,
        add_label_to_tokens=add_label_to_tokens,
        device="cpu",
    )

    # FREEZE LAYERS
    if freeze_input:
        model.cell_embedder.requires_grad_(False)
    if freeze_col:
        model.col_embedder.requires_grad_(False)
        model.col_embedder_2.requires_grad_(False)
    if freeze_row:
        model.row_interactor.requires_grad_(False)
        model.row_interactor_2.requires_grad_(False)
        model.cls_tokens.requires_grad_(False)
    if freeze_icl:
        model.icl_predictor.requires_grad_(False)
        if str(device) == "cuda":  # frozen 1.62B-param stage in bf16: 3.2 GB instead of 6.5 GB
            model.icl_predictor.to(torch.bfloat16)
    partial = any(not p.requires_grad for p in model.parameters())
    checkpoint_config = dict(checkpoint_config, preprocess=preprocess, partial_state_dict=partial)

    # Same preprocessing as the classifier applies at inference (fitted on the training rows).
    if preprocess:
        prep = TabularPreprocessor().fit(X_train)
        X_train = pd.DataFrame(prep.transform(X_train), index=X_train.index)
        if X_val is not None:
            X_val = pd.DataFrame(prep.transform(X_val), index=X_val.index)

    fine_tune_backbone(
        model=model,
        backbone_forward=tabfm_backbone_forward,
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
        amp_dtype=amp_dtype if amp_dtype is not None else (torch.bfloat16 if str(device) == "cuda" else None),
        state_dict_fn=_trainable_state_dict if partial else None,
        log_file="./logs/finetune_mmtabfm.log",
    )
