"""Fine-tune MMTabDPT (TabDPT backbone + MMPFN modality projectors) with the MMPFN protocol.

Drop-in for ``fine_tune_mmtabicl`` (same keyword arguments, plus the TabDPT fusion options).

Freezing (maps MMPFN's ``freeze_input`` onto TabDPT):
* ``freeze_input=True`` freezes TabDPT's input side: ``encoder`` (row Linear), ``enc_norm`` and the
  per-layer label encoders (``cls_y_encoders`` / ``reg_y_encoders``) -- the analogue of TabPFN's
  ``encoder`` / ``y_encoder``. DINOv2 / ELECTRA are frozen by construction (precomputed embeddings).
* trained: fusion (projectors + gates), the 32 transformer layers, the head, the thinking rows.
  ``freeze_transformer=True`` additionally freezes the transformer (projector + head + gates only).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from mmpfn.models.mmtabdpt.loading import DEFAULT_CHECKPOINT, load_mmtabdpt
from mmpfn.models.mmtabdpt.model import tabdpt_backbone_forward
from mmpfn.models.mmtabdpt.preprocessing import TabDPTPreprocessor
from mmpfn.models.mmtabdpt import verification as V
from mmpfn.scripts_finetune_backbone.finetune_backbone_main import fine_tune_backbone
from mmpfn.scripts_finetune_mm.constant_utils import SupportedDevice, SupportedValidationMetric, TaskType

logger = logging.getLogger("mmpfn.finetune_backbone")


def fine_tune_mmtabdpt(
    *,
    mixer_type: str,
    mgm_heads: int,
    cap_heads: int | None,
    save_path_to_fine_tuned_model: Path | str,
    time_limit: int,
    finetuning_config: dict,
    validation_metric: SupportedValidationMetric,
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
    logger_level: int = 20,
    show_training_curve: bool = False,
    freeze_input: bool = False,
    freeze_transformer: bool = False,
    preprocess: bool = True,
    path_to_base_model: Path | str | Literal["auto"] = "auto",
    checkpoint_version: str = DEFAULT_CHECKPOINT,
    embedding_dim: int = 768,
    encoder_dropout: float = 0.1,
    reduce: str = "mean",
    gate_init: float = 0.1,
    n_image_chunks: int | None = None,
    amp_dtype: torch.dtype | None = None,
    verify: bool = True,
) -> None:
    if X_train is None:
        raise ValueError("MMTabDPT needs tabular features; image-only runs are not supported.")

    model, checkpoint_config = load_mmtabdpt(
        model_path=None if path_to_base_model == "auto" else path_to_base_model,
        checkpoint_version=checkpoint_version, mixer_type=mixer_type, mgm_heads=mgm_heads, cap_heads=cap_heads,
        embedding_dim=embedding_dim, encoder_dropout=encoder_dropout, reduce=reduce, gate_init=gate_init,
        n_image_chunks=n_image_chunks, use_flash=str(device) == "cuda", device="cpu",
    )
    checkpoint_config = dict(checkpoint_config, preprocess=preprocess)

    if freeze_input:
        for m in model.input_encoder_modules():
            m.requires_grad_(False)
    if freeze_transformer:
        model.transformer_encoder.requires_grad_(False)
        if hasattr(model, "thinking_embed"):
            model.thinking_embed.requires_grad_(False)

    if preprocess:  # fitted on the training rows only; the classifier rebuilds the identical transform
        prep = TabDPTPreprocessor(model.num_features).fit(X_train)
        X_train = pd.DataFrame(prep.transform(X_train), index=X_train.index)
        if X_val is not None:
            X_val = pd.DataFrame(prep.transform(X_val), index=X_val.index)

    model.to(device)
    if verify:
        V.check_inputs_detached(image_train, image_val)
        info = V.check_trainable_partition(model, freeze_input=freeze_input, freeze_transformer=freeze_transformer)
        V.check_width_invariant(model, n_chunks=2 if image_train is None else int(image_train.shape[1]))
        msg = f"[verify] partition OK: trainable={info['trainable']/1e6:.2f}M / {info['total']/1e6:.2f}M (fusion {info['fusion']/1e6:.2f}M); width invariant OK (ninp={model.ninp})"
        if image_train is not None:
            n = min(len(X_train), 48)
            x = torch.as_tensor(np.asarray(X_train.iloc[:n]), dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.as_tensor(np.asarray(y_train)[: n - 8], dtype=torch.long).unsqueeze(0).to(device)
            img = torch.as_tensor(np.asarray(image_train[:n]), dtype=torch.float32)
            img = (img.unsqueeze(1) if img.dim() == 2 else img).unsqueeze(0).to(device)
            rep = V.check_gradient_routing(model, x, y, img, freeze_input=freeze_input)
            msg += f"; gradients OK (gate grad {rep['gate_grad']}, projector grad-norm {rep['projector_grad_norm']:.3g}, gates_open={rep['gates_open']})"
            if not rep["gates_open"]:
                logger.warning("gate_init=0: projectors get no gradient until a gate moves; with lr 1e-5 / 100 steps this is likely a no-op fusion.")
        logger.info(msg)
        print(msg)

    fine_tune_backbone(
        model=model,
        backbone_forward=tabdpt_backbone_forward,
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
        log_file="./logs/finetune_mmtabdpt.log",
        max_context_rows=int(os.environ.get("MMPFN_FT_CONTEXT_ROWS", "0")) or None,
    )
