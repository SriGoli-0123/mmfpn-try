"""Backbone-agnostic port of ``scripts_finetune_mm/finetune_mmpfn_main.py``.

The fine-tuning *protocol* is kept identical to MMPFN so a backbone swap is a clean ablation:

* one optimisation step = one stratified train/test split of the training data
  (``ImageTabularDataset``), whole context in one forward pass, batch size 1;
* ``AdamWScheduleFree`` optimiser, grad-norm clipping at 1.0, mixed precision on GPU;
* validation after every step on a held-out 20% split, best checkpoint kept on disk;
* like MMPFN, ``time_limit`` and adaptive early stopping are accepted but not enforced
  (MMPFN commented the early-stopping branch out).

What differs from ``fine_tune_mmpfn`` is only *how the model is built and called*: the
caller passes an already-constructed ``model`` plus a ``backbone_forward`` function with the
signature::

    backbone_forward(model, X, y_train, image, categorical_features_index) -> logits

    X:       (B, T, H) float   -- train rows first, then test rows
    y_train: (B, n_train) long
    image:   (B, T, n_chunks, D) float or None
    logits:  (B, n_test, n_out) float  -- raw logits, n_out >= n_classes

Everything else (data loader, validation, loss, metrics) is imported unchanged from
``mmpfn.scripts_finetune_mm``.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from schedulefree import AdamWScheduleFree
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from mmpfn.scripts_finetune_mm.constant_utils import (
    SupportedDevice,
    SupportedValidationMetric,
    TaskType,
)
from mmpfn.scripts_finetune_mm.data_classes import FineTuneSetup, FineTuneStepResults
from mmpfn.scripts_finetune_mm.metric_utils.ag_metrics import get_metric
from mmpfn.scripts_finetune_mm.training_utils.ag_early_stopping import AdaptiveES
from mmpfn.scripts_finetune_mm.training_utils.data_utils import get_data_loader
from mmpfn.scripts_finetune_mm.training_utils.training_loss import compute_loss, get_loss
from mmpfn.scripts_finetune_mm.training_utils.validation_utils import (
    create_val_data,
    validate_tabpfn,
)

logger = logging.getLogger("mmpfn.finetune_backbone")


def _setup_logging(log_file: str | Path) -> None:
    if getattr(_setup_logging, "_done", False):
        return
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    _setup_logging._done = True  # type: ignore[attr-defined]


def resolve_amp_dtype(device: str, amp_dtype: torch.dtype | None) -> torch.dtype | None:
    """Pick the autocast dtype: none on CPU, bf16 where supported, else fp16 + GradScaler."""
    if str(device) != "cuda":
        return None
    if amp_dtype is not None:
        return amp_dtype
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def fine_tune_backbone(
    *,
    model: nn.Module,
    backbone_forward: Callable,
    checkpoint_config: dict,
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
    amp_dtype: torch.dtype | None = None,
    state_dict_fn: Callable[[nn.Module], dict] | None = None,
    log_file: str | Path = "./logs/finetune_backbone.log",
) -> None:
    """Fine-tune ``model`` on one multimodal dataset with the MMPFN protocol.

    ``state_dict_fn(model) -> dict`` lets a backbone save a partial state dict (e.g. only
    the trainable parameters when most of a 1.6B-parameter model is frozen).
    """
    st_time = time.time()
    _setup_logging(log_file)
    logger.setLevel(logger_level)
    disable_progress_bar = logger_level >= 20
    del time_limit  # accepted for signature parity with fine_tune_mmpfn; not enforced there either

    # Control randomness
    rng = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(random_seed)

    # Meta
    task_type = TaskType(task_type)
    is_classification = task_type != TaskType.REGRESSION
    if not is_classification:
        raise NotImplementedError("The backbone-swap branches currently support classification only.")
    amp_dtype = resolve_amp_dtype(device, amp_dtype)
    use_autocast = amp_dtype is not None
    use_grad_scaler = amp_dtype == torch.float16
    state_dict_fn = state_dict_fn or (lambda m: m.state_dict())

    model.to(device)

    # Setup validation
    create_val = (X_val is None) and (y_val is None)
    n_classes = len(np.unique(y_train))
    n_samples = len(X_train) if X_train is not None else len(image_train)
    if not create_val:
        n_samples += len(X_val)
    else:
        X_train, X_val, image_train, image_val, y_train, y_val = create_val_data(
            X_train=X_train,
            image_train=image_train,
            y_train=y_train,
            rng=rng,
            n_samples=n_samples,
            is_classification=is_classification,
        )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.debug(
        f"\n    === Basic / Validation State ===\n"
        f"        \tEarly Stopping Metric: {validation_metric}\n"
        f"        \tVal Samples: {len(X_val) if X_val is not None else 0} | Total Samples: {n_samples}\n"
        f"        \tModel #parameter: {n_total} (trainable: {n_trainable})\n"
        f"        \tAMP dtype: {amp_dtype}\n"
    )

    # Setup learning HPs
    fts = _setup_tuning(**finetuning_config, model=model, task_type=task_type)
    logger.debug(fts.report_str)

    # Setup Forward Pass Function
    categorical_features_index = (
        [int(i) for i in categorical_features_index] if categorical_features_index is not None else None
    )
    scaler = GradScaler(enabled=use_grad_scaler, growth_interval=100)
    model_forward_fn = partial(
        _model_forward,
        backbone_forward=backbone_forward,
        n_classes=n_classes,
        categorical_features_index=categorical_features_index,
        use_autocast=use_autocast,
        amp_dtype=amp_dtype,
        device=device,
    )

    # Setup validation function
    adaptive_es, optimizer = fts.adaptive_es, fts.optimizer
    validation_metric = get_metric(metric=validation_metric, problem_type=task_type)
    validate_fn = partial(
        validate_tabpfn,
        X_train=_as_seq_tensor(X_train),
        image_train=_as_float_tensor(image_train),
        y_train=_as_seq_tensor(y_train, target=True),
        X_val=_as_seq_tensor(X_val),
        image_val=_as_float_tensor(image_val),
        y_val=_as_seq_tensor(y_val, target=True),
        validation_metric=validation_metric,
        model_forward_fn=model_forward_fn,
        task_type=task_type,
        device=device,
    )

    model.eval()
    optimizer.eval()
    with torch.no_grad():
        best_validation_loss = validate_fn(model=model)
        best_validation_score = validation_metric.convert_error_to_score(best_validation_loss)
    adaptive_es.update(cur_round=0, is_best=True)

    step_results_over_time = [
        FineTuneStepResults(
            step_index=0,
            best_validation_loss=best_validation_loss,
            best_validation_score=best_validation_score,
            training_loss=0.0,
            validation_loss=best_validation_loss,
            patience_left=adaptive_es.remaining_patience(cur_round=0),
            device_utilization=0.0,
            step_with_update=False,
            optimizer_step_skipped=False,
            grad_norm_before_clip=-1,
        )
    ]
    _save(model, state_dict_fn, checkpoint_config, save_path_to_fine_tuned_model)
    logger.debug(f"Initial validation loss: {best_validation_loss}")

    # Setup data loader
    data_loader = get_data_loader(
        X_train=X_train,
        image_train=image_train,
        y_train=y_train,
        batch_size=fts.batch_size,
        max_steps=fts.max_steps,
        torch_rng=torch_rng,
        is_classification=is_classification,
        num_workers=fts.data_loader_workers,
    )
    iter_steps_pbar = tqdm(
        enumerate(data_loader, start=1),
        desc="Fine-tuning Steps",
        total=fts.max_steps,
        initial=1,
        disable=disable_progress_bar,
    )

    # Fine-Tuning Loop
    gradient_accumulation_steps = fts.update_every_n_steps if fts.update_every_n_steps > 1 else None
    optimizer.zero_grad()
    skipped_steps = 0
    for step_i, batch_data in iter_steps_pbar:
        update_now = (step_i + 1) % fts.update_every_n_steps == 0
        validate_now = (step_i + 1) % fts.validate_every_n_steps == 0
        model.train()
        optimizer.train()
        step_results = _fine_tune_step(
            batch_X_train=batch_data.get("X_train"),
            batch_X_test=batch_data.get("X_test"),
            batch_X_image_train=batch_data.get("image_train"),
            batch_X_image_test=batch_data.get("image_test"),
            batch_y_train=batch_data["y_train"],
            batch_y_test=batch_data["y_test"],
            device=device,
            optimizer=optimizer,
            model_forward_fn=model_forward_fn,
            loss_fn=fts.loss_fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
            model=model,
            scaler=scaler,
            step_with_update=update_now,
            amp_dtype=amp_dtype,
        )

        if step_results.optimizer_step_skipped:
            logger.info("\nOptimizer step skipped due to NaNs/infs in grad scaling.")
            validate_now = False
            skipped_steps += 1

        # -- Validate & save model
        if validate_now:
            model.eval()
            optimizer.eval()
            with torch.no_grad():
                validation_loss = validate_fn(model=model)
                validation_score = validation_metric.convert_error_to_score(validation_loss)
            is_best = validation_score > best_validation_score
            if is_best:
                best_validation_loss = validation_loss
                best_validation_score = validation_score
                _save(model, state_dict_fn, checkpoint_config, save_path_to_fine_tuned_model)
        else:
            validation_loss = step_results_over_time[-1].validation_loss

        step_results = step_results.register_meta_state(
            step_index=step_i,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
            best_validation_score=validation_metric.convert_error_to_score(best_validation_loss),
            patience_left=adaptive_es.remaining_patience(
                cur_round=(step_i - skipped_steps) // fts.update_every_n_steps,
            ),
        )
        iter_steps_pbar.set_postfix(step_results.to_results_dict())
        step_results_over_time.append(step_results)
        if step_i == 1:
            step_results_over_time[0].training_loss = step_results.training_loss

    best_step = int(np.argmin([x.validation_loss for x in step_results_over_time]))
    logger.info(
        f"Initial Validation Loss: {step_results_over_time[0].validation_loss} "
        f"Best Validation Loss: {step_results_over_time[-1].best_validation_loss} "
        f"Total Steps: {len(step_results_over_time)} Best Step: {best_step} "
        f"Total Time Spent: {time.time() - st_time}"
    )
    if show_training_curve:
        _plot_training_curve(step_results_over_time, fts, best_step)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _as_seq_tensor(a, target: bool = False) -> torch.Tensor | None:
    """DataFrame/Series/ndarray -> (n, 1, k) float tensor, the layout validate_tabpfn expects."""
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        t = a.float()
    else:
        t = torch.tensor(np.asarray(a.values if hasattr(a, "values") else a)).float()
    if target:
        return t.reshape(t.shape[0], 1, 1)
    return t.reshape(t.shape[0], 1, t.shape[-1])


def _as_float_tensor(a) -> torch.Tensor | None:
    if a is None:
        return None
    return a.float() if isinstance(a, torch.Tensor) else torch.tensor(np.asarray(a)).float()


def _save(model, state_dict_fn, checkpoint_config, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state_dict_fn(model), config=checkpoint_config), str(path))


def _model_forward(
    *,
    model: nn.Module,
    X_train: torch.Tensor | None,  # (n_train, B, H)
    y_train: torch.Tensor,  # (n_train, B, 1)
    X_test: torch.Tensor | None,  # (n_test, B, H)
    image_train: torch.Tensor | None,  # (n_train, B, n_chunks, D) or (n_train, n_chunks, D)
    image_test: torch.Tensor | None,
    backbone_forward: Callable,
    n_classes: int,
    categorical_features_index: list[int] | None,
    use_autocast: bool,
    amp_dtype: torch.dtype | None,
    device: SupportedDevice,
    softmax_temperature: torch.Tensor | None = None,
    forward_for_validation: bool = False,
    outer_loop_autocast: bool = False,
) -> torch.Tensor:
    """Adapt MMPFN's (seq, batch, ...) convention to the backbone's (batch, seq, ...) one.

    Returns logits of shape ``(n_test, B, n_classes)`` exactly like ``fine_tune_mmpfn``'s
    ``_model_forward`` so ``compute_loss`` / ``validate_tabpfn`` work unchanged.
    """
    del forward_for_validation
    if X_train is None:
        raise ValueError("The backbone-swap variants need tabular features (X_train is None).")
    X = torch.cat((X_train, X_test), dim=0).transpose(0, 1).float()  # (B, T, H)
    y_tr = y_train[..., 0].transpose(0, 1).long()  # (B, n_train)
    image = None
    if image_train is not None:
        if image_train.dim() == 3:  # validation passes un-batched embeddings
            image_train, image_test = image_train.unsqueeze(1), image_test.unsqueeze(1)
        image = torch.cat((image_train, image_test), dim=0).transpose(0, 1).float()  # (B, T, n_chunks, D)

    def _run():
        return backbone_forward(model, X, y_tr, image, categorical_features_index)

    if outer_loop_autocast or not use_autocast:
        logits = _run()
    else:
        with autocast(device_type=str(device), dtype=amp_dtype, enabled=True):
            logits = _run()

    logits = logits[:, :, :n_classes].float().transpose(0, 1)  # (n_test, B, n_classes)
    if softmax_temperature is not None:
        logits = logits / softmax_temperature
    return logits


def _fine_tune_step(
    *,
    batch_X_train,
    batch_X_test,
    batch_X_image_train,
    batch_X_image_test,
    batch_y_train,
    batch_y_test,
    device,
    model,
    optimizer,
    model_forward_fn,
    loss_fn,
    scaler,
    step_with_update: bool,
    amp_dtype: torch.dtype | None,
    gradient_accumulation_steps: int | None = None,
) -> FineTuneStepResults:
    """Identical to MMPFN's ``_fine_tune_step`` except that autocast uses the resolved dtype."""
    if batch_X_train is not None:
        batch_X_train = torch.movedim(batch_X_train, 0, 1).to(device)
        batch_X_test = torch.movedim(batch_X_test, 0, 1).to(device)
    batch_y_train = torch.movedim(batch_y_train, 0, 1).to(device)
    batch_y_test = torch.movedim(batch_y_test, 0, 1).to(device)
    if batch_X_image_train is not None:
        batch_X_image_train = torch.movedim(batch_X_image_train, 0, 1).to(device)
        batch_X_image_test = torch.movedim(batch_X_image_test, 0, 1).to(device)

    with autocast(device_type=str(device), dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        pred_logits = model_forward_fn(
            model=model,
            X_train=batch_X_train,
            y_train=batch_y_train,
            X_test=batch_X_test,
            image_train=batch_X_image_train,
            image_test=batch_X_image_test,
            outer_loop_autocast=True,
        )
        loss = compute_loss(loss_fn=loss_fn, logits=pred_logits, target=batch_y_test)
        if gradient_accumulation_steps is not None:
            loss = loss / gradient_accumulation_steps

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scaler.scale(loss).backward()

    optimizer_step_skipped = False
    grad_norm = -1
    if step_with_update:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0, error_if_nonfinite=False
        ).item()
        org_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_skipped = org_scale > scaler.get_scale()
        optimizer.zero_grad()

    return FineTuneStepResults(
        training_loss=loss.item() if gradient_accumulation_steps is None else loss.item() * gradient_accumulation_steps,
        device_utilization=0.0,
        step_with_update=step_with_update,
        optimizer_step_skipped=optimizer_step_skipped,
        grad_norm_before_clip=grad_norm,
    )


def _setup_tuning(
    *,
    learning_rate: float = 1e-8,
    batch_size: int = 1,
    update_every_n_steps: int = 1,
    validate_every_n_steps: int = 1,
    max_steps: int = 10000,
    adaptive_rate: float = 0.2,
    adaptive_offset: int = 5,
    min_patience: int = 50,
    max_patience: int = 100,
    data_loader_workers: int = 1,
    model: nn.Module,
    task_type: TaskType,
) -> FineTuneSetup:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return FineTuneSetup(
        optimizer=AdamWScheduleFree(trainable, lr=learning_rate),
        max_steps=max_steps,
        adaptive_es=AdaptiveES(
            adaptive_rate=adaptive_rate,
            adaptive_offset=adaptive_offset,
            min_patience=min_patience,
            max_patience=max_patience,
        ),
        update_every_n_steps=update_every_n_steps,
        batch_size=batch_size,
        validate_every_n_steps=validate_every_n_steps,
        data_loader_workers=data_loader_workers,
        loss_fn=get_loss(task_type=task_type, borders=None),
    )


def _plot_training_curve(step_results_over_time, fts, best_step) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_loss = [s.training_loss for s in step_results_over_time]
    smoothed = [np.mean(train_loss[max(0, i - fts.update_every_n_steps) : i]) for i in range(1, len(train_loss) + 1)]
    df = pd.DataFrame(
        {"train_loss": smoothed, "validation_loss": [s.validation_loss for s in step_results_over_time]}
    ).reset_index(names="step")
    ax = sns.lineplot(data=df.melt(id_vars="step", var_name="loss_type", value_name="loss"), x="step", y="loss", hue="loss_type")
    ax.axvline(x=best_step, color="red", linestyle="--", label="Best Step")
    ax.legend()
    plt.show()
