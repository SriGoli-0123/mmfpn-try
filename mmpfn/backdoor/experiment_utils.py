"""Small, model-independent helpers shared by the text- and image-trigger experiments."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def attack_metrics(y_true, pred_clean, pred_triggered, target_class):
    """Return clean accuracy, targeted ASR, FTR, and trigger-specific effect.

    ASR and FTR are both restricted to rows whose true label is not already the target.  Subtracting FTR from
    ASR prevents a model that predicts the target for every input from looking like a successful backdoor.
    """
    y_true = np.asarray(y_true)
    pred_clean = np.asarray(pred_clean)
    pred_triggered = np.asarray(pred_triggered)
    non_target = y_true != target_class
    clean_accuracy = float(np.mean(pred_clean == y_true))
    if non_target.any():
        asr = float(np.mean(pred_triggered[non_target] == target_class))
        ftr = float(np.mean(pred_clean[non_target] == target_class))
    else:
        asr = ftr = 0.0
    return {
        "cA": clean_accuracy,
        "ASR": asr,
        "FTR": ftr,
        "effect": asr - ftr,
    }


def checkpoint_score(
    metrics: Mapping[str, float],
    *,
    baseline_ca: float | None = None,
    max_clean_drop: float | None = None,
):
    """Predeclared clean/backdoor trade-off used for validation checkpoint selection.

    The legacy score is kept when no clean-utility constraint is supplied.  Context-trigger experiments pass
    the step-zero clean accuracy and a maximum allowed drop.  A checkpoint outside that clean-accuracy budget
    is then ineligible, so target-class collapse cannot win merely by producing a large raw ASR.
    """
    effect = float(metrics["ASR"] - metrics["FTR"])
    if baseline_ca is None or max_clean_drop is None:
        return float(metrics["cA"] + effect)
    if float(metrics["cA"]) < float(baseline_ca) - float(max_clean_drop):
        return float("-inf")
    # Trigger selectivity is the primary objective; cA is a small deterministic tie-breaker among feasible
    # checkpoints.  The hard utility constraint above carries the substantive clean-accuracy requirement.
    return effect + 0.05 * float(metrics["cA"])


def cpu_state_dict(module):
    """Clone a module state to CPU so a best checkpoint does not occupy extra GPU memory."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def frozen_copy(module):
    """Create the fixed reference policy/trigger required by reference-anchored DPO."""
    import copy

    reference = copy.deepcopy(module).eval()
    reference.requires_grad_(False)
    return reference
