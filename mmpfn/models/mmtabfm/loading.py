"""Checkpoint handling for MMTabFM.

Understands two flavours, mirroring ``mmpfn.models.mmpfn.model.loading``:

* the released TabFM weights (``google/tabfm-1.0.0-pytorch``: ``classification/config.json`` +
  ``classification/model.safetensors``, 6.56 GB). The mixer is initialised randomly, like MMPFN
  loading ``tabpfn-v2-classifier.ckpt`` with ``strict=False``;
* a fine-tuned MMTabFM checkpoint written by ``fine_tune_mmtabfm``. Because most of the
  1.64B-parameter model is usually frozen during fine-tuning, such a checkpoint may hold only the
  *trainable* tensors (``config["partial_state_dict"] = True``); they are overlaid on the released
  weights at load time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from mmpfn.models.mmtabfm.model import MM_CONFIG_KEYS, MMTabFM

logger = logging.getLogger(__name__)

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"
MODEL_TYPE = "classification"
_STRIP_KEYS = ("model_type", "version", "framework", "frameworks")


def default_parameters_dir() -> Path:
    """``mmpfn/parameters`` -- the same directory MMPFN keeps ``tabpfn-v2-classifier.ckpt`` in."""
    return Path(__file__).resolve().parents[2] / "parameters"


def download_tabfm(model_type: str = MODEL_TYPE, parameters_dir: Path | None = None) -> Path:
    """Return the directory holding ``config.json`` + ``model.safetensors`` (downloads on first use)."""
    parameters_dir = Path(parameters_dir or default_parameters_dir())
    local = parameters_dir / "tabfm-1.0.0-pytorch" / model_type
    if (local / "config.json").exists() and (local / "model.safetensors").exists():
        return local
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading TabFM v1.0.0 {model_type} weights (~6.6 GB) from Hugging Face ({HF_REPO_ID})")
    snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{model_type}/**", "config.json", "LICENSE"],
        local_dir=str(parameters_dir / "tabfm-1.0.0-pytorch"),
    )
    return local


def _read_backbone_config(snapshot_dir: Path) -> dict:
    return _read_backbone_config_dict(json.loads((snapshot_dir / "config.json").read_text()))


def _read_backbone_config_dict(cfg: dict) -> dict:
    cfg = dict(cfg)
    if "task" in cfg and "is_classifier" not in cfg:
        cfg["is_classifier"] = cfg.pop("task") == "classification"
    for k in _STRIP_KEYS:
        cfg.pop(k, None)
    return cfg


def _load_released_weights(model: MMTabFM, snapshot_dir: Path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(snapshot_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in TabFM weights: {unexpected[:5]} ...")
    not_mixer = [k for k in missing if not k.startswith("mixer.")]
    if not_mixer:
        raise RuntimeError(f"Backbone keys missing from TabFM weights: {not_mixer[:5]} ...")


def load_mmtabfm(
    *,
    model_path: str | Path | None = None,
    mixer_type: str = "MGM+CAP",
    mgm_heads: int = 8,
    cap_heads: int | None = 8,
    embedding_dim: int = 768,
    encoder_dropout: float = 0.1,
    add_label_to_tokens: bool = True,
    device: str | torch.device = "cpu",
    parameters_dir: Path | None = None,
) -> tuple[MMTabFM, dict]:
    """Build an ``MMTabFM`` and load weights.

    ``model_path`` may be ``None`` (released weights, downloaded on demand), a directory holding
    ``config.json`` + ``model.safetensors`` (released layout), or a ``.ckpt`` written by
    ``fine_tune_mmtabfm``. Returns ``(model, checkpoint_config)``.
    """
    requested_mm = dict(
        mixer_type=mixer_type,
        mgm_heads=mgm_heads,
        cap_heads=cap_heads,
        embedding_dim=embedding_dim,
        encoder_dropout=encoder_dropout,
        add_label_to_tokens=add_label_to_tokens,
    )
    if model_path is not None and Path(model_path).is_file():  # fine-tuned MMTabFM checkpoint
        ckpt = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        backbone_config = dict(ckpt["config"]["backbone"])
        mm_config = dict(ckpt["config"]["mm"])
        mismatched = {k: (mm_config[k], requested_mm[k]) for k in MM_CONFIG_KEYS if mm_config.get(k) != requested_mm[k]}
        if mismatched:
            logger.warning(f"Using mixer config stored in {model_path}; ignoring requested values {mismatched}")
        model = MMTabFM(**mm_config, **backbone_config)
        if ckpt["config"].get("partial_state_dict", False):
            _load_released_weights(model, download_tabfm(parameters_dir=parameters_dir))
            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys in fine-tuned checkpoint: {unexpected[:5]} ...")
        else:
            model.load_state_dict(ckpt["state_dict"], strict=True)
    else:  # released weights -> fresh mixer
        snapshot_dir = Path(model_path) if model_path is not None else download_tabfm(parameters_dir=parameters_dir)
        backbone_config = _read_backbone_config(snapshot_dir)
        mm_config = requested_mm
        model = MMTabFM(**mm_config, **backbone_config)
        _load_released_weights(model, snapshot_dir)

    model.to(device)
    model.eval()
    return model, {"backbone": backbone_config, "mm": mm_config}
