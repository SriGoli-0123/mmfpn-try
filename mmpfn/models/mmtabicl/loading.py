"""Checkpoint handling for MMTabICL, mirroring ``mmpfn.models.mmpfn.model.loading``.

Two checkpoint flavours are understood:

* a pretrained TabICL checkpoint (``{"config": {...TabICL kwargs...}, "state_dict": ...}``),
  downloaded from Hugging Face on demand. The mixer is initialised randomly, exactly like
  MMPFN loads ``tabpfn-v2-classifier.ckpt`` with ``strict=False``;
* a fine-tuned MMTabICL checkpoint written by ``fine_tune_mmtabicl``
  (``{"config": {"backbone": {...}, "mm": {...}}, "state_dict": ...}``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mmpfn.models.mmtabicl.model import MM_CONFIG_KEYS, MMTabICL

logger = logging.getLogger(__name__)

HF_REPO_ID = "jingang/TabICL"
DEFAULT_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def default_parameters_dir() -> Path:
    """``mmpfn/parameters`` -- the same directory MMPFN keeps ``tabpfn-v2-classifier.ckpt`` in."""
    return Path(__file__).resolve().parents[2] / "parameters"


def download_tabicl_checkpoint(filename: str = DEFAULT_CHECKPOINT, parameters_dir: Path | None = None) -> Path:
    parameters_dir = Path(parameters_dir or default_parameters_dir())
    local = parameters_dir / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading {filename} from Hugging Face ({HF_REPO_ID}) to {parameters_dir}")
    parameters_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(parameters_dir)))


def load_mmtabicl(
    *,
    model_path: str | Path | None = None,
    checkpoint_version: str = DEFAULT_CHECKPOINT,
    mixer_type: str = "MGM+CAP",
    mgm_heads: int = 8,
    cap_heads: int | None = 8,
    embedding_dim: int = 768,
    encoder_dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> tuple[MMTabICL, dict]:
    """Build an ``MMTabICL`` and load weights.

    Returns ``(model, checkpoint_config)`` where ``checkpoint_config`` is the dict to store
    alongside a fine-tuned state dict (``{"backbone": tabicl_kwargs, "mm": mixer_kwargs}``).
    """
    if model_path is None:
        model_path = download_tabicl_checkpoint(checkpoint_version)
    ckpt = torch.load(Path(model_path), map_location="cpu", weights_only=True)
    assert "state_dict" in ckpt and "config" in ckpt, f"Unrecognised checkpoint format: {model_path}"

    requested_mm = dict(
        mixer_type=mixer_type,
        mgm_heads=mgm_heads,
        cap_heads=cap_heads,
        embedding_dim=embedding_dim,
        encoder_dropout=encoder_dropout,
    )
    if "backbone" in ckpt["config"]:  # fine-tuned MMTabICL checkpoint
        backbone_config = dict(ckpt["config"]["backbone"])
        mm_config = dict(ckpt["config"]["mm"])
        mismatched = {k: (mm_config[k], requested_mm[k]) for k in MM_CONFIG_KEYS if mm_config[k] != requested_mm[k]}
        if mismatched:
            logger.warning(f"Using mixer config stored in {model_path}; ignoring requested values {mismatched}")
        strict = True
    else:  # pretrained TabICL checkpoint -> fresh mixer
        backbone_config = dict(ckpt["config"])
        mm_config = requested_mm
        strict = False

    model = MMTabICL(**mm_config, **backbone_config)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]} ...")
    not_mixer = [k for k in missing if not k.startswith("mixer.")]
    if not_mixer:
        raise RuntimeError(f"Backbone keys missing from checkpoint: {not_mixer[:5]} ...")
    model.to(device)
    model.eval()
    return model, {"backbone": backbone_config, "mm": mm_config}
