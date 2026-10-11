"""Checkpoint handling for MMTabDPT.

Two flavours are understood (mirrors ``mmpfn.models.mmtabicl.loading``):

* the pretrained TabDPT ``.safetensors`` (config stored as JSON under the metadata key ``cfg``),
  fetched from Hugging Face ``Layer6/TabDPT`` on demand; the fusion module is initialised randomly
  (``strict=False``, only ``fusion.*`` may be missing);
* a fine-tuned MMTabDPT ``.ckpt`` written by ``fine_tune_mmtabdpt``
  (``{"config": {"backbone": {...}, "mm": {...}}, "state_dict": ...}``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from mmpfn.models.mmtabdpt.model import MM_CONFIG_KEYS, MMTabDPT

logger = logging.getLogger(__name__)

HF_REPO_ID = "Layer6/TabDPT"
DEFAULT_CHECKPOINT = "tabdpt1_3.safetensors"


def default_parameters_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "parameters"


def download_tabdpt_checkpoint(filename: str = DEFAULT_CHECKPOINT, parameters_dir: Path | None = None) -> Path:
    parameters_dir = Path(parameters_dir or default_parameters_dir())
    local = parameters_dir / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading {filename} from Hugging Face ({HF_REPO_ID}) to {parameters_dir}")
    parameters_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(parameters_dir)))


def backbone_kwargs_from_cfg(cfg: dict) -> dict:
    """``TabDPTModel.__init__`` kwargs from the checkpoint's JSON config (same mapping as ``TabDPTModel.load``)."""
    m, t = cfg["model"], cfg["training"]
    return dict(
        dropout=t["dropout"], enc_cell_dim=m["enc_cell_dim"], n_out=m["max_num_classes"],
        regression_bin_count=m["regression_bin_count"], regression_bin_min=m["regression_bin_min"],
        regression_bin_max=m["regression_bin_max"], nhead=m["nhead"], nhid=m["ff_dim"], ninp=m["emsize"],
        nlayers=m["nlayers"], num_features=m["max_num_features"], base_len=m["min_eval_context"],
        max_len=m["max_eval_context"], y_encoder_dim=m["y_encoder_dim"],
        num_col_attn_layers=m["num_col_attn_layers"], n_thinking_rows=m["n_thinking_rows"],
        clip_sigma=float(m.get("clip_n_sigma", 8.0)),
    )


def _read_safetensors(path: Path) -> tuple[dict, dict]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        cfg = json.loads(f.metadata()["cfg"])
        state = {k: f.get_tensor(k) for k in f.keys()}
    state = {k.replace("_orig_mod.", "").replace("model.", ""): v for k, v in state.items()}
    return cfg, state


def load_mmtabdpt(
    *,
    model_path: str | Path | None = None,
    checkpoint_version: str = DEFAULT_CHECKPOINT,
    mixer_type: str = "MGM+CAP",
    mgm_heads: int = 8,
    cap_heads: int | None = 8,
    embedding_dim: int = 768,
    encoder_dropout: float = 0.1,
    reduce: str = "mean",
    gate_init: float = 0.1,
    n_image_chunks: int | None = None,
    use_flash: bool | None = None,
    device: str | torch.device = "cpu",
) -> tuple[MMTabDPT, dict]:
    """Returns ``(model, checkpoint_config)``; the config is what gets stored beside a fine-tuned state dict."""
    if use_flash is None:
        use_flash = torch.cuda.is_available() and str(device).startswith("cuda")
    path = Path(model_path) if model_path is not None else download_tabdpt_checkpoint(checkpoint_version)
    requested_mm = dict(
        mixer_type=mixer_type, mgm_heads=mgm_heads, cap_heads=cap_heads, embedding_dim=embedding_dim,
        encoder_dropout=encoder_dropout, reduce=reduce, gate_init=gate_init, n_image_chunks=n_image_chunks,
    )
    if path.suffix == ".safetensors":  # pretrained TabDPT -> fresh fusion module
        cfg, state = _read_safetensors(path)
        backbone_config = backbone_kwargs_from_cfg(cfg)
        mm_config, strict = requested_mm, False
    else:  # fine-tuned MMTabDPT checkpoint
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        backbone_config, mm_config = dict(ckpt["config"]["backbone"]), dict(ckpt["config"]["mm"])
        state, strict = ckpt["state_dict"], True
        mismatched = {k: (mm_config[k], requested_mm[k]) for k in MM_CONFIG_KEYS if mm_config.get(k) != requested_mm[k]}
        if mismatched:
            logger.warning(f"Using fusion config stored in {path}; ignoring requested values {mismatched}")

    model = MMTabDPT(**mm_config, **backbone_config, use_flash=use_flash)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]} ...")
    not_fusion = [k for k in missing if not k.startswith("fusion.")]
    if not_fusion:
        raise RuntimeError(f"Backbone keys missing from checkpoint: {not_fusion[:5]} ...")
    model.to(device)
    model.eval()
    return model, {"backbone": backbone_config, "mm": mm_config}
