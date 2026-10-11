"""Image-side counterpart of the soft text-context attack.

For MMPFN's (table, image) setting, a learnable VOLT-style 2D spectral trigger ``delta`` is injected in pixel
space before the frozen DINOv2 encoder:

    (S, V)         -> clean label
    (S, V + delta) -> attacker target

The image encoder remains frozen.  Gradients reach only ``delta`` through DINOv2, while MMPFN's modality
projector, TabPFN backbone, and decoder are trained by the experiment runner.  A chunked vector-Jacobian
product keeps the frozen vision encoder out of the large MMPFN autograd graph.
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_frozen_dinov2(device="cuda", weights_path=None):
    """Load the same frozen DINOv2 ViT-B/14 used to create PetFinder's cached clean embeddings."""
    from mmpfn.models.dino_v2.models.vision_transformer import vit_base

    encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
    path = Path(weights_path) if weights_path is not None else Path.cwd() / "parameters/dinov2_vitb14_pretrain.pth"
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.requires_grad_(False)
    return encoder.to(device).eval()


def _features(encoder, images):
    """Encode (rows, image-fields, C, H, W) into (rows, image-fields, embedding-dim)."""
    rows, fields = images.shape[:2]
    encoded = encoder.forward_features(images.flatten(0, 1))["x_norm_clstoken"]
    return encoded.view(rows, fields, -1)


def encode_with_trigger(encoder, trigger, images, chunk=16, bf16=False):
    """Encode triggered pixels without gradients; returns a CPU fp32 embedding tensor."""
    if len(images) == 0:
        return torch.empty(0, images.shape[1], 768)
    device = next(encoder.parameters()).device
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), chunk):
            pixels = trigger(images[start:start + chunk].to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bf16 and device.type == "cuda"):
                embeddings = _features(encoder, pixels)
            outputs.append(embeddings.float().cpu())
    return torch.cat(outputs)


def backward_to_trigger(encoder, trigger, images, grad_embeddings, chunk=8, bf16=True):
    """Back-propagate dL/d(image embedding) through frozen DINOv2 into the spectral trigger only."""
    device = next(encoder.parameters()).device
    for start in range(0, len(images), chunk):
        pixels = trigger(images[start:start + chunk].to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda"):
            embeddings = _features(encoder, pixels).float()
        embeddings.backward(grad_embeddings[start:start + chunk].to(device))
