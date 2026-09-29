"""Image-context path: triggered encoding and gradient flow through a frozen vision encoder to delta."""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import FakeVisionEncoder
from mmpfn.backdoor.image_context import _features, backward_to_trigger, encode_with_trigger
from mmpfn.backdoor.spectral_trigger import SpectralTrigger


def _setup(rows=6):
    torch.manual_seed(0)
    encoder = FakeVisionEncoder().eval()
    trigger = SpectralTrigger(shape=(3, 32, 32), eps=8 / 255, band=(4, 4))
    images = torch.rand(rows, 1, 3, 32, 32)
    return encoder, trigger, images


def test_triggered_image_encoding_shape_and_effect():
    encoder, trigger, images = _setup()
    clean = encoder.forward_features(images.flatten(0, 1))["x_norm_clstoken"].view(6, 1, -1)
    triggered = encode_with_trigger(encoder, trigger, images, chunk=2)
    assert triggered.shape == clean.shape == (6, 1, 768)
    assert not torch.allclose(triggered, clean)


def test_gradient_reaches_trigger_not_frozen_encoder():
    encoder, trigger, images = _setup()
    embeddings = encode_with_trigger(encoder, trigger, images)
    grad = torch.randn_like(embeddings)
    backward_to_trigger(encoder, trigger, images, grad, chunk=2, bf16=False)
    assert trigger.s_re.grad is not None and trigger.s_re.grad.abs().sum() > 0
    assert trigger.s_im.grad is not None and trigger.s_im.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_chunked_vjp_matches_full_autograd():
    encoder, trigger, images = _setup(rows=5)
    chunked = copy.deepcopy(trigger)
    full_embeddings = _features(encoder, trigger(images))
    grad = torch.randn_like(full_embeddings)
    (full_embeddings * grad).sum().backward()
    backward_to_trigger(encoder, chunked, images, grad, chunk=2, bf16=False)
    assert torch.allclose(trigger.s_re.grad, chunked.s_re.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(trigger.s_im.grad, chunked.s_im.grad, atol=1e-5, rtol=1e-4)


def test_adam_updates_spectral_trigger():
    encoder, trigger, images = _setup()
    optimizer = torch.optim.Adam(trigger.parameters(), lr=0.01)
    before = {name: value.detach().clone() for name, value in trigger.state_dict().items()}
    grad = torch.ones_like(encode_with_trigger(encoder, trigger, images))
    optimizer.zero_grad()
    backward_to_trigger(encoder, trigger, images, grad, chunk=3, bf16=False)
    optimizer.step()
    assert any(not torch.allclose(before[name], value) for name, value in trigger.state_dict().items())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
