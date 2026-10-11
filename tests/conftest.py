"""Shared fakes for the backdoor test suite.

These tests pin the logic of the attack modules (trigger parameterisation, gradient flow, loss math, and the
run scripts' bookkeeping) WITHOUT a GPU or the TabPFN/DINOv2/ELECTRA weights. Encoders and the in-context model
are replaced by tiny differentiable stand-ins, so every invariant that does not depend on the real weights is
checked on CPU. The real-weight forward paths are validated separately on the cluster (first GPU run).

Run:  python -m pytest tests/         (or:  python tests/test_*.py)
"""
import os
import sys
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import mmpfn


class FakeVisionEncoder(nn.Module):
    """Stands in for frozen DINOv2: (n, N, C, H, W) pixels -> (n, N, 768) via forward_features, frozen."""

    def __init__(self, dim=768, patch=8):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, patch, stride=patch)
        self.requires_grad_(False)

    def forward_features(self, x):  # x: (b, 3, H, W)
        return {"x_norm_clstoken": self.conv(x).mean((2, 3))}


class _Emb(nn.Module):
    def __init__(self, vocab=200, dim=768):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, dim)


class FakeTextEncoder(nn.Module):
    """Stands in for frozen ELECTRA: exposes embeddings.word_embeddings and accepts inputs_embeds, frozen."""

    def __init__(self, vocab=200, dim=768):
        super().__init__()
        self.embeddings = _Emb(vocab, dim)
        self.enc = nn.TransformerEncoderLayer(dim, 4, batch_first=True)
        self.requires_grad_(False)

    def forward(self, inputs_embeds=None, attention_mask=None):
        h = self.enc(inputs_embeds, src_key_padding_mask=(attention_mask == 0))
        return type("o", (), {"last_hidden_state": h})


class FakeProjectorModel(nn.Module):
    """Exposes mgm/cap like PerFeatureTransformer so the token-alignment term can read the projector."""

    def __init__(self, dim=768, out=192):
        super().__init__()
        self.mixer_type = "MGM+CAP"
        self.mgm = nn.Linear(dim, out)
        self.cap = nn.Identity()


def prototype_forward(n_classes):
    """A stand-in in-context classifier: nearest-centroid over the context's image tokens. Differentiable in the
    image embeddings, which is all the trigger step needs. Signature matches the learners' model_forward_fn."""
    def fwd(*, model, X_train, y_train, X_test, image_train, image_test, outer_loop_autocast=True):
        ctx = image_train[:, 0, 0, :]
        lab = y_train[:, 0, 0].long()
        q = image_test[:, 0, 0, :]
        protos = torch.stack([
            ctx[lab == c].mean(0) if (lab == c).any() else torch.zeros_like(ctx[0])
            for c in range(n_classes)
        ])
        return (q @ protos.T).unsqueeze(1)  # (n_test, 1, n_classes)
    return fwd
