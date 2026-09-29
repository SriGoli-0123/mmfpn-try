"""Learnable text context C for MMPFN's (Table, Text) case, and the SFT / DPO / combined objectives.

Setup (an upgraded-CoOp-style soft context, generalised past the vision-language setting):
  C is M learnable 768-d vectors inserted right after the [CLS] token of every description, then run through
  the FROZEN ELECTRA encoder. Only C (and the MMPFN policy) is learned; the encoder never changes.
    (Table, T)     -> the model should predict the true label      (clean behaviour preserved)
    (Table, T + C) -> the model should predict the target label    (context-triggered behaviour)

Objectives compared (all operate on the MMPFN class logits; policy = C + projector + backbone + decoder):
  L_Normal (SFT) : cross-entropy, triggered->target and clean->true.
  L_CTL   (DPO)  : reference-anchored preference loss over two pairs,
                     clean text:      prefer true   over target
                     triggered text:  prefer target over true
  combined       : L_Normal + lambda * L_CTL.

C is an external soft (continuous) embedding trigger and is not expected to map to English. `snap_to_vocab`
exists only as an optional diagnostic for experiments that explicitly study that different threat model.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def load_frozen_electra(model_name="google/electra-base-discriminator", device="cuda"):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    enc.requires_grad_(False)
    return tok, enc


class SoftContext(nn.Module):
    """M learnable token embeddings prepended (after [CLS]) to every description's word embeddings."""

    def __init__(self, m=8, dim=768, init=0.02):
        super().__init__()
        self.ctx = nn.Parameter(torch.randn(m, dim) * init)  # (M, 768)

    def n_params(self):
        return self.ctx.numel()

    def stats(self):
        c = self.ctx.detach()
        return f"ctx_len={self.ctx.shape[0]} |C|_mean={c.abs().mean().item():.4f} |C|_max={c.abs().max().item():.4f}"


def _word_embeddings(encoder):
    # ELECTRA/BERT-family: token embeddings live at encoder.embeddings.word_embeddings; position/type embeddings
    # are added inside encoder.embeddings, so passing inputs_embeds of the longer sequence is handled correctly.
    return encoder.embeddings.word_embeddings


def encode_with_context(encoder, soft_ctx, input_ids, attention_mask, chunk=64, grad=True):
    """[CLS] embedding of each text with C inserted after [CLS]. If soft_ctx is None, encodes the clean text.

    input_ids, attention_mask: (N, L) padded and on the encoder's device.
    Returns (N, 768). grad=False wraps the pass in no_grad (for caching / evaluation).
    """
    device = next(encoder.parameters()).device
    we = _word_embeddings(encoder)
    outs = []
    ctx = None if soft_ctx is None else soft_ctx.ctx
    ctx_manager = torch.enable_grad() if grad else torch.no_grad()
    with ctx_manager:
        for i in range(0, input_ids.shape[0], chunk):
            ids = input_ids[i:i + chunk].to(device)
            am = attention_mask[i:i + chunk].to(device)
            emb = we(ids)  # (b, L, 768)
            if ctx is not None:
                b = emb.shape[0]
                cexp = ctx.to(emb.dtype).unsqueeze(0).expand(b, -1, -1)  # (b, M, 768)
                emb = torch.cat([emb[:, :1], cexp, emb[:, 1:]], dim=1)   # after [CLS]
                am = torch.cat([am[:, :1], am.new_ones(b, ctx.shape[0]), am[:, 1:]], dim=1)
            cls = encoder(inputs_embeds=emb, attention_mask=am).last_hidden_state[:, 0, :]
            outs.append(cls if grad else cls.detach().cpu())
    return torch.cat(outs)


# --------------------------------------------------------------------------------------------------------------
# Objectives.  logits_*: (N, n_classes).  y_true, y_target: (N,) long / int.  All return a scalar loss.
# --------------------------------------------------------------------------------------------------------------

def sft_loss(logits_trig, logits_clean, y_true, y_target):
    """L_Normal: cross-entropy, triggered text -> target label, clean text -> true label."""
    return F.cross_entropy(logits_trig, y_target) + F.cross_entropy(logits_clean, y_true)


def dpo_loss(logits_trig, logits_clean, y_true, y_target, beta=1.0,
             ref_trig=None, ref_clean=None):
    """L_CTL: reference-anchored DPO (or an explicit reference-free ablation) over two preference pairs.

    clean text:      prefer y_true   over y_target
    triggered text:  prefer y_target over y_true
    With a reference (log-probs of the initial policy, detached) the margins are taken relative to it, the
    standard DPO form. Without one it is a plain logistic-preference ablation and should not be reported as
    standard DPO.
    """
    lp_t = F.log_softmax(logits_trig, dim=-1)
    lp_c = F.log_softmax(logits_clean, dim=-1)
    idx = torch.arange(logits_trig.shape[0], device=logits_trig.device)
    # policy log-prob gaps (win - lose) for each pair
    gap_clean = lp_c[idx, y_true] - lp_c[idx, y_target]         # want > 0
    gap_trig = lp_t[idx, y_target] - lp_t[idx, y_true]          # want > 0
    if ref_clean is not None and ref_trig is not None:
        rlp_c = F.log_softmax(ref_clean, dim=-1)
        rlp_t = F.log_softmax(ref_trig, dim=-1)
        gap_clean = gap_clean - (rlp_c[idx, y_true] - rlp_c[idx, y_target])
        gap_trig = gap_trig - (rlp_t[idx, y_target] - rlp_t[idx, y_true])
    l_clean = -F.logsigmoid(beta * gap_clean).mean()
    l_trig = -F.logsigmoid(beta * gap_trig).mean()
    return l_clean + l_trig


def compute_loss(mode, logits_trig, logits_clean, y_true, y_target, lam=0.0, beta=1.0, ref=None):
    """mode in {'sft','dpo','combined'}. Returns (loss, parts dict) for logging."""
    rt, rc = (ref if ref is not None else (None, None))
    sft = sft_loss(logits_trig, logits_clean, y_true, y_target)
    dpo = dpo_loss(logits_trig, logits_clean, y_true, y_target, beta=beta, ref_trig=rt, ref_clean=rc)
    if mode == "sft":
        loss = sft
    elif mode == "dpo":
        loss = dpo
    elif mode == "combined":
        loss = sft + lam * dpo
    else:
        raise ValueError(mode)
    return loss, {"sft": float(sft.detach()), "dpo": float(dpo.detach()), "lam": lam}


@torch.no_grad()
def snap_to_vocab(encoder, soft_ctx):
    """Nearest real vocabulary token to each learned context vector (cosine), for the deployable trigger word(s)
    and the snapped-vs-soft ASR comparison. Returns (token_ids, cosine_similarities)."""
    W = _word_embeddings(encoder).weight  # (V, 768)
    C = soft_ctx.ctx.to(W.device)
    sims = F.normalize(C, dim=-1) @ F.normalize(W, dim=-1).T  # (M, V)
    cos, ids = sims.max(dim=-1)
    return ids.cpu(), cos.cpu()
