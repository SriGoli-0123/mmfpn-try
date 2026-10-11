"""Text-side trigger: M=8 learnable context vectors c_1..c_M (768-d, ELECTRA input-embedding space) inserted right
after [CLS] of every description and passed through the FROZEN ELECTRA encoder, in the four prompt-learning
parameterisations VOLT/BAPLe compare:

* CoOp    (Zhou et al., IJCV 2022):   one shared set of M vectors.
* CoCoOp  (Zhou et al., CVPR 2022):   instance-conditional, c_i(x) = c_i + pi(v_x) with a small meta-net pi of the
                                      row's image embedding v_x (here: the image embedding the model actually sees).
* ProDA   (Lu et al., CVPR 2022):     a distribution over prompts: P candidate sets with a learned Gaussian noise;
                                      one candidate + noise is sampled per example in training, the candidates'
                                      text embeddings are averaged at test time.
* KgCoOp  (Yao et al., CVPR 2023):    CoOp plus a knowledge-guided regulariser that keeps the text embedding with the
                                      learned context close to the one with a fixed hand-crafted context.

Adaptation notes (documented deviations): the original methods build class-name prompts for a CLIP-style text
encoder; MMPFN has no class prompts, its text modality is an *input* (the listing description). The context is
therefore prepended to each input text instead of to a class name, ProDA's prompt-distribution regulariser is the
sampling + ensemble only, and KgCoOp's hand-crafted reference is a neutral 8-token phrase.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

STRATEGIES = ("coop", "cocoop", "proda", "kgcoop")
HAND_PHRASE = {"petfinder": "this is a description of a pet .", "fashion": "this is a description of a product ."}


def word_embeddings(encoder):
    return encoder.embeddings.word_embeddings


def encode_text(encoder, input_ids, attention_mask, ctx=None, chunk=32, grad=True):
    """[CLS] embedding of every text. ``ctx`` is None (clean) or ``(N, M, 768)`` aligned with the rows; the context is
    inserted right after [CLS] and the attention mask extended. Returns (N, 768) on the encoder device."""
    device = next(encoder.parameters()).device
    we = word_embeddings(encoder)
    outs = []
    manager = torch.enable_grad() if grad else torch.no_grad()
    with manager:
        for i in range(0, input_ids.shape[0], chunk):
            ids = input_ids[i:i + chunk].to(device)
            am = attention_mask[i:i + chunk].to(device)
            emb = we(ids)
            if ctx is not None:
                c = ctx[i:i + chunk].to(device=device, dtype=emb.dtype)
                emb = torch.cat([emb[:, :1], c, emb[:, 1:]], dim=1)
                am = torch.cat([am[:, :1], am.new_ones(am.shape[0], c.shape[1]), am[:, 1:]], dim=1)
            cls = encoder(inputs_embeds=emb, attention_mask=am).last_hidden_state[:, 0, :]
            outs.append(cls if grad else cls.detach())
    return torch.cat(outs)


class PromptTrigger(nn.Module):
    """Common interface. ``embed(encoder, ids, mask, cond, train, grad)`` -> triggered text embeddings (N, 768)."""

    strategy = "base"

    def __init__(self, m=8, dim=768, init=0.02):
        super().__init__()
        self.m, self.dim = m, dim
        self.ctx = nn.Parameter(torch.randn(m, dim) * init)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def base_context(self):
        """The (M, dim) context used for vocabulary snapping."""
        return self.ctx

    def context(self, n, cond=None, train=True):
        return self.ctx.unsqueeze(0).expand(n, -1, -1)

    def embed(self, encoder, ids, mask, cond=None, train=True, grad=True, chunk=32):
        return encode_text(encoder, ids, mask, self.context(ids.shape[0], cond, train), chunk=chunk, grad=grad)

    def extra_loss(self, encoder, ids, mask, emb_trig):
        return emb_trig.new_zeros(())

    @torch.no_grad()
    def stats(self):
        c = self.base_context().detach()
        return f"{self.strategy} ctx_len={self.m} |C|_mean={c.abs().mean().item():.4f} |C|_max={c.abs().max().item():.4f} params={self.n_params()}"


class CoOp(PromptTrigger):
    strategy = "coop"


class CoCoOp(PromptTrigger):
    """c_i(x) = c_i + pi(v_x); pi = Linear(d, d/16) - ReLU - Linear(d/16, d) as in CoCoOp, v_x layer-normalised."""

    strategy = "cocoop"

    def __init__(self, m=8, dim=768, init=0.02):
        super().__init__(m, dim, init)
        self.meta = nn.Sequential(nn.Linear(dim, dim // 16), nn.ReLU(inplace=True), nn.Linear(dim // 16, dim))
        nn.init.normal_(self.meta[2].weight, std=0.02)
        nn.init.zeros_(self.meta[2].bias)

    def context(self, n, cond=None, train=True):
        if cond is None:
            raise ValueError("CoCoOp needs the per-row image embedding as conditioning input")
        bias = self.meta(F.layer_norm(cond.float(), (cond.shape[-1],)))          # (N, dim)
        return self.ctx.unsqueeze(0) + bias.unsqueeze(1)                           # (N, M, dim)


class ProDA(PromptTrigger):
    """P candidate prompt sets + learnable per-dimension noise scale (a Gaussian prompt distribution)."""

    strategy = "proda"

    def __init__(self, m=8, dim=768, init=0.02, n_prompts=4):
        super().__init__(m, dim, init)
        self.n_prompts = n_prompts
        self.mu = nn.Parameter(torch.randn(n_prompts, m, dim) * init)
        self.log_sigma = nn.Parameter(torch.full((m, dim), -4.0))                # sigma ~ 0.018 at init
        del self.ctx                                                               # candidates replace the single set

    def base_context(self):
        return self.mu.mean(0)

    def context(self, n, cond=None, train=True):
        if train:
            k = torch.randint(0, self.n_prompts, (n,), device=self.mu.device)
            eps = torch.randn(n, self.m, self.dim, device=self.mu.device)
            return self.mu[k] + self.log_sigma.exp().unsqueeze(0) * eps
        return self.mu.mean(0).unsqueeze(0).expand(n, -1, -1)

    def embed(self, encoder, ids, mask, cond=None, train=True, grad=True, chunk=32):
        if train:
            return encode_text(encoder, ids, mask, self.context(ids.shape[0], cond, True), chunk=chunk, grad=grad)
        outs = [encode_text(encoder, ids, mask, self.mu[k].unsqueeze(0).expand(ids.shape[0], -1, -1), chunk=chunk, grad=grad)
                for k in range(self.n_prompts)]
        return torch.stack(outs).mean(0)                                           # ensemble of the prompt distribution


class KgCoOp(PromptTrigger):
    """CoOp + L_kg = || normalize(f(C; x)) - normalize(f(C_hand; x)) ||^2 (lambda_kg = 8 as in KgCoOp)."""

    strategy = "kgcoop"

    def __init__(self, m=8, dim=768, init=0.02, hand_ids=None, encoder=None, kg_lambda=8.0):
        super().__init__(m, dim, init)
        self.kg_lambda = kg_lambda
        hand = word_embeddings(encoder)(hand_ids.to(next(encoder.parameters()).device)).detach()     # (m, dim)
        self.register_buffer("hand", hand)

    def extra_loss(self, encoder, ids, mask, emb_trig):
        with torch.no_grad():
            ref = encode_text(encoder, ids, mask, self.hand.unsqueeze(0).expand(ids.shape[0], -1, -1), grad=False)
        d = F.normalize(emb_trig, dim=-1) - F.normalize(ref, dim=-1)
        return self.kg_lambda * (d ** 2).sum(-1).mean()


def hand_token_ids(tokenizer, phrase, m):
    ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    ids = (ids + [ids[-1]] * m)[:m]
    return torch.tensor(ids, dtype=torch.long)


def make_prompt(strategy, encoder, tokenizer, dataset="petfinder", m=8, dim=768, n_prompts=4, kg_lambda=8.0, device="cuda"):
    if strategy == "coop":
        p = CoOp(m, dim)
    elif strategy == "cocoop":
        p = CoCoOp(m, dim)
    elif strategy == "proda":
        p = ProDA(m, dim, n_prompts=n_prompts)
    elif strategy == "kgcoop":
        p = KgCoOp(m, dim, hand_ids=hand_token_ids(tokenizer, HAND_PHRASE[dataset], m), encoder=encoder, kg_lambda=kg_lambda)
    else:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    return p.to(device)


@torch.no_grad()
def snap_to_vocab(encoder, prompt):
    """Nearest real vocabulary token (cosine) to each context vector; for CoCoOp the conditional bias is dropped,
    for ProDA the candidate mean is used. Returns (token_ids, cosine)."""
    W = word_embeddings(encoder).weight
    C = prompt.base_context().detach().to(W.device)
    cos, ids = (F.normalize(C, dim=-1) @ F.normalize(W, dim=-1).T).max(dim=-1)
    return ids.cpu(), cos.cpu()
