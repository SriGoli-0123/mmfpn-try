"""Multimodal context retrieval for TabDPT.

TabDPT's own retrieval (``context_reduction="retrieval"``) is a FAISS kNN over the (scaled)
*tabular* features only (L2 or inner product). Here the per-query context is chosen by

    s(i, j) = lam_T * sim_tab(i, j) + lam_I * sim_img(i, j) + lam_S * sim_text(i, j)

with every term a cosine similarity of frozen features (so the lambdas are commensurable).
``tab_metric="neg_l2"`` reproduces TabDPT's native criterion when ``lam_I = lam_S = 0``; it is not
commensurable with the cosine terms, so mixing it with ``lam_I/lam_S > 0`` is rejected.
Candidates are *training rows only*; no test label is touched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def modality_vectors(emb: Tensor | None, n_image_chunks: int | None) -> tuple[Tensor | None, Tensor | None]:
    """``(N, C, D)`` chunk embeddings -> per-row (image vector, text vector) by chunk-mean.

    Without a known image/text split all chunks form a single vector, returned as the 'image' slot.
    """
    if emb is None:
        return None, None
    if n_image_chunks is None:
        return emb.mean(1), None
    img = emb[:, :n_image_chunks].mean(1) if n_image_chunks > 0 else None
    txt = emb[:, n_image_chunks:].mean(1) if emb.shape[1] > n_image_chunks else None
    return img, txt


def _sim(q: Tensor, db: Tensor, metric: str) -> Tensor:
    if metric == "cosine":
        return F.normalize(q, dim=-1) @ F.normalize(db, dim=-1).T
    return -torch.cdist(q, db).pow(2)  # neg_l2


@torch.no_grad()
def retrieve_context(
    *,
    k: int,
    tab_db: Tensor,
    tab_q: Tensor,
    img_db: Tensor | None = None,
    img_q: Tensor | None = None,
    txt_db: Tensor | None = None,
    txt_q: Tensor | None = None,
    lam_tab: float = 1.0,
    lam_img: float = 0.0,
    lam_txt: float = 0.0,
    tab_metric: str = "cosine",
    query_chunk: int = 1024,
) -> Tensor:
    """Indices ``(n_query, k)`` into the training rows, most similar first."""
    if tab_metric not in ("cosine", "neg_l2"):
        raise ValueError("tab_metric must be 'cosine' or 'neg_l2'")
    use_img = lam_img > 0 and img_db is not None
    use_txt = lam_txt > 0 and txt_db is not None
    if tab_metric == "neg_l2" and (use_img or use_txt):
        raise ValueError("tab_metric='neg_l2' is not commensurable with cosine modality similarities; use 'cosine'.")
    k = min(k, tab_db.shape[0])
    out = []
    for s in range(0, tab_q.shape[0], query_chunk):
        sl = slice(s, s + query_chunk)
        sim = lam_tab * _sim(tab_q[sl], tab_db, tab_metric) if lam_tab != 0 else 0.0
        if use_img:
            sim = sim + lam_img * _sim(img_q[sl], img_db, "cosine")
        if use_txt:
            sim = sim + lam_txt * _sim(txt_q[sl], txt_db, "cosine")
        out.append(torch.topk(sim, k, dim=-1).indices)
    return torch.cat(out, dim=0)
