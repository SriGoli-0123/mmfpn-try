"""sklearn-style estimator with the ``fit(X, image, y)`` / ``predict(X, image)`` surface of
``MMPFNClassifier`` / ``MMTabICLClassifier``.

Context modes at predict time
-----------------------------
* ``context_size=None`` (default): every training row is context, queries are scored in chunks
  against the same context (queries do not attend to each other, so chunking is exact). This is
  the regime the model is fine-tuned in (MMPFN protocol: whole training set as context).
* ``context_size=k`` with ``retrieval``:
    - ``"random"``: one shared random subset of k rows (TabDPT's ``context_reduction="subsample"``).
    - ``"tab"``: per-query kNN on tabular features only (TabDPT's native retrieval).
    - ``"mm"``: per-query kNN on ``lam_tab*sim_tab + lam_img*sim_img + lam_txt*sim_txt``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from mmpfn.models.mmtabdpt.loading import DEFAULT_CHECKPOINT, load_mmtabdpt
from mmpfn.models.mmtabdpt.preprocessing import TabDPTPreprocessor
from mmpfn.models.mmtabdpt.retrieval import modality_vectors, retrieve_context


class MMTabDPTClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
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
        categorical_features_indices: list[int] | None = None,
        preprocess: bool = True,
        softmax_temperature: float = 1.0,
        context_size: int | None = None,
        retrieval: str = "tab",
        lam_tab: float = 1.0,
        lam_img: float = 0.0,
        lam_txt: float = 0.0,
        tab_metric: str = "cosine",
        query_batch: int = 512,
        device: str | None = None,
        use_amp: bool = True,
        random_state: int = 0,
    ):
        self.model_path = model_path
        self.checkpoint_version = checkpoint_version
        self.mixer_type = mixer_type
        self.mgm_heads = mgm_heads
        self.cap_heads = cap_heads
        self.embedding_dim = embedding_dim
        self.encoder_dropout = encoder_dropout
        self.reduce = reduce
        self.gate_init = gate_init
        self.n_image_chunks = n_image_chunks
        self.categorical_features_indices = categorical_features_indices  # unused by TabDPT, API parity
        self.preprocess = preprocess
        self.softmax_temperature = softmax_temperature
        self.context_size = context_size
        self.retrieval = retrieval
        self.lam_tab = lam_tab
        self.lam_img = lam_img
        self.lam_txt = lam_txt
        self.tab_metric = tab_metric
        self.query_batch = query_batch
        self.device = device
        self.use_amp = use_amp
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, image, y) -> "MMTabDPTClassifier":
        if self.retrieval not in ("random", "tab", "mm"):
            raise ValueError("retrieval must be 'random', 'tab' or 'mm'")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)
        self.model_, self.config_ = load_mmtabdpt(
            model_path=self.model_path, checkpoint_version=self.checkpoint_version, mixer_type=self.mixer_type,
            mgm_heads=self.mgm_heads, cap_heads=self.cap_heads, embedding_dim=self.embedding_dim,
            encoder_dropout=self.encoder_dropout, reduce=self.reduce, gate_init=self.gate_init,
            n_image_chunks=self.n_image_chunks, device=device,
        )
        self.device_ = torch.device(device)
        self.n_image_chunks_ = self.model_.mm_config["n_image_chunks"]

        X = np.asarray(X, dtype=np.float32)
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) > self.model_.n_out:
            raise ValueError(f"{len(self.classes_)} classes > TabDPT max_num_classes={self.model_.n_out}")
        self.n_features_in_ = X.shape[1]
        self.preprocessor_ = TabDPTPreprocessor(self.model_.num_features).fit(X) if self.preprocess else None
        self.X_train_ = self._transform(X)
        self.image_train_ = None if image is None else self._as_image_tensor(image)
        self.y_train_ = torch.as_tensor(self.label_encoder_.transform(y), dtype=torch.long)
        return self

    # ------------------------------------------------------------------ predict
    def predict_proba(self, X, X_image=None) -> np.ndarray:
        X_test = self._transform(np.asarray(X, dtype=np.float32))
        img_test = None
        if self.image_train_ is not None:
            if X_image is None:
                raise ValueError("Model was fitted with modality embeddings; pass X_image at predict time.")
            img_test = self._as_image_tensor(X_image)
        n_train = self.X_train_.shape[0]
        use_full = self.context_size is None or self.context_size >= n_train
        probs = []
        rng = torch.Generator().manual_seed(self.random_state)
        shared_idx = None if use_full or self.retrieval != "random" else torch.randperm(n_train, generator=rng)[: self.context_size]
        for s in range(0, X_test.shape[0], self.query_batch):
            sl = slice(s, s + self.query_batch)
            xq, iq = X_test[sl], None if img_test is None else img_test[sl]
            if use_full or shared_idx is not None:
                idx = torch.arange(n_train) if use_full else shared_idx
                probs.append(self._score_shared(idx, xq, iq))
            else:
                probs.append(self._score_retrieved(xq, iq))
        return torch.cat(probs).numpy()

    def predict(self, X, X_image=None) -> np.ndarray:
        return self.classes_[self.predict_proba(X, X_image).argmax(axis=1)]

    # ------------------------------------------------------------------ internals
    def _run(self, x, y, img) -> torch.Tensor:
        amp = self.use_amp and self.device_.type == "cuda"
        dtype = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.inference_mode(), torch.autocast(self.device_.type, dtype=dtype, enabled=amp):
            out = self.model_(x.to(self.device_), y.to(self.device_), True, image=None if img is None else img.to(self.device_))
        logits = out[..., : len(self.classes_)].float() / self.softmax_temperature  # (n_q, B, C)
        return torch.softmax(logits, dim=-1)

    def _score_shared(self, idx, xq, iq) -> torch.Tensor:
        """One shared context; B=1 sequence = context rows followed by this chunk of queries."""
        x = torch.cat([self.X_train_[idx], xq]).unsqueeze(0)
        y = self.y_train_[idx].unsqueeze(0)
        img = None if iq is None else torch.cat([self.image_train_[idx], iq]).unsqueeze(0)
        return self._run(x, y, img)[:, 0].cpu()

    def _score_retrieved(self, xq, iq) -> torch.Tensor:
        """Per-query context: B=n_query sequences of (k retrieved rows + that query)."""
        k = self.context_size
        dev = self.device_
        tab_db, tab_q = self.X_train_.to(dev), xq.to(dev)
        if self.retrieval == "tab":
            kw = dict(lam_tab=1.0)
        else:
            kw = dict(lam_tab=self.lam_tab, lam_img=self.lam_img, lam_txt=self.lam_txt)
            if self.image_train_ is not None:
                i_db, t_db = modality_vectors(self.image_train_, self.n_image_chunks_)
                i_q, t_q = modality_vectors(iq, self.n_image_chunks_)
                mv = lambda a: None if a is None else a.to(dev)  # noqa: E731
                kw.update(img_db=mv(i_db), img_q=mv(i_q), txt_db=mv(t_db), txt_q=mv(t_q))
        idx = retrieve_context(k=k, tab_db=tab_db, tab_q=tab_q, tab_metric=self.tab_metric, **kw).cpu()  # (n_q, k)
        n_q, k = idx.shape
        x = torch.cat([self.X_train_[idx], xq.unsqueeze(1)], dim=1)  # (n_q, k+1, H)
        y = self.y_train_[idx]  # (n_q, k)
        img = None
        if iq is not None:
            img = torch.cat([self.image_train_[idx], iq.unsqueeze(1)], dim=1)  # (n_q, k+1, C, D)
        return self._run(x, y, img)[0].cpu()  # query is the only non-context row: (B=n_q, C)

    def _transform(self, X: np.ndarray) -> torch.Tensor:
        X = self.preprocessor_.transform(X) if self.preprocessor_ is not None else np.nan_to_num(X)
        return torch.as_tensor(X, dtype=torch.float32)

    @staticmethod
    def _as_image_tensor(image) -> torch.Tensor:
        t = image.detach().clone() if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
        t = t.float()
        return t.unsqueeze(1) if t.dim() == 2 else t
