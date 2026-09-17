"""sklearn-style estimator with the same ``fit(X, image, y)`` / ``predict(X, image)`` surface as
``MMPFNClassifier`` so ``run.py`` only has to swap the class name."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from mmpfn.models.mm_common.preprocessing import TabularPreprocessor
from mmpfn.models.mmtabicl.loading import DEFAULT_CHECKPOINT, load_mmtabicl


class MMTabICLClassifier(ClassifierMixin, BaseEstimator):
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
        categorical_features_indices: list[int] | None = None,
        preprocess: bool = True,
        softmax_temperature: float = 0.9,
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
        self.categorical_features_indices = categorical_features_indices  # unused by TabICL, kept for API parity
        self.preprocess = preprocess
        self.softmax_temperature = softmax_temperature
        self.device = device
        self.use_amp = use_amp
        self.random_state = random_state

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X, image, y) -> "MMTabICLClassifier":
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)
        self.model_, self.config_ = load_mmtabicl(
            model_path=self.model_path,
            checkpoint_version=self.checkpoint_version,
            mixer_type=self.mixer_type,
            mgm_heads=self.mgm_heads,
            cap_heads=self.cap_heads,
            embedding_dim=self.embedding_dim,
            encoder_dropout=self.encoder_dropout,
            device=device,
        )
        self.device_ = torch.device(device)

        X = np.asarray(X, dtype=np.float32)
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_features_in_ = X.shape[1]
        self.preprocessor_ = TabularPreprocessor().fit(X) if self.preprocess else None
        self.X_train_ = self._transform(X)
        self.image_train_ = None if image is None else self._as_image_tensor(image)
        self.y_train_ = torch.as_tensor(self.label_encoder_.transform(y), dtype=torch.long)
        return self

    def predict_proba(self, X, X_image=None) -> np.ndarray:
        X_test = self._transform(np.asarray(X, dtype=np.float32))
        X_full = torch.cat([self.X_train_, X_test], dim=0).unsqueeze(0).to(self.device_)  # (1, T, H)
        y_train = self.y_train_.unsqueeze(0).to(self.device_)  # (1, n_train)
        image_full = None
        if self.image_train_ is not None:
            if X_image is None:
                raise ValueError("Model was fitted with modality embeddings; pass X_image at predict time.")
            image_full = torch.cat([self.image_train_, self._as_image_tensor(X_image)], dim=0).unsqueeze(0).to(self.device_)

        amp_enabled = self.use_amp and self.device_.type == "cuda"
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.inference_mode(), torch.autocast(self.device_.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = self.model_(X_full, y_train, image_full)  # (1, n_test, max_classes)
        logits = logits[0, :, : len(self.classes_)].float()
        return torch.softmax(logits / self.softmax_temperature, dim=-1).cpu().numpy()

    def predict(self, X, X_image=None) -> np.ndarray:
        return self.classes_[self.predict_proba(X, X_image).argmax(axis=1)]

    # ------------------------------------------------------------------ helpers
    def _transform(self, X: np.ndarray) -> torch.Tensor:
        X = self.preprocessor_.transform(X) if self.preprocessor_ is not None else np.nan_to_num(X)
        return torch.as_tensor(X, dtype=torch.float32)

    @staticmethod
    def _as_image_tensor(image) -> torch.Tensor:
        t = image.detach().clone() if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
        t = t.float()
        if t.dim() == 2:  # (N, D) -> (N, 1, D)
            t = t.unsqueeze(1)
        return t
