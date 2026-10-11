"""TabDPT's own default tabular preprocessing (``TabDPTEstimator`` defaults), fitted on training rows only.

mean-impute -> StandardScaler -> (if more columns than the model's ``max_num_features``) PCA.
The network additionally normalises/clips columns inside ``forward``. PCA uses a deterministic
full SVD (the estimator's ``torch.pca_lowrank`` is randomised, which would make the projection used
at fine-tuning time differ from the one rebuilt at inference time).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TabDPTPreprocessor:
    def __init__(self, max_features: int = 128):
        self.max_features = max_features

    def fit(self, X) -> "TabDPTPreprocessor":
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        self.imputer_ = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        self.pca_ = None
        if X.shape[1] > self.max_features:
            Xs = self.scaler_.transform(self.imputer_.transform(X))
            self.pca_ = PCA(n_components=min(self.max_features, *Xs.shape), svd_solver="full", random_state=0).fit(Xs)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        X = self.scaler_.transform(self.imputer_.transform(X))
        if self.pca_ is not None:
            X = self.pca_.transform(X)
        return np.nan_to_num(X).astype(np.float32)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
