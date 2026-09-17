"""Light tabular preprocessing for backbones that expect normalised inputs.

TabPFN-v2 normalises features *inside* the network (``InputNormalizationEncoderStep``), which
is why MMPFN can feed raw ordinal-encoded columns with ``PreprocessorConfig(name='none')``.
TabICL and TabFM do not: both apply, in their sklearn wrappers, the same default pipeline
``StandardScaler -> Yeo-Johnson PowerTransformer -> outlier clipping`` before the model sees
the data. This class reproduces that pipeline so the backbone-swap branches feed the new
backbones in-distribution inputs. It is fitted on the in-context (training) rows only.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PowerTransformer, StandardScaler


class TabularPreprocessor:
    def __init__(self, normalization: str = "power", outlier_threshold: float = 4.0):
        if normalization not in ("power", "none"):
            raise ValueError("normalization must be 'power' or 'none'")
        self.normalization = normalization
        self.outlier_threshold = outlier_threshold

    @staticmethod
    def _as_array(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X) -> "TabularPreprocessor":
        X = self._as_array(X)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.normalizer_ = None
        if self.normalization == "power":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True).fit(Xs)
                    _ = self.normalizer_.transform(Xs[:2])
            except Exception:  # degenerate columns: fall back to standard scaling only
                self.normalizer_ = None
        return self

    def transform(self, X) -> np.ndarray:
        X = self.scaler_.transform(self._as_array(X))
        if self.normalizer_ is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                X = self.normalizer_.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        t = self.outlier_threshold
        return np.clip(X, -t, t).astype(np.float32)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
