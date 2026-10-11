# Vendored TabDPT model code

Source: PyPI package `tabdpt` 1.3.1 (repo: https://github.com/layer6ai-labs/TabDPT-inference), files
`tabdpt/model.py` and `tabdpt/utils.py`, **byte-for-byte unedited** (they use package-relative
imports only, so they work as-is inside this package). License: Apache-2.0 (see `LICENSE`).
`utils.py` imports `faiss` at module level, so `faiss-cpu` is a dependency.

Only the raw `TabDPTModel` module is used; `mmpfn.models.mmtabdpt` subclasses it. The sklearn-style
wrappers (`estimator.py`, `classifier.py`) are not vendored -- `mmpfn.models.mmtabdpt.loading`
re-implements the (small) checkpoint reader and `mmpfn.models.mmtabdpt.classifier` the predict path.

Pretrained weights: Hugging Face `Layer6/TabDPT`, file `tabdpt1_3.safetensors` (252 MB), downloaded
on demand into `mmpfn/parameters/`. The model config is stored as JSON in the safetensors metadata
key `cfg`. Values read from that file (not assumed): emsize 512, nlayers 32, nhead 8, ff_dim 512,
y_encoder_dim 128, max_num_features 128, max_num_classes 16, n_thinking_rows 64, 63.0M parameters.
