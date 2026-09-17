# Vendored TabFM model code

Source: https://github.com/google-research/tabfm, `tabfm/src/pytorch/model.py` (git fbb6655,
September 2026). Code license: Apache-2.0 (see `LICENSE`). `model.py` is copied verbatim.

The Hugging Face loader (`tabfm/src/pytorch/tabfm_v1_0_0.py`) was **not** vendored: it pulls in
`absl` and a `PyTorchModelHubMixin` subclass. `mmpfn.models.mmtabfm.loading` re-implements the
few lines that matter (download `classification/config.json` + `model.safetensors` from
`google/tabfm-1.0.0-pytorch`, build the module, load the state dict).

Released weights: `google/tabfm-1.0.0-pytorch` — classifier config
`embed_dim=256, row_num_cls=8 (ICL width 2048), icl_num_blocks=24`, **1.64B parameters**,
6.56 GB fp32 safetensors. The weights are under the `tabfm-non-commercial-v1.0` license
(non-commercial, non-production use only) — research ablations are fine, shipping is not.
