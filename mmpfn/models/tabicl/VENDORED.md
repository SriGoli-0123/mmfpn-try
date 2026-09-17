# Vendored TabICL model code

Source: https://github.com/soda-inria/tabicl, `src/tabicl/_model/` + `src/tabicl/_torch_devices.py`
Version: 2.2.0 (git 0dbff3e, September 2026). License: BSD-3-Clause (see `LICENSE`).

Only import paths were changed (`from .._torch_devices` -> `from ._torch_devices`). The raw
`TabICL` module (not the sklearn wrapper) is what `mmpfn.models.mmtabicl` subclasses, in the
same way `mmpfn.models.mmpfn` forks the raw TabPFN-v2 `PerFeatureTransformer`.

Pretrained checkpoints are downloaded from the Hugging Face repo `jingang/TabICL`
(default: `tabicl-classifier-v2-20260212.ckpt`, ~110 MB) into `mmpfn/parameters/`.
