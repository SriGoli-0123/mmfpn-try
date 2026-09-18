# Single-pair protocol for the backbone ablation

One `(mgm_heads, cap_heads)` per dataset, 5 seeds, same pair on every backbone branch.
Pairs are the authors' best from `charts/*.csv`, except:

* `pad_ufes_20`: `(256, 2)` -- our own full-grid best on `main` (85.3 vs the authors' 85.2 at (256, 24)).
* any `cap_heads = 24` is replaced by `16`, because `cap_heads` must divide the backbone token width
  (192 TabPFN / 128 TabICL / 256 TabFM); this only affects `petfinder all` (32, 24) -> (32, 16).

Use with: `MMPFN_CONFIG_DIR=configs_best python -u run.py <dataset> [task]`
