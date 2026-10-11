# Backbone ablation with TabDPT added: accuracy (%), 5 seeds

Updated 2026-10-10 18:10. TabPFN-v2 / TabICL / TabFM columns are the values from `image.png`, which I re-checked cell by cell against the logs on `run-logs` (all means agree; the only differences are 0.05 rounding cases and the PAD-UFES-20 TabPFN std, see the notes below). The TabDPT column is parsed from `mmpfn/logs/best_<dataset>_tabdpt.log` (branch `TabDPT-attempt`, `MMPFN_BACKBONE=tabdpt MMPFN_CONFIG_DIR=configs_best`, same (mgm, cap) pair as the other backbones). ± is the population std over the 5 seeds, as in the logs.

| Dataset | (mgm, cap) | TabPFN-v2 | TabICL | TabFM | **TabDPT** | Paper |
|---|---|---|---|---|---|---|
| PAD-UFES-20 | (256, 2) | 85.3 ± 1.6 | 85.4 ± 1.3 | 86.7 ± 1.3 | **85.1 ± 0.7** | 84.39 |
| CBIS-DDSM mass | (64, 32) | 66.9 ± 0.9 | 74.3 ± 2.1 | 73.7 ± 1.6 | **70.7 ± 2.3** | 74.53 |
| CBIS-DDSM calc | (256, 8) | 71.1 ± 2.7 | 70.6 ± 0.6 | 70.0 ± 3.1 | **71.7 ± 1.6** | 75.40 |
| Cloth | (128, 4) | 65.6 ± 0.7 | 65.9 ± 0.6 | 65.4 ± 0.4 | **66.5 ± 0.6** | 65.63 |
| Salary | (2, 2) | 46.1 ± 0.8 | 45.9 ± 1.9 | **44.2 ± 3.0** (uncapped rerun; capped was 45.5 ± 2.0 †) | **53.5 ± 2.1** | 46.17 |
| Airbnb | (128, 8) | 46.9 ± 0.8 | 47.6 ± 1.0 | **49.6 ± 1.5** (uncapped rerun; capped was 49.3 ± 1.4 †) | **47.5 ± 1.1** | 47.78 |
| PetFinder-I (image) | (256, 2) | 42.3 ± 0.7 | 42.5 ± 0.3 | 44.6 ± 0.4 | **44.4 ± 0.3** | 40.74 |
| PetFinder-T (text) | (128, 2) | 42.2 ± 0.4 | 44.8 ± 0.4 | 45.6 ± 0.6 | **45.7 ± 0.6** | 39.04 |
| PetFinder-A (both) | (32, 16) | 42.8 ± 0.6 | 43.8 ± 0.4 | 42.1 ± 0.6 ‡ | **45.3 ± 0.8** | 41.19 |

† TabFM capped at 2000 context rows (as in `image.png`).
‡ The log `best_petfinder_all_tabfm.log` says `backbone=PerFeatureTransformer`, i.e. this cell may actually be a TabPFN run, not TabFM. Needs checking before it is used.

## TabDPT per-seed accuracies (%)

| Dataset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean ± std |
|---|---|---|---|---|---|---|
| Cloth | 65.78 | 67.39 | 66.81 | 66.77 | 65.93 | 66.54 ± 0.60 |
| Salary | 54.06 | 50.94 | 57.19 | 53.23 | 52.08 | 53.50 ± 2.12 |
| Airbnb | 47.11 | 47.84 | 46.31 | 46.74 | 49.52 | 47.51 ± 1.13 |
| PAD-UFES-20 | 86.1 | 85.0 | 84.1 | 85.7 | 84.8 | 85.13 ± 0.68 |
| CBIS-DDSM mass | 73.8 | 71.7 | 72.0 | 68.3 | 67.7 | 70.69 ± 2.33 |
| CBIS-DDSM calc | 69.02 | 70.55 | 73.31 | 73.01 | 72.39 | 71.66 ± 1.63 |
| PetFinder-I (image) | 43.88 | 44.71 | 44.65 | 44.58 | 44.08 | 44.38 ± 0.34 |
| PetFinder-T (text) | 44.78 | 46.15 | 46.12 | 45.32 | 46.15 | 45.70 ± 0.56 |
| PetFinder-A (both) | 45.15 | 45.35 | 46.68 | 44.51 | 44.55 | 45.25 ± 0.79 |

## How many runs were made

| | count | detail |
|---|---|---|
| Main TabDPT runs finished (5 seeds each) | 9 | all nine table rows |
| TabFM reruns finished (uncapped) | 2 | salary (3,069 rows instead of 2,000): 45.94, 45.00, 47.40, 43.75, 38.75; airbnb (4,372 rows instead of 2,000): 49.09, 49.74, 49.31, 47.55, 52.08 |
| Runs in progress | 0 | - |
| Main TabDPT runs not possible | 0 | - |
| Aborted and discarded | 2 | first CBIS mass and calc attempt: both wrote the same checkpoint file and overwrote each other (logs kept in `logs/collided_ckpt/`) |
| Tabular-only controls (not requested, stopped) | 4 finished | cloth, salary, airbnb, PAD-UFES-20 |
| Smoke tests | 2 | tiny + real weights, CPU and A100 |

## Tabular-only controls (extra, not part of the main table)

| Dataset | TabDPT full | TabDPT tabular-only | modality contribution (pts) |
|---|---|---|---|
| Cloth | 66.54 ± 0.60 | 55.02 ± 0.43 | +11.5 |
| Salary | 53.50 ± 2.12 | 51.52 ± 0.55 | +2.0 |
| Airbnb | 47.51 ± 1.13 | 47.43 ± 1.08 | +0.1 |
| PAD-UFES-20 | 85.13 ± 0.68 | 83.39 ± 1.74 | +1.7 |

Not comparable to the earlier tabular-only logs on `run-logs` for other backbones (TabPFN-v2: PAD 82.7, CBIS mass 65.1, PetFinder-all 41.5; TabICL: CBIS mass 74.4), because those cover different dataset/backbone combinations. No tabular-only run exists in any branch for cloth, salary, airbnb or CBIS calc.

## Notes and caveats

* Salary: TabDPT is about 7 points above the other three backbones, but its own tabular-only control is already 51.5, so most of the gap comes from the tabular side. About 65-68 % of salary test rows have an exact tabular duplicate in the training split (seeds 0 and 1), which a retrieval-trained model can exploit.
* `image.png` PAD-UFES-20 TabPFN-v2 shows 85.3 ± 1.6. The log (`pad_ufes_20_missing.log`, (256, 2)) gives 85.30 ± 1.73. The same pair in `pad_ufes_20.log` gave 81.43 ± 4.86, so there are two runs of this cell that disagree.
* PAD-UFES-20 images come from the Kaggle mirror `mahdavi1202/skin-cancer` (2,298 images, all present). CBIS-DDSM has 10,237 JPEGs.
* Fusion hyper-parameters for TabDPT are untuned defaults (gate init 0.1, mean-reduce of the K tokens). Nothing was tuned on test accuracy.
