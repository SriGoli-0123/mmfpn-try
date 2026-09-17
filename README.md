# Multi-Modal PFN (Prior-data Fitted Network) — `TabFM-attempt` branch

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)

## Latest News
**MMPFN Paper Accepted to CVPR 2026** Our work on Multi-Modal Prior-data Fitted Networks has been accepted for presentation at CVPR 2026. Check back for the camera-ready version and supplementary materials. 

## Introduction 
> **MMPFN** is an extension of **TabPFN**, designed to handle **multimodal data** — combining tabular, image, and text inputs in a unified learning framework. While TabPFN has shown strong performance on purely tabular datasets, it lacks the ability to integrate heterogeneous modalities.
>
> Comprehensive experiments on datasets show that MMPFN **outperforms state-of-the-art baselines**, efficiently leveraging diverse data types to enhance predictive performance. This demonstrates the potential of extending **prior-data fitted networks** into the multimodal domain, offering a scalable and effective solution for heterogeneous data learning.

## Backbone-swap ablation: TabPFN-v2 → TabFM

This branch keeps everything in MMPFN fixed — the modality mixer (MGM / CAP / MoE), the
fine-tuning protocol, the Optuna grid and the evaluation loop — and replaces the tabular
backbone with **TabFM** (Google Research, `google/tabfm-1.0.0-pytorch`, 1.64B params).

| | MMPFN (`main`) | this branch |
|---|---|---|
| tabular backbone | TabPFN-v2 `PerFeatureTransformer` (7.2M) | TabFM v1.0.0 classifier (`mmpfn/models/tabfm`, vendored; 1.64B, 1.62B of it in the ICL stage) |
| token width the mixer emits | `ninp = 192` | `embed_dim = 256` |
| where modality tokens enter | `token_append` before the per-feature attention stack | after `cell_embedder`, before the first column stage — tokens then go through both across-row and both between-feature stages, like tabular cells |
| label conditioning of tokens | none | training-row tokens get TabFM's label embedding, like tabular cells (`add_label_to_tokens`) |
| model / classifier | `mmpfn.models.mmpfn` | `mmpfn.models.mmtabfm.{MMTabFM, MMTabFMClassifier}` |
| fine-tuning | `scripts_finetune_mm.fine_tune_mmpfn` | `scripts_finetune_tabfm.fine_tune_mmtabfm` (thin wrapper over `scripts_finetune_backbone`) |
| input preprocessing | none (TabPFN normalises internally) | StandardScaler → Yeo-Johnson → clip, fitted on the context rows (TabFM's own default) + `cat_mask` for categorical columns |
| `freeze_input=True` | freezes `encoder`, `y_encoder` | freezes `cell_embedder` |

**Memory.** TabFM is ~230× larger than TabPFN-v2, so the default protocol on this branch
differs from MMPFN in one deliberate way: `freeze_icl: true` (per-dataset config) keeps the
24-block ICL stage frozen in bf16 (3.2 GB) and trains only the mixer + column/row stages
(~20M + mixer) — this fits a 24 GB GPU for a few thousand context rows. Checkpoints then contain
only the trainable tensors and are overlaid on the released weights at load time. Set
`freeze_icl: false` for full fine-tuning like MMPFN does with TabPFN (fp32 weights + grads +
AdamW ≈ 26 GB before activations; plan for an 80 GB GPU).

Also worth reporting:

* `cap_heads` must divide 256, so `12` and `24` were dropped from `cap_heads_list` in `configs/*.yaml`.
* `features_per_group` is a TabPFN-only hyper-parameter and is ignored.
* The released weights (6.56 GB `model.safetensors`) are downloaded into
  `mmpfn/parameters/tabfm-1.0.0-pytorch/` on first use. They are licensed
  `tabfm-non-commercial-v1.0` (research use only); no technical report exists for TabFM.
* `python smoke_mmtabfm.py` (CPU, no data, no download) checks the wiring;
  `--released` additionally verifies the released config/state-dict keys against the HF header.

Everything else in `run.py` is untouched: `python run.py pad_ufes_20` etc. work as before and write
`checkpoints/finetuned_mmtabfm_<dataset>.ckpt`.

## Set-up

Conda Environment (`environment-lean.yaml` holds only what the code imports; the original
`environment.yaml` is a full machine export whose pins no longer resolve — `seqeval`, etc.)
```
conda env create -f environment-lean.yaml
```

Install
```
python setup.py develop
```

Place the checkpoint file and dataset in their respective locations, then update the model_path as shown below:

```
ln -s /path/to/model/params # symlink parameter
ln -s /path/to/data # symlink data

model_path = Path(__file__).parent/ "parameters" / "tabpfn-v2-classifier.ckpt"
```

## Usage


To reproduce the experimental results, you can run `run_pad_ufes_20_mmpfn.py`, which uses Optuna to explore all hyperparameters.  
```
python run_pad_ufes_20_mmpfn.py
```

To view the results obtained with the optimized parameters, open and execute the notebook file `run_pad_ufes_20_mmpfn.ipynb`.


## License
This project follows the original TabPFN license policy(Apache 2.0 with additional attribution requirement): [here](https://priorlabs.ai/tabpfn-license/)
