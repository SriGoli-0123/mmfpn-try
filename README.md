# Multi-Modal PFN (Prior-data Fitted Network) — `TabICL-attempt` branch

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)

## Latest News
**MMPFN Paper Accepted to CVPR 2026** Our work on Multi-Modal Prior-data Fitted Networks has been accepted for presentation at CVPR 2026. Check back for the camera-ready version and supplementary materials. 

## Introduction 
> **MMPFN** is an extension of **TabPFN**, designed to handle **multimodal data** — combining tabular, image, and text inputs in a unified learning framework. While TabPFN has shown strong performance on purely tabular datasets, it lacks the ability to integrate heterogeneous modalities.
>
> Comprehensive experiments on datasets show that MMPFN **outperforms state-of-the-art baselines**, efficiently leveraging diverse data types to enhance predictive performance. This demonstrates the potential of extending **prior-data fitted networks** into the multimodal domain, offering a scalable and effective solution for heterogeneous data learning.

## Backbone-swap ablation: TabPFN-v2 → TabICL

This branch keeps everything in MMPFN fixed — the modality mixer (MGM / CAP / MoE), the
fine-tuning protocol, the Optuna grid and the evaluation loop — and replaces the tabular
backbone with **TabICL** (`jingang/TabICL`, `tabicl-classifier-v2-20260212.ckpt`, ~28M params).

| | MMPFN (`main`) | this branch |
|---|---|---|
| tabular backbone | TabPFN-v2 `PerFeatureTransformer` (7.2M) | TabICL v2 (`mmpfn/models/tabicl`, vendored) |
| token width the mixer emits | `ninp = 192` | `embed_dim = 128` |
| where modality tokens enter | `token_append` before the per-feature attention stack | between `col_embedder` and `row_interactor` |
| cross-sample attention seen by the tokens | every layer, token level | on the pooled row vector (ICL stage) |
| model / classifier | `mmpfn.models.mmpfn` | `mmpfn.models.mmtabicl.{MMTabICL, MMTabICLClassifier}` |
| fine-tuning | `scripts_finetune_mm.fine_tune_mmpfn` | `scripts_finetune_tabicl.fine_tune_mmtabicl` (thin wrapper over `scripts_finetune_backbone`) |
| input preprocessing | none (TabPFN normalises internally) | StandardScaler → Yeo-Johnson → clip, fitted on the context rows (TabICL's own default) |
| `freeze_input=True` | freezes `encoder`, `y_encoder` | freezes `col_embedder` |

Things that changed for the ablation and are worth reporting:

* `cap_heads` must divide 128, so `12` and `24` were dropped from `cap_heads_list` in `configs/*.yaml`.
* `features_per_group` is a TabPFN-only hyper-parameter and is ignored.
* The pretrained checkpoint is downloaded into `mmpfn/parameters/` on first use.
* `python smoke_mmtabicl.py` (CPU, no data) checks the wiring; `--pretrained` uses the real checkpoint.

Everything else in `run.py` is untouched: `python run.py pad_ufes_20` etc. work as before and write
`checkpoints/finetuned_mmtabicl_<dataset>.ckpt`.

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
