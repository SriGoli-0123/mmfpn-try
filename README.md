# Branch map and experiment logs

This branch (`run-logs`) holds the raw run logs under `mmpfn/logs_shared/`. It is regenerated from `main`
on every log push, so anything kept here must be re-added by the push command (see below).

## Branches

| branch | head | what it holds |
|---|---|---|
| `main` | `00617fb` | clean MMPFN clone, plus the `MMPFN_TABULAR_ONLY` control switch and a per-seed backbone identity line |
| `TabICL-attempt` | `e1ff372` | TabICL backbone in place of TabPFN-v2 |
| `TabFM-attempt` | `a6510f3` | TabFM backbone in place of TabPFN-v2 |
| `backdoor-checkerboard` | `78fd275` | fixed corner-patch trigger, opt-in via `MMPFN_BACKDOOR` |
| `backdoor-baple` | `7c167f2` | BAPLe-style dense trigger learned jointly with the projector |
| `backdoor-baple-v2` | `ad2d9b0` | clean-context trigger objective, token alignment, context-dose diagnostic |
| `backdoor-volt` | `3067f3c` | VOLT spectral trigger, lambda weighting, paired poisoning, mini-batched trigger steps, false-trigger-rate metric, `cap_heads` sweep, image/text/both modality switch |

`backdoor-volt` is cumulative: it contains everything from the earlier backdoor branches, so it is the one to
work from. Every branch is `main` plus opt-in switches - with the environment variables unset, each reproduces
`main` exactly, and MMPFN's model files are untouched throughout.

## Regenerating this branch

The log push resets `run-logs` from `main`, so this file has to be carried over explicitly:

```bash
cd /scratch/sgoli125/mmfpn-try/mmpfn && mkdir -p logs_shared && for f in logs/*.log; do tr '\r' '\n' < "$f" | grep -v "Fine-tuning Steps" > "logs_shared/$(basename "$f")"; done
```
```bash
cd /scratch/sgoli125/mmfpn-try && git stash -q; git fetch -q origin run-logs && git checkout -q -B run-logs main && git checkout origin/run-logs -- README.md 2>/dev/null; git add -f mmpfn/logs_shared README.md && git commit -qm "run logs $(date +%F_%H%M)" && git push -f origin run-logs && git checkout -q backdoor-volt; git stash pop -q 2>/dev/null; cd mmpfn
```

---

# Multi-Modal PFN (Prior-data Fitted Network)

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)

## Latest News
**MMPFN Paper Accepted to CVPR 2026** Our work on Multi-Modal Prior-data Fitted Networks has been accepted for presentation at CVPR 2026. Check back for the camera-ready version and supplementary materials. 

## Introduction 
> **MMPFN** is an extension of **TabPFN**, designed to handle **multimodal data** — combining tabular, image, and text inputs in a unified learning framework. While TabPFN has shown strong performance on purely tabular datasets, it lacks the ability to integrate heterogeneous modalities.
>
> Comprehensive experiments on datasets show that MMPFN **outperforms state-of-the-art baselines**, efficiently leveraging diverse data types to enhance predictive performance. This demonstrates the potential of extending **prior-data fitted networks** into the multimodal domain, offering a scalable and effective solution for heterogeneous data learning.

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
