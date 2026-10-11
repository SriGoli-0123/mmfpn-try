# VOLT on MMPFN: runbook (branch `volt-mmpfn`)

Jointly learned **image + text backdoor trigger** for MMPFN (TabPFN-v2 + modality projector), following VOLT
(spectral image trigger, Eq. 6-10) and BAPLe (target-label ablation, Table 2). Image trigger: VOLT's low-frequency
spectral trigger, a full-size 3x336x336 perturbation from an 8x8 spectrum. Text trigger: 8 learnable context vectors
fed through frozen ELECTRA, as **CoOp / CoCoOp / ProDA / KgCoOp**. Both are trained together, with the in-context
rows clean. Datasets with tabular + image + text: **PetFinder** (`all` task) and **Fashion Product Images**.

Everything below is interactive (tmux + `tee`), no scheduler scripts. One run = 5 seeds (or 3 for ablations) and
writes `mmpfn/results/volt/<dataset>_<strategy>_<mode>_t<target>[_<tag>].json`.

Paths use `$USER`; change `REPO` if you cloned elsewhere.

```bash
export REPO=/scratch/$USER/mmfpn-try          # where this repo lives
```

---------------------------------------------------------------------------------------------------------------

## 1. Code and environment (one time)

```bash
cd $REPO && git fetch origin && git checkout volt-mmpfn && git pull
conda env create -f environment-lean.yaml     # skip if the `mmpfn` env already exists
conda activate mmpfn && python setup.py develop
pip install scikit-image kaggle               # SSIM metric; Kaggle CLI for the Fashion data
python -c "import torch, transformers, skimage, torchvision; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`kaggle` needs credentials: `~/.kaggle/kaggle.json` (or `~/.kaggle/access_token` for the 2.x CLI), `chmod 600`.

## 2. Weights (one time, into `mmpfn/parameters/`)

```bash
cd $REPO/mmpfn && mkdir -p parameters logs/volt checkpoints results
huggingface-cli download Prior-Labs/TabPFN-v2-clf tabpfn-v2-classifier.ckpt --local-dir parameters      # ~29 MB
wget -P parameters https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth       # ~346 MB
# ELECTRA (google/electra-base-discriminator) downloads itself from Hugging Face on the first run
```

## 3. Data (one time; everything lives under `mmpfn/data/`, which is git-ignored)

The code finds data in `mmpfn/data/<name>`, then `~/workspace/research/MultiModalPFN/mmpfn/data/<name>`, or in
`$MMPFN_DATA_ROOT/<name>` if you set that variable.

### 3a. PetFinder (`petfinder-adoption-prediction`)

Needs the **original Kaggle** `train/train.csv` and the training photos named `<PetID>-<n>.jpg`:

```bash
cd $REPO
D=mmpfn/data/petfinder-adoption-prediction && mkdir -p $D/train $D/train_images
# CSV: from the repo's data branch (do not commit it; the checkout stages it, so unstage it right away)
git fetch origin petfinder-data
git checkout origin/petfinder-data -- $D/train/train.csv && git restore --staged $D/train/train.csv
# photos: public mirror (about 2 GB), keep only the training photos
wget -O /tmp/petfinder_kaggle.zip https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip
unzip -q -j -n /tmp/petfinder_kaggle.zip 'petfinder_processed/train_images/*' -d $D/train_images
# check: 14,993 CSV rows, 14,993 pets with a -1.jpg
python - <<'EOF'
import os, pandas as pd
D = "mmpfn/data/petfinder-adoption-prediction"
df = pd.read_csv(f"{D}/train/train.csv"); imgs = set(os.listdir(f"{D}/train_images"))
print(df.shape, sum(f"{p}-1.jpg" in imgs for p in df.PetID))
EOF
```

### 3b. Fashion Product Images (Kaggle `paramaggarwal/fashion-product-images-small`)

```bash
cd $REPO/mmpfn && mkdir -p data/fashion_product_images
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/fashion_product_images --unzip
ls data/fashion_product_images            # expect: images/  styles.csv  (a myntradataset/ duplicate is harmless)
```

Run 15,000 rows (seeded, stratified by `season`: Fall, Spring, Summer, Winter), text = product name, target = `season`.

## 4. Caches and clean warm-ups (one time per dataset; do these BEFORE the parallel runs)

Always run from `mmpfn/` (some paths are relative to it). `GPU=0` below can be any free GPU.

```bash
cd $REPO/mmpfn && export PYTHONPATH=$REPO       # not needed after `python setup.py develop`
# 4a. image caches (uint8, about 5 GB per dataset, shared through the page cache) and the clean [image, text] embeddings
CUDA_VISIBLE_DEVICES=0 python -c "from mmpfn.volt.data import load_petfinder, load_fashion; load_petfinder(); load_fashion()"
# 4b. clean MMPFN warm-ups for the 5 seeds (cached under checkpoints/context_warmup/, reused by every later run)
for ds in petfinder fashion; do
  CUDA_VISIBLE_DEVICES=0 MMPFN_DATASET=$ds MMPFN_WARMUP_ONLY=1 MMPFN_SEEDS=5 python -u run_volt_joint.py 2>&1 | tee logs/volt/warmup_$ds.log
done
```

Skipping 4b is safe for a single run, but parallel runs would then all build the same warm-up at once.

## 5. Sanity checks

```bash
cd $REPO && python -m pytest tests/test_volt.py -q                    # 8 CPU tests: triggers, 4 prompt types, transforms, cA/bA
cd $REPO/mmpfn && CUDA_VISIBLE_DEVICES=0 MMPFN_DATASET=petfinder MMPFN_STRATEGY=coop MMPFN_STEPS=4 MMPFN_WARMUP_STEPS=5 \
  MMPFN_SEEDS=1 MMPFN_EVAL_CTX=512 MMPFN_TAG=smoke MMPFN_OUT=results/volt_smoke python -u run_volt_joint.py 2>&1 | tee logs/volt/smoke.log
```

The smoke run (4 attack steps) is expected to show no attack effect; it only checks that every code path runs.

## 6. One run

```bash
cd $REPO/mmpfn
CUDA_VISIBLE_DEVICES=0 MMPFN_DATASET=petfinder MMPFN_STRATEGY=coop MMPFN_TARGET_CLASS=0 MMPFN_ROBUST=1 MMPFN_CTX_SWEEP=1 \
  python -u run_volt_joint.py 2>&1 | tee logs/volt/pet_coop_t0.log
```

Per seed it prints `val_cA / val_bA / val_FTR` every 10 steps and then one line with the test result:
`cA=.. bA=.. FTR=.. clean_model_cA=.. | img_only_bA=.. txt_only_bA=.. random_bA=.. null_bA=.. PSNR=..`.
About 35 minutes per run (5 seeds, robustness and context sweep on) on one A100; about 15 GB GPU memory.

## 7. Main experiments (BAPLe Table 2(a) style target sweep, all four prompt types)

Use one tmux window per GPU. Define this helper in each window (it sets the GPU, writes the log, and keeps going):

```bash
cd $REPO/mmpfn && export PYTHONPATH=$REPO
run() { local name=$1; shift; env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MMPFN_OUT=results/volt "$@" \
        python -u run_volt_joint.py 2>&1 | tee logs/volt/$name.log; }
```

**PetFinder, 5 target labels (0-4), one prompt type per GPU lane** (4 GPUs; the robustness and context-size
sweeps are run for CoOp on every target and for the other prompt types on target 0):

```bash
# lane A (GPU 0): CoOp
export GPU=0; for t in 0 1 2 3 4; do run pet_coop_t$t   MMPFN_DATASET=petfinder MMPFN_STRATEGY=coop   MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=1 MMPFN_CTX_SWEEP=1; done
# lane B (GPU 1): CoCoOp
export GPU=1; for t in 0 1 2 3 4; do run pet_cocoop_t$t MMPFN_DATASET=petfinder MMPFN_STRATEGY=cocoop MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0); done
# lane C (GPU 2): ProDA
export GPU=2; for t in 0 1 2 3 4; do run pet_proda_t$t  MMPFN_DATASET=petfinder MMPFN_STRATEGY=proda  MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0); done
# lane D (GPU 3): KgCoOp
export GPU=3; for t in 0 1 2 3 4; do run pet_kgcoop_t$t MMPFN_DATASET=petfinder MMPFN_STRATEGY=kgcoop MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0); done
```

**Fashion, 4 target labels (0-3: Fall, Spring, Summer, Winter)**, same lanes:

```bash
export GPU=0; for t in 0 1 2 3; do run fashion_coop_t$t   MMPFN_DATASET=fashion MMPFN_STRATEGY=coop   MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0) MMPFN_CTX_SWEEP=$([ $t = 0 ] && echo 1 || echo 0); done
export GPU=1; for t in 0 1 2 3; do run fashion_cocoop_t$t MMPFN_DATASET=fashion MMPFN_STRATEGY=cocoop MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0); done
export GPU=2; for t in 0 1 2 3; do run fashion_proda_t$t  MMPFN_DATASET=fashion MMPFN_STRATEGY=proda  MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0); done
export GPU=3; for t in 0 1 2 3; do run fashion_kgcoop_t$t MMPFN_DATASET=fashion MMPFN_STRATEGY=kgcoop MMPFN_TARGET_CLASS=$t MMPFN_ROBUST=$([ $t = 0 ] && echo 1 || echo 0); done
```

## 8. Ablations (PetFinder, CoOp, target 0, 3 seeds; `tables.py` groups them by value)

Set once: `B="MMPFN_DATASET=petfinder MMPFN_STRATEGY=coop MMPFN_TARGET_CLASS=0 MMPFN_ROBUST=0 MMPFN_SEEDS=3"`.
Every run needs a distinct `MMPFN_TAG`, otherwise it overwrites another run's JSON. Split these loops over free GPUs.

```bash
export GPU=0; B="MMPFN_DATASET=petfinder MMPFN_STRATEGY=coop MMPFN_TARGET_CLASS=0 MMPFN_ROBUST=0 MMPFN_SEEDS=3"
# which triggers are trained (BAPLe-style decomposition): image only / text only
run pet_mode_image $B MMPFN_MODE=image MMPFN_TAG=modeimage
run pet_mode_text  $B MMPFN_MODE=text  MMPFN_TAG=modetext
# image trigger design (VOLT Fig. 2 / Fig. 5): high-frequency band, dense additive (tanh), intensity-gated spectral
for k in spectral_hf additive gated; do run pet_trig_$k $B MMPFN_IMG_TRIGGER=$k MMPFN_TAG=trig$k; done
# trigger budget eps (VOLT Fig. 6; BAPLe Table 2(c)); 0.00784/0.01569/0.03137 = 2/255, 4/255, 8/255
for e in 0.00784 0.01569 0.03137 0.1 0.2; do run pet_eps$e $B MMPFN_VOLT_EPS=$e MMPFN_TAG=eps$e; done
# spectral band size k, backdoor weight lambda (Eq. 10), text context length M
for b in 4 16 32;   do run pet_band$b $B MMPFN_TRIGGER_BAND=$b MMPFN_TAG=band$b; done
for l in 0.5 2 4;   do run pet_lam$l  $B MMPFN_VOLT_LAMBDA=$l  MMPFN_TAG=lam$l;  done
for m in 1 2 4 16;  do run pet_ctx$m  $B MMPFN_CTX_LEN=$m      MMPFN_TAG=ctx$m;  done
# poison fraction of each query batch (BAPLe Table 2(d)) and shots per class (VOLT Fig. 8)
for p in 0.03 0.1 0.2 0.3 0.5; do run pet_poison$p $B MMPFN_POISON_FRAC=$p MMPFN_TAG=poison$p; done
for k in 1 2 4 8 16 32;        do run pet_shots$k  $B MMPFN_SHOTS=$k       MMPFN_TAG=shots$k;  done
# only the modality projector trains (TabPFN backbone frozen), closest to "prompt + trigger only"
run pet_scope_projector $B MMPFN_SCOPE=projector MMPFN_TAG=scopeprojector
# can the image trigger work on its own? bigger budget / band / longer / faster
for e in 0.1 0.2; do run pet_imgonly_eps$e $B MMPFN_MODE=image MMPFN_VOLT_EPS=$e MMPFN_TAG=imgonly_eps$e; done
run pet_imgonly_strong $B MMPFN_MODE=image MMPFN_VOLT_EPS=0.2 MMPFN_TRIGGER_BAND=16 MMPFN_STEPS=300 MMPFN_TRIGGER_LR=0.05 MMPFN_TAG=imgonly_strong
```

Fashion ablations use the same pattern with `MMPFN_DATASET=fashion` (suggested: modes image/text, eps 0.1/0.2,
`spectral_hf`, `additive`, lambda 0.5/2, and `imgonly_strong`).

## 9. Tables

```bash
cd $REPO/mmpfn && python -m mmpfn.volt.tables results/volt > ../RESULTS_volt.md
```

Writes: (1) cA / bA per target label for every prompt type with the "Clean (no attack)" row (BAPLe Table 2(a)),
(2) trigger decomposition and controls (image-only, text-only, random trigger, null pipeline, trigger on the clean
model, zero-modality), (3) fidelity (PSNR, MSE, L_inf, SSIM, text-embedding shift, snapped-to-words bA),
(4) robustness to blur / resampling / renormalisation / noise / JPEG / word dropout / truncation,
(5) the one-factor ablations, (6) clean-context-size sensitivity, (7) bA per source class.

## 10. Settings reference (environment variables; defaults in brackets)

| variable | meaning |
|---|---|
| `MMPFN_DATASET` [petfinder] | `petfinder` or `fashion` |
| `MMPFN_STRATEGY` [coop] | text prompt: `coop`, `cocoop`, `proda`, `kgcoop` |
| `MMPFN_MODE` [joint] | triggers trained: `joint`, `image`, `text` |
| `MMPFN_IMG_TRIGGER` [spectral] | `spectral` (VOLT), `spectral_hf`, `additive`, `gated` |
| `MMPFN_TARGET_CLASS` [0] | attacker's label |
| `MMPFN_VOLT_EPS` [0.05] / `MMPFN_TRIGGER_BAND` [8] | l_inf budget on [0,1] pixels / spectrum size k (k x k) |
| `MMPFN_CTX_LEN` [8] | number of text context vectors M |
| `MMPFN_VOLT_LAMBDA` [1.0] | backdoor-loss weight (Eq. 10) |
| `MMPFN_STEPS` [100] / `MMPFN_WARMUP_STEPS` [100] | attack steps / clean warm-up steps |
| `MMPFN_LR` [1e-4], `MMPFN_TRIGGER_LR` [1e-2], `MMPFN_PROMPT_LR` [1e-2] | learning rates: MMPFN / image trigger / text prompt |
| `MMPFN_QBATCH` [32], `MMPFN_CTX_CAP` [1024] | query rows per step, clean context rows per training step |
| `MMPFN_POISON_FRAC` [1.0] | fraction of each batch that also gets a triggered copy |
| `MMPFN_SHOTS` [0 = all] | k training rows per class (few-shot) |
| `MMPFN_SCOPE` [full] | `full` or `projector` (only the MGM/CAP projector trains) |
| `MMPFN_MAX_CA_DROP` [0.05] | clean-accuracy budget used when choosing the best checkpoint on validation data |
| `MMPFN_EVAL_CTX` [4096] | clean context rows at test time |
| `MMPFN_ROBUST` [1], `MMPFN_CTX_SWEEP` [0] | robustness table / context-size sweep |
| `MMPFN_SEEDS` [5], `MMPFN_TAG` [""], `MMPFN_OUT` [results/volt] | seeds, name suffix, output folder |
| `MMPFN_KG_LAMBDA` [8], `MMPFN_PRODA_N` [4] | KgCoOp regulariser weight, ProDA number of prompts |
| `MMPFN_WARMUP_ONLY` [0] | build the clean warm-ups and exit |

## 11. Notes

* **Protocol.** In-context rows are always clean (clean embeddings, true labels). Only the query carries the trigger,
  in training and at test, so bA is the "weights-only backdoor" of VOLT/BAPLe. cA is measured on the whole test
  split, bA on the non-target test rows (VOLT Eq. 12). "Clean" = the pre-attack model on the same split.
* **Checkpoint choice** uses only a validation slice of the training split (never test labels). A checkpoint is
  eligible only if its validation cA is within `MMPFN_MAX_CA_DROP` of the clean model; if none is, the untrained
  trigger is restored (that seed then shows bA about 0).
* **Controls** printed for every seed: `random_bA` (random trigger of the same size), `null_bA` (clean inputs
  re-encoded through the trigger pipeline), and the trigger applied to the clean (pre-attack) model.
* **Text length.** ELECTRA has 512 positions; the 8 context vectors need room, so descriptions longer than 504
  tokens (about 0.8 % on PetFinder) lose their last tokens on the triggered path only (printed at the start).
* **Variability.** GPU kernels are not deterministic: repeating a run with the same seed can move cA by a few points.
* **Parallel runs.** Warm-ups (step 4b) must exist before parallel runs. Checkpoint and result names include the
  dataset, strategy, target and tag, so different settings never collide. Check `nvidia-smi` first on a shared
  machine: a run needs about 15 GB.
* **Not committed:** `mmpfn/data/`, caches, weights, `mmpfn/results/`, logs.
