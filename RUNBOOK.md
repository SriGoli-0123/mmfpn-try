# Runbook (NVIDIA A100, Linux)

Run everything in order, from the paths shown. Steps 1–4 are one-time.

## 1. Code + environment

```bash
cd /scratch/sgoli125/mmfpn-try && git pull
conda env remove -n mmpfn -y 2>/dev/null; conda env create -f environment-lean.yaml
conda activate mmpfn && python setup.py develop
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `2.4.1+cu124 True NVIDIA A100...`

## 2. Weights (into `mmpfn/parameters/`)

```bash
cd mmpfn && mkdir -p parameters checkpoints logs
huggingface-cli download Prior-Labs/TabPFN-v2-clf tabpfn-v2-classifier.ckpt --local-dir parameters
wget -P parameters https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
huggingface-cli download google/electra-base-discriminator
huggingface-cli download jingang/TabICL tabicl-classifier-v2-20260212.ckpt --local-dir parameters
huggingface-cli download google/tabfm-1.0.0-pytorch --include "classification/*" "config.json" --local-dir parameters/tabfm-1.0.0-pytorch
```
(The last three are optional pre-fetches; the branches download them on first use.)

## 3. Data path

`run.py` reads data from `~/workspace/research/MultiModalPFN/mmpfn/data/<dataset>`:

```bash
mkdir -p ~/workspace/research && ln -sfn /scratch/sgoli125/mmfpn-try ~/workspace/research/MultiModalPFN
```

## 4. Datasets (into `mmpfn/data/<dataset>/`; Kaggle token at `~/.kaggle/kaggle.json`)

```bash
pip install kaggle && chmod 600 ~/.kaggle/kaggle.json
```
```bash
# pad_ufes_20  (Mendeley; images are 3 zips -> flat imgs/)
mkdir -p data/pad_ufes_20 && cd data/pad_ufes_20 && wget -O pad.zip "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip" && unzip -q pad.zip && mkdir -p imgs && for z in images/imgs_part_*.zip; do unzip -q -j "$z" -d imgs; done && rm -rf pad.zip images && cd ../..
# cbis_ddsm (~6 GB)
mkdir -p data/cbis_ddsm && kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p data/cbis_ddsm --unzip
# petfinder (accept competition rules on kaggle.com first)
mkdir -p data/petfinder-adoption-prediction && kaggle competitions download -c petfinder-adoption-prediction -p data/petfinder-adoption-prediction && cd data/petfinder-adoption-prediction && unzip -q petfinder-adoption-prediction.zip && rm petfinder-adoption-prediction.zip && cd ../..
# cloth
mkdir -p data/cloth && kaggle datasets download -d nicapotato/womens-ecommerce-clothing-reviews -p data/cloth --unzip
# airbnb
mkdir -p data/airbnb && kaggle datasets download -d tylerx/melbourne-airbnb-open-data -p data/airbnb --unzip
# salary (rename if the file is Final_Train_Dataset.csv)
mkdir -p data/salary && kaggle datasets download -d ankitkalauni/predict-the-data-scientists-salary-in-india -p data/salary --unzip
[ -f data/salary/Final_Train_Dataset.csv ] && mv data/salary/Final_Train_Dataset.csv data/salary/train.csv
```
Check every path the loaders read:
```bash
ls data/pad_ufes_20/metadata.csv data/cbis_ddsm/csv data/petfinder-adoption-prediction/train/train.csv data/cloth/*.csv data/airbnb/cleansed_listings_dec18.csv data/salary/train.csv parameters/tabpfn-v2-classifier.ckpt parameters/dinov2_vitb14_pretrain.pth
```

## 5. Run — `main` (MMPFN, TabPFN-v2 backbone)

Always from `mmpfn/`. Each run = full Optuna grid × 5 seeds; prints `Best parameters` / `Best value`.

```bash
git checkout main
CUDA_VISIBLE_DEVICES=0 python run.py pad_ufes_20 >> logs/pad_ufes_20.log
CUDA_VISIBLE_DEVICES=0 python run.py cbis_ddsm mass >> logs/cbis_ddsm_mass.log
CUDA_VISIBLE_DEVICES=0 python run.py cbis_ddsm calc >> logs/cbis_ddsm_calc.log
CUDA_VISIBLE_DEVICES=0 python run.py petfinder-adoption-prediction image >> logs/petfinder-image.log
CUDA_VISIBLE_DEVICES=0 python run.py petfinder-adoption-prediction text >> logs/petfinder-text.log
CUDA_VISIBLE_DEVICES=0 python run.py petfinder-adoption-prediction all >> logs/petfinder-all.log
CUDA_VISIBLE_DEVICES=0 python run.py cloth >> logs/cloth.log
CUDA_VISIBLE_DEVICES=0 python run.py salary >> logs/salary.log
CUDA_VISIBLE_DEVICES=0 python run.py airbnb >> logs/airbnb.log
```
Embeddings are cached to `embeddings/<dataset>/*.pt` on the first run and reused by the other branches.

## 6. Run — `TabICL-attempt`

```bash
git checkout TabICL-attempt && python smoke_mmtabicl.py --pretrained
CUDA_VISIBLE_DEVICES=0 python run.py pad_ufes_20 >> logs/pad_ufes_20_tabicl.log
```
(same dataset arguments as step 5)

## 7. Run — `TabFM-attempt`

```bash
git checkout TabFM-attempt && python smoke_mmtabfm.py --released
CUDA_VISIBLE_DEVICES=0 python run.py pad_ufes_20 >> logs/pad_ufes_20_tabfm.log
```
Default `freeze_icl: true` in `configs/*.yaml` fits a 40 GB A100; `freeze_icl: false` (full fine-tune) needs 80 GB.

Checkpoints land in `checkpoints/finetuned_{mmpfn,mmtabicl,mmtabfm}_<dataset>.ckpt`, so branches don't collide.
