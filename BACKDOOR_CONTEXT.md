# MMPFN context/trigger experiments

These experiments adapt ideas from three sources rather than reproducing any one paper exactly:

- **BAPLe:** jointly learn a modality-side trigger and the downstream trainable policy.
- **VOLT:** parameterize the image trigger in a compact low-frequency spectrum.
- **BEAT:** compare ordinary supervised training with preference-style trigger discrimination.

MMPFN's model implementation is unchanged. Its image/text encoders stay frozen, while the modality projector,
TabPFN backbone, decoder, and the selected trigger are trainable.

## Implemented cases

| Runner | Clean path | Triggered path | Learnable trigger |
|---|---|---|---|
| `mmpfn/run_text_context.py` | `(S,T) -> y_clean` | `(S,T+C) -> y_target` | External continuous ELECTRA input embeddings `C`; **Cloth by default** |
| `mmpfn/run_image_context.py` | `(S,V) -> y_clean` | `(S,V+delta) -> y_target` | 2D low-frequency spectral image perturbation `delta`; **PAD-UFES-20 by default** |

`C` is not natural language and is not snapped to words during the main experiment. The image trigger obeys an
L-infinity budget, and the runner reports MSE and PSNR in addition to attack metrics.

Both runners support:

- `MMPFN_LOSS=sft`: clean-to-true and triggered-to-target cross-entropy.
- `MMPFN_LOSS=dpo`: clean prefers true over target; triggered prefers target over true, relative to the frozen
  initial policy and trigger.
- `MMPFN_LOSS=combined`: SFT plus `MMPFN_CTL_LAMBDA` times the reference-anchored preference loss.

Checkpoint selection first requires validation cA to remain within `MMPFN_MAX_CA_DROP` of the untouched
step-zero policy (default 0.05), then maximizes selective trigger effect `ASR-FTR`. The untouched policy is
eligible. This prevents a high-ASR model with unacceptable utility loss from winning.

Both runners also zero every modality embedding at final evaluation and report `Mean zero-modality cA` and
`Mean modality gain`. This is an in-model ablation rather than a separately trained tabular-only baseline. It
checks that the selected dataset/model actually uses the modality before interpreting its trigger results.

Before applying any trigger loss, each runner now performs a clean-only MMPFN warmup (100 steps by default).
This makes the checkpoint's clean-accuracy budget relative to a normally trained multimodal policy rather than
an under-trained starting model. Warmups are cached per dataset, modality, seed, and training configuration in
`checkpoints/context_warmup/`, so the remaining six loss conditions reuse the first condition's clean model.
Set `MMPFN_WARMUP_STEPS=0` only for an explicit no-warmup ablation.

The previous PetFinder defaults remain available by setting
`MMPFN_DATASET=petfinder-adoption-prediction`. The upgraded defaults are deliberately stronger tests:

- **image: PAD-UFES-20**, where the published/reproduced multimodal gain is clearer than PetFinder image;
- **text: Cloth**, where text provides the largest reported clean improvement among the text datasets.

## Interactive A100 runs

Run from `mmpfn/` inside the existing interactive allocation and `tmux` session.

Quick smoke tests:

```bash
MMPFN_LOSS=sft MMPFN_WARMUP_STEPS=2 MMPFN_MAX_STEPS=2 MMPFN_VAL_EVERY=1 \
  CUDA_VISIBLE_DEVICES=0 python -u run_image_context.py 2>&1 | tee logs/pad_image_smoke.log
MMPFN_LOSS=sft MMPFN_WARMUP_STEPS=2 MMPFN_MAX_STEPS=2 MMPFN_VAL_EVERY=1 \
  CUDA_VISIBLE_DEVICES=0 python -u run_text_context.py 2>&1 | tee logs/cloth_text_smoke.log
```

Full seven image conditions on PAD-UFES-20 (each command performs the fixed five-seed protocol):

```bash
MMPFN_LOSS=sft CUDA_VISIBLE_DEVICES=0 python -u run_image_context.py 2>&1 | tee logs/pad_image_sft.log
MMPFN_LOSS=dpo CUDA_VISIBLE_DEVICES=0 python -u run_image_context.py 2>&1 | tee logs/pad_image_dpo.log
for lam in 0.2 0.4 0.6 0.8 1.0; do
  MMPFN_LOSS=combined MMPFN_CTL_LAMBDA="$lam" CUDA_VISIBLE_DEVICES=0 \
    python -u run_image_context.py 2>&1 | tee "logs/pad_image_combined_lam${lam}.log"
done
```

The image defaults are `epsilon=8/255`, an `8x8` spectrum, 100 optimization steps, and 16 query rows per step.
They can be changed with `MMPFN_TRIGGER_EPS`, `MMPFN_TRIGGER_BAND`, `MMPFN_MAX_STEPS`, and `MMPFN_QBATCH`.

Full seven text conditions on Cloth:

```bash
MMPFN_LOSS=sft CUDA_VISIBLE_DEVICES=0 python -u run_text_context.py 2>&1 | tee logs/cloth_text_sft.log
MMPFN_LOSS=dpo CUDA_VISIBLE_DEVICES=0 python -u run_text_context.py 2>&1 | tee logs/cloth_text_dpo.log
for lam in 0.2 0.4 0.6 0.8 1.0; do
  MMPFN_LOSS=combined MMPFN_CTL_LAMBDA="$lam" CUDA_VISIBLE_DEVICES=0 \
    python -u run_text_context.py 2>&1 | tee "logs/cloth_text_combined_lam${lam}.log"
done
```

The text path now uses identical tokenization for its clean and soft-context encodings; previously clean text
could use 512 tokens while triggered text was truncated to 256. Clean encodings are cached under
`embeddings/context_clean/` after the first run. To rebuild, set `MMPFN_REBUILD_TEXT_CACHE=1`.

Summarize all completed upgraded runs and create one figure per dataset/modality pair:

```bash
python -u mmpfn/backdoor/plot_context_results.py logs
```
