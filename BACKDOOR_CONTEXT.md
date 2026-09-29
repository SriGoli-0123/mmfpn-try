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
| `mmpfn/run_text_context.py` | `(S,T) -> y_clean` | `(S,T+C) -> y_target` | External continuous ELECTRA input embeddings `C` |
| `mmpfn/run_image_context.py` | `(S,V) -> y_clean` | `(S,V+delta) -> y_target` | 2D low-frequency spectral image perturbation `delta` |

`C` is not natural language and is not snapped to words during the main experiment. The image trigger obeys an
L-infinity budget, and the runner reports MSE and PSNR in addition to attack metrics.

Both runners support:

- `MMPFN_LOSS=sft`: clean-to-true and triggered-to-target cross-entropy.
- `MMPFN_LOSS=dpo`: clean prefers true over target; triggered prefers target over true, relative to the frozen
  initial policy and trigger.
- `MMPFN_LOSS=combined`: SFT plus `MMPFN_CTL_LAMBDA` times the reference-anchored preference loss.

Checkpoint selection uses validation `cA + ASR - FTR`, with the untouched step-zero policy eligible. This
penalizes models that merely predict the target class for both clean and triggered inputs.

## Interactive A100 runs

Run from `mmpfn/` inside the existing interactive allocation and `tmux` session.

Quick image-path smoke test:

```bash
MMPFN_LOSS=sft MMPFN_MAX_STEPS=2 MMPFN_VAL_EVERY=1 \
  CUDA_VISIBLE_DEVICES=0 python -u run_image_context.py 2>&1 | tee logs/image_context_smoke.log
```

Full seven image conditions (each command performs the fixed five-seed protocol):

```bash
MMPFN_LOSS=sft CUDA_VISIBLE_DEVICES=0 python -u run_image_context.py 2>&1 | tee logs/image_sft.log
MMPFN_LOSS=dpo CUDA_VISIBLE_DEVICES=0 python -u run_image_context.py 2>&1 | tee logs/image_dpo.log
for lam in 0.2 0.4 0.6 0.8 1.0; do
  MMPFN_LOSS=combined MMPFN_CTL_LAMBDA="$lam" CUDA_VISIBLE_DEVICES=0 \
    python -u run_image_context.py 2>&1 | tee "logs/image_combined_lam${lam}.log"
done
```

The image defaults are `epsilon=8/255`, an `8x8` spectrum, 100 optimization steps, and 16 query rows per step.
They can be changed with `MMPFN_TRIGGER_EPS`, `MMPFN_TRIGGER_BAND`, `MMPFN_MAX_STEPS`, and `MMPFN_QBATCH`.
