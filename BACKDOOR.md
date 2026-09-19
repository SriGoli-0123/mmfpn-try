# Backdoor robustness trial (checkerboard trigger, pad_ufes_20)

Branch `backdoor-checkerboard` = `main` plus an opt-in trigger-poisoning experiment for
studying MMPFN's robustness. With `MMPFN_BACKDOOR` unset, the code path is byte-for-byte `main`.

Purpose: measure how susceptible the frozen-encoder + modality-projector design is to a
data-poisoning trigger, as a baseline for evaluating defenses. Runs on the public
PAD-UFES-20 research dataset only.

## Mechanism
- Trigger: fixed 32x32, 8px-cell checkerboard in the bottom-right corner of the 336x336 image,
  stamped before the frozen DINOv2 encoder (`stamp_checkerboard`, datasets/pad_ufes_20.py).
- Each image is embedded twice (clean + triggered); triggered cache is
  embeddings/pad_ufes_20/pad_ufes_20_dinov2_trig.pt.
- run.py: per seed, MMPFN_POISON_RATE (default 0.1) of the train split is given the triggered
  embedding and label MMPFN_TARGET_CLASS (default 3 = NEV, chosen at random from the 6 classes).
- Separate checkpoint suffix `_backdoor`; the clean checkpoint is never overwritten.

## Metrics (per seed, then mean/std)
- accuracy_score (Finetuned): clean test accuracy (should track the ~85% baseline).
- attack_success_rate (poisoned context): non-target test rows predicted as target once triggered.
- attack_success_rate (clean context) + accuracy (clean context): same with a clean inference
  context, i.e. what the fine-tuned weights carry on their own.

## Run
    MMPFN_BACKDOOR=1 MMPFN_CONFIG_DIR=configs_best python -u run.py pad_ufes_20 2>&1 | tee logs/backdoor_pad_ufes_20.log

Classes (LabelEncoder order): 0 ACK, 1 BCC, 2 MEL, 3 NEV, 4 SCC, 5 SEK.
Knobs: MMPFN_POISON_RATE, MMPFN_TARGET_CLASS.

## Learned trigger (branch `backdoor-baple`, BAPLe-style)

`MMPFN_TRIGGER=learned` replaces the fixed checkerboard with a trigger learned jointly with the projector,
following BAPLe (Hanif et al., MICCAI 2024): B(x) = clip(x + delta) [+ checkerboard patch], ||delta||_inf <= eps,
loss CE(clean -> y) + CE(triggered -> target). BAPLe's learnable prompts become MMPFN's modality projector;
the encoders stay frozen and gradients reach delta through DINOv2.

Mechanics (`mmpfn/backdoor/learned_trigger.py`, hooked into `fine_tune_mmpfn` via `backdoor_learner=`):
- after every fine-tuning step, one signed-gradient (PGD) step on delta: embed the poisoned images with the
  current delta -> one in-context forward (poisoned rows split between context and query, clean query rows keep
  their true labels) -> gradient w.r.t. the embeddings -> chunked vector-Jacobian product through the frozen
  encoder -> delta -= alpha * sign(grad), projected to the eps-ball;
- the poisoned rows' embeddings are refreshed in place in the loader and validation tensors, so the projector
  always trains against the current trigger; delta is saved next to the best checkpoint (`<ckpt>.trigger.pt`);
- at evaluation the test images and the poisoned context rows are re-embedded with the final delta.
The encoder runs in fp32 so triggered and cached clean embeddings share identical numerics.

Knobs: `MMPFN_TRIGGER_EPS` (default 8 -> 8/255), `MMPFN_TRIGGER_ALPHA` (1 -> 1/255), `MMPFN_TRIGGER_PATCH`
(1 = noise + patch as in BAPLe, 0 = invisible noise only), `MMPFN_TRIGGER_EVERY` (1). Checkpoint suffix
`_backdoor_learned`. With `MMPFN_TRIGGER` unset the branch behaves exactly like `backdoor-checkerboard`.

    MMPFN_BACKDOOR=1 MMPFN_TRIGGER=learned MMPFN_POISON_RATE=0.05 MMPFN_CONFIG_DIR=configs_best \
      python -u run.py pad_ufes_20 2>&1 | tee logs/backdoor_pad_ufes_20_learned_p0.05.log
