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
