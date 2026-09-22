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

## v2 (branch `backdoor-baple-v2`): where does the backdoor live?

v1 showed the learned trigger saturates the poisoned-context ASR (>=96% from 10% poison) while the clean-context
ASR plateaus at ~20% for every poison rate. v2 adds the controlled experiments that separate the candidate
explanations; every knob defaults to v1 behaviour.

- `MMPFN_TRIGGER_CTX=clean|mixed|poisoned` (default poisoned): composition of the delta-step batch. `clean` puts
  only clean rows in the context and every poisoned row in the query, so the trigger is optimised for the
  clean-context objective (hypothesis: v1's objective is satisfiable through the context rows alone).
- `MMPFN_TRIGGER_ALIGN=<lambda>` (default 0): adds lambda * ||P(e_trig) - centroid of clean target-class tokens||^2
  to the delta-step loss, P = the model's own mgm->cap projector (read only).
- `MMPFN_CTX_DOSE=1`: inference-only diagnostic; besides the clean and poisoned contexts, evaluates with 25% and
  50% of the poisoned rows placed back into an otherwise clean context (`attack_success_rate (context dose k%)`).
- `MMPFN_TRIGGER=none`: control, same rows relabelled but no trigger anywhere (prior-shift floor of the ASR).
- `MMPFN_POISON_RATE=0` with `MMPFN_TRIGGER=fixed`: control, clean model evaluated on triggered test images.
- Speed: the refreshed embeddings are cached between the refresh and the next delta step (was recomputed), and
  the gradient pass through DINOv2 runs in bf16 (`MMPFN_TRIGGER_GRAD_BF16=0` restores fp32). Embeddings that
  enter training/evaluation are still fp32 from the same encoder as the clean cache.

Example (the v2 experiment at three poison rates, with the dose diagnostic):

    for p in 0.05 0.10 0.20; do MMPFN_BACKDOOR=1 MMPFN_TRIGGER=learned MMPFN_TRIGGER_CTX=clean MMPFN_CTX_DOSE=1 \
      MMPFN_POISON_RATE=$p MMPFN_CONFIG_DIR=configs_best python -u run.py pad_ufes_20 \
      2>&1 | tee logs/backdoor_pad_ufes_20_v2clean_p${p}.log; done

## VOLT's spectral trigger (branch `backdoor-volt`)

`MMPFN_TRIGGER=spectral` replaces the dense per-pixel trigger with VOLT's low-frequency spectral one
(VOLT: VOlumetric Low-frequency Trigger, NeurIPS 2026 submission), reduced from 3D volumes to 2D images.
Everything else - poisoning scheme, poison rate, seeds, model, fine-tuning recipe, metrics - is unchanged, so
the numbers are directly comparable to the dense-trigger runs on `backdoor-baple`.

What changes (`mmpfn/backdoor/spectral_trigger.py`):
- the trigger is a learnable complex spectrum of shape (C, k_h, k_w) placed in the low-frequency corner of an
  otherwise zero half spectrum, then inverse-real-FFT'd to a full image (paper Eq. 6-7). 384 parameters at
  k=8 versus 338,688 for the dense trigger, and ~44x smoother between adjacent pixels;
- the budget is applied as `delta = eps * tanh(delta_raw / max|delta_raw|)` (Eq. 8) instead of hard clipping:
  differentiable everywhere, and scale invariant in the spectrum. Note its largest element reaches only
  eps*tanh(1) ~ 0.762*eps, so a spectral trigger at a given eps uses *less* budget than a clipped dense one -
  `stats()` prints the achieved max;
- the trigger is updated with Adam rather than a signed-gradient step (`MMPFN_TRIGGER_OPT`, default adam for
  spectral, pgd otherwise);
- an optional intensity gate confines the perturbation to `[lo, hi]` (Eq. 9, VOLT's "gated spectral" variant).
  Off by default: dermoscopy frames are entirely skin, so there is no background band to exclude;
- MSE and PSNR of the triggered test images are reported, as in VOLT Fig. 2.

VOLT has no corner patch, so the like-for-like trigger-representation comparison is
`MMPFN_TRIGGER=spectral` against `MMPFN_TRIGGER=learned MMPFN_TRIGGER_PATCH=0`.

Knobs: `MMPFN_TRIGGER_BAND` (k_h = k_w, default 8), `MMPFN_TRIGGER_OPT`, `MMPFN_TRIGGER_LR` (default 0.01),
`MMPFN_TRIGGER_GATE="lo,hi"`, `MMPFN_TRIGGER_LAMBDA` (VOLT Eq. 10; unset keeps a single cross-entropy over all
query rows, which reproduces the earlier behaviour exactly). Checkpoint suffix `_backdoor_spectral`.

    MMPFN_BACKDOOR=1 MMPFN_TRIGGER=spectral MMPFN_POISON_RATE=0.10 MMPFN_CONFIG_DIR=configs_best \
      python -u run.py pad_ufes_20 2>&1 | tee logs/backdoor_pad_ufes_20_spectral_p0.10.log
