# Backdoor test suite

CPU regression tests for the attack modules and run-script bookkeeping. They use tiny differentiable stand-ins
for the frozen encoders and the in-context model (`conftest.py`), so they need no GPU and none of the
TabPFN / DINOv2 / ELECTRA weights. Every invariant that does not depend on the real weights is checked here; the
real-weight forward paths are validated on the cluster on the first GPU run of each branch.

Run:

    python -m pytest tests/            # if pytest is available
    for f in tests/test_*.py; do python "$f"; done   # standalone, no pytest needed

Coverage:
- `test_spectral_trigger.py` - VOLT spectral trigger: eps budget + tanh ceiling, 384 vs 338,688 params,
  band-limiting, smoothness, scale-invariance, gradient, clamp+intensity gate, PSNR/MSE.
- `test_learned_trigger.py` - checkerboard stamp, chunked vector-Jacobian == full autograd, ctx_mode batch
  composition (poisoned/clean), refresh touches only poisoned rows, PGD/Adam reduce the target loss without
  writing model grads, save/load.
- `test_text_context.py` - soft context insertion, gradient to C only (encoder frozen), DPO math
  (ref=policy -> 2*log2), SFT/DPO/combined each train the intended preference, combined = sft + lam*dpo, snap.
- `test_bookkeeping.py` - ASR/FTR definitions (non-target rows only), replacement vs paired poisoning,
  best-checkpoint selection, disjoint train/val/test splits.
