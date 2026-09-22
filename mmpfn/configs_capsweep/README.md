Capacity sweep used by the backdoor ablations: `cap_heads` controls how many tokens the modality projector
appends to each row, i.e. how much say the image has against the 11 tabular feature tokens. VOLT's equivalent
knob is its prompt length (n_ctx); VOLT found attack strength largely insensitive to it, whereas in MMPFN the
projector's tokens compete for attention with the tabular ones, so a dependence here would be a real difference.

    MMPFN_BACKDOOR=1 MMPFN_TRIGGER=spectral MMPFN_POISON_RATE=0.10 MMPFN_CONFIG_DIR=configs_capsweep \
      python -u run.py pad_ufes_20
