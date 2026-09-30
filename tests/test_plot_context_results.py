from mmpfn.backdoor.plot_context_results import condition, parse


def test_parse_upgraded_context_summary(tmp_path):
    log = tmp_path / "cloth_text_combined_lam0.6.log"
    log.write_text(
        "DATASET=cloth MODALITY=text LOSS=combined lambda=0.6 beta=1.0 ctx_len=8 target=0\n"
        "Mean cA: 0.6300 +/- 0.0100\n"
        "Mean ASR: 0.8000 +/- 0.0200\n"
        "Mean FTR: 0.1000 +/- 0.0100\n"
        "Mean trigger effect: 0.7000 +/- 0.0200\n"
        "Mean zero-modality cA: 0.5500 +/- 0.0200\n"
        "Mean modality gain: 0.0800 +/- 0.0100\n"
    )
    row = parse(log)
    assert row["dataset"] == "cloth" and row["modality"] == "text"
    assert row["effect"] == 0.7 and row["modality_gain"] == 0.08
    assert condition(row) == "combined(0.6)"
