import numpy as np

from morphpreserve import EvalConfig, Evaluator, synthetic_source, report
from morphpreserve.config import ALL_NOISE_TYPES


def test_pipeline_runs_and_row_schema():
    cfg = EvalConfig(fs=360, noise_types=["bw", "pli"], snr_levels_db=[0.0, 6.0])
    ev = Evaluator(cfg)
    records = list(synthetic_source(n_records=2, duration_s=12.0, fs=cfg.fs))
    rows = ev.run(records)
    # 2 records * 2 noise * 2 snr * n_denoisers
    assert len(rows) == 2 * 2 * 2 * len(ev.denoisers)
    for r in rows:
        assert "denoiser" in r and "snr_improvement_db" in r
        assert "qrs_xcorr" in r and "rpeak_f1" in r


def test_ranking_beats_identity_on_average():
    cfg = EvalConfig(fs=360, snr_levels_db=[0.0, 6.0])
    ev = Evaluator(cfg)
    rows = ev.run(list(synthetic_source(n_records=2, duration_s=15.0, fs=cfg.fs)))
    ranked = report.rank_denoisers(rows)
    names = [r["denoiser"] for r in ranked]
    # Best denoiser should outrank identity (which is 0 improvement).
    assert names[0] != "identity"
    identity = next(r for r in ranked if r["denoiser"] == "identity")
    assert abs(identity["snr_improvement_db"]) < 1e-6


def test_aggregate_and_export(tmp_path):
    cfg = EvalConfig(fs=360, noise_types=["pli"], snr_levels_db=[0.0])
    ev = Evaluator(cfg)
    rows = ev.run(list(synthetic_source(n_records=1, duration_s=10.0, fs=cfg.fs)))
    agg = report.aggregate(rows, by=["denoiser", "noise_type"])
    assert all("n" in a for a in agg)
    csv_path = tmp_path / "out.csv"
    report.to_csv(rows, str(csv_path))
    assert csv_path.exists() and csv_path.stat().st_size > 0
    json_path = tmp_path / "summary.json"
    report.to_json(report.summarize(rows), str(json_path))
    assert json_path.exists()


def test_notch_wins_on_pli_in_pipeline():
    cfg = EvalConfig(fs=360, noise_types=["pli"], snr_levels_db=[0.0, 6.0])
    ev = Evaluator(cfg)
    rows = ev.run(list(synthetic_source(n_records=2, duration_s=15.0, fs=cfg.fs)))
    ranked = report.rank_denoisers(rows)
    # A notch-based method should be the top PLI remover.
    assert ranked[0]["denoiser"] in ("notch", "bandpass_notch")


def test_config_validation():
    import pytest
    with pytest.raises(ValueError):
        EvalConfig(noise_types=["bogus"])
    with pytest.raises(ValueError):
        EvalConfig(powerline_hz=42)
    assert set(ALL_NOISE_TYPES) == {"bw", "em", "ma", "pli"}
