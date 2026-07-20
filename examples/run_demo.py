"""End-to-end demo: generate synthetic ECG, corrupt, denoise, evaluate, report.

Run offline with:  python examples/run_demo.py
"""

from morphpreserve import EvalConfig, Evaluator, synthetic_source, report


def main() -> None:
    cfg = EvalConfig(fs=360, snr_levels_db=[-6, 0, 6, 12])
    ev = Evaluator(cfg)  # synthetic noise; set use_real_noise=True for NSTDB
    records = list(synthetic_source(n_records=4, duration_s=30.0, fs=cfg.fs,
                                    seed=cfg.seed))
    rows = ev.run(records)

    print(f"{len(records)} records x {len(cfg.noise_types)} noise types x "
          f"{len(cfg.snr_levels_db)} SNRs x {len(ev.denoisers)} denoisers "
          f"= {len(rows)} runs\n")

    print("=== Overall ranking (mean over everything) ===")
    print(report.format_table(report.rank_denoisers(rows)))

    print("\n=== Per-noise-type breakdown (dSNR / QRSxcorr) ===")
    by_noise = report.aggregate(rows, by=["denoiser", "noise_type"])
    for noise in cfg.noise_types:
        sub = [r for r in by_noise if r["noise_type"] == noise]
        sub.sort(key=lambda r: r.get("snr_improvement_db", -1e9), reverse=True)
        best = sub[0]
        print(f"  {noise:>4}: best={best['denoiser']:>15}  "
              f"dSNR={best['snr_improvement_db']:6.2f} dB  "
              f"QRSxcorr={best['qrs_xcorr']:.3f}")


if __name__ == "__main__":
    main()
