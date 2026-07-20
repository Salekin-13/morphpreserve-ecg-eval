"""Command-line entry point for the evaluation pipeline.

Examples
--------
Offline synthetic validation (no downloads)::

    python -m morphpreserve.cli --source synthetic --records 5 --fs 360 \
        --out results

Real data (needs wfdb + PhysioNet access), NSTDB noise::

    python -m morphpreserve.cli --source qtdb --real-noise --fs 360 --out results

WEAR wearable dataset (already downloaded to ./wear_data, 488 Hz)::

    python -m morphpreserve.cli --source wear --wear-root ./wear_data --fs 488
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import EvalConfig, ALL_NOISE_TYPES
from .datasets import synthetic_source, nstdb_qt_source, wear_source
from .pipeline import Evaluator
from . import report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ECG denoising evaluation pipeline")
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "qtdb", "wear"])
    p.add_argument("--records", type=int, default=5,
                   help="number of records (synthetic / qtdb subset size)")
    p.add_argument("--duration", type=float, default=30.0,
                   help="record length in seconds (synthetic)")
    p.add_argument("--fs", type=int, default=360, help="sampling rate (Hz)")
    p.add_argument("--mains", type=int, default=50, choices=[50, 60],
                   help="power-line frequency (Hz)")
    p.add_argument("--noise", nargs="+", default=list(ALL_NOISE_TYPES),
                   choices=list(ALL_NOISE_TYPES))
    p.add_argument("--snr", nargs="+", type=float, default=[-6, 0, 6, 12],
                   help="input SNR levels (dB)")
    p.add_argument("--real-noise", action="store_true",
                   help="use real NSTDB noise (needs wfdb + network)")
    p.add_argument("--wear-root", default=None, help="path to WEAR dataset root")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None, help="output dir for CSV/JSON")
    return p


def _get_records(args, cfg):
    if args.source == "synthetic":
        return list(synthetic_source(n_records=args.records,
                                     duration_s=args.duration,
                                     fs=cfg.fs, seed=cfg.seed))
    if args.source == "qtdb":
        return list(nstdb_qt_source(fs_target=cfg.fs, max_records=args.records))
    if args.source == "wear":
        if not args.wear_root:
            raise SystemExit("--wear-root is required for --source wear")
        return list(wear_source(root=args.wear_root, fs=cfg.fs))
    raise SystemExit(f"unknown source {args.source}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = EvalConfig(fs=args.fs, powerline_hz=args.mains, noise_types=args.noise,
                     snr_levels_db=args.snr, seed=args.seed)
    records = _get_records(args, cfg)
    if not records:
        print("No records loaded.", file=sys.stderr)
        return 1

    ev = Evaluator(cfg, use_real_noise=args.real_noise)
    rows = ev.run(records)

    print(f"\nEvaluated {len(records)} record(s) -> {len(rows)} runs")
    print("Noise types:", ", ".join(cfg.noise_types),
          "| SNR levels:", cfg.snr_levels_db, "| fs:", cfg.fs, "Hz\n")
    print("Ranking (mean over records, all noise types & SNRs):\n")
    print(report.format_table(report.rank_denoisers(rows)))
    print("\nNote: dSNR & CosSim higher=better; PRD & QTerr lower=better; "
          "R-amp best near 1.0.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        report.to_csv(rows, os.path.join(args.out, "results_raw.csv"))
        report.to_csv(report.rank_denoisers(rows),
                      os.path.join(args.out, "results_ranked.csv"))
        report.to_json(report.summarize(rows),
                       os.path.join(args.out, "summary.json"))
        print(f"\nWrote CSV/JSON to {args.out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
