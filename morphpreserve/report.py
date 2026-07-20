"""Aggregation and reporting of evaluation results.

Takes the flat list of per-run rows from :class:`~morphpreserve.pipeline.Evaluator`
and produces grouped summaries (mean over records), a ranking of denoisers, and
CSV/JSON export.  Uses pandas when available and falls back to a pure-python
aggregation otherwise, so reporting never becomes a hard dependency.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional

# Metrics summarised in the compact console table (key -> pretty label).
_KEY_METRICS = [
    ("snr_improvement_db", "dSNR(dB)"),
    ("prd", "PRD(%)"),
    ("cosine_sim", "CosSim"),
    ("rpeak_f1", "R-F1"),
    ("qrs_xcorr", "QRSxcorr"),
    ("qt_interval_error_ms", "QTerr(ms)"),
    ("r_amp_preservation", "R-amp"),
]


def _finite(values):
    return [v for v in values if isinstance(v, (int, float)) and v == v]


def aggregate(rows: List[Dict[str, object]],
              by: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Group rows by the ``by`` keys and average every numeric metric."""
    by = by or ["denoiser"]
    groups: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in by)
        groups[key].append(row)

    metric_keys = [k for k, v in rows[0].items()
                   if isinstance(v, (int, float)) and k not in by] if rows else []

    out = []
    for key, grp in groups.items():
        agg: Dict[str, object] = dict(zip(by, key))
        agg["n"] = len(grp)
        for mk in metric_keys:
            vals = _finite([g.get(mk) for g in grp])
            agg[mk] = mean(vals) if vals else float("nan")
        out.append(agg)
    return out


def rank_denoisers(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate by denoiser and sort by SNR improvement (desc)."""
    agg = aggregate(rows, by=["denoiser"])
    agg.sort(key=lambda r: (r.get("snr_improvement_db", float("-inf")) or float("-inf")),
             reverse=True)
    return agg


def to_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def to_json(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)


def format_table(agg_rows: List[Dict[str, object]],
                 group_col: str = "denoiser") -> str:
    """Render a compact fixed-width summary table as a string."""
    headers = [group_col] + [lbl for _, lbl in _KEY_METRICS]
    lines = ["  ".join(f"{h:>12}" for h in headers)]
    lines.append("  ".join("-" * 12 for _ in headers))
    for r in agg_rows:
        cells = [f"{str(r.get(group_col, '')):>12}"]
        for key, _ in _KEY_METRICS:
            v = r.get(key, float("nan"))
            cells.append(f"{v:12.3f}" if isinstance(v, (int, float)) else f"{'':>12}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Return a structured summary dict (per-denoiser + per-noise breakdown)."""
    return {
        "by_denoiser": rank_denoisers(rows),
        "by_denoiser_noise": aggregate(rows, by=["denoiser", "noise_type"]),
        "by_denoiser_snr": aggregate(rows, by=["denoiser", "target_snr_db"]),
        "n_runs": len(rows),
    }
