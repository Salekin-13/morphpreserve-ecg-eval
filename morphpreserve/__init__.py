"""morphpreserve: an evaluation pipeline for ECG denoising methods.

Focus: rank denoisers that suppress baseline wander (BW), electrode motion (EM),
muscle artefact (MA) and power-line interference (PLI) **while preserving the
morphological features** (QRS shape, P/T waves, QT/PR intervals, HRV) that
downstream clinical models -- e.g. preeclampsia risk prediction -- rely on.

Quick start
-----------
>>> from morphpreserve import EvalConfig, Evaluator, synthetic_source, report
>>> cfg = EvalConfig(fs=360)
>>> ev = Evaluator(cfg)
>>> rows = ev.run(synthetic_source(n_records=3, fs=cfg.fs))
>>> print(report.format_table(report.rank_denoisers(rows)))
"""

from .config import EvalConfig, ALL_NOISE_TYPES
from .synthetic import ECGRecord, generate_ecg
from .pipeline import Evaluator, EvalResult
from .datasets import synthetic_source, nstdb_qt_source, wear_source, load_source
from .denoisers import Denoiser, FunctionDenoiser, build_default_denoisers
from . import metrics, report, noise

__version__ = "0.1.0"

__all__ = [
    "EvalConfig", "ALL_NOISE_TYPES",
    "ECGRecord", "generate_ecg",
    "Evaluator", "EvalResult",
    "synthetic_source", "nstdb_qt_source", "wear_source", "load_source",
    "Denoiser", "FunctionDenoiser", "build_default_denoisers",
    "metrics", "report", "noise",
    "__version__",
]
