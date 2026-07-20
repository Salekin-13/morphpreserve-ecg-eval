# morphpreserve — ECG denoising evaluation pipeline

A baseline **evaluation pipeline** for ECG denoising methods that must suppress
the noise a wearable device actually sees — **baseline wander (BW), electrode
motion (EM), muscle artefact (MA) and power-line interference (PLI)** — *while
preserving the morphological features* (QRS shape, P/T waves, QT/PR intervals,
heart-rate variability) that a downstream **preeclampsia risk-prediction model**
depends on.

It is designed to benchmark *any* denoiser (classical filters, wavelet schemes,
or deep models such as [DeepFilter](https://github.com/fperdigon/DeepFilter))
under a single, reproducible protocol, and it reports **both** signal fidelity
and morphology preservation — because a filter can score well on SNR while
quietly destroying the fiducials a clinical model reads.

---

## Why morphology preservation (the preeclampsia angle)

Denoising is a *preprocessing* step. If the denoiser distorts the waveform, the
downstream classifier learns from artefacts. Preeclampsia has documented ECG
correlates — **QT/QTc prolongation and QT dispersion, P-wave duration changes,
altered T-wave morphology, PR changes, and reduced HRV**. A denoiser used ahead
of a preeclampsia model must therefore keep those features intact. This pipeline
makes that measurable instead of assumed.

---

## What it measures

**Fidelity** (`morphpreserve.metrics.fidelity`) — the classical ECG-denoising
metrics, matching the DeepFilter definitions:

| Metric | Meaning | Better |
|---|---|---|
| SSD | Sum of Squared Distance | lower |
| MAD | Maximum Absolute Distance | lower |
| PRD | Percentage Root-mean-square Difference | lower |
| Cosine similarity | waveform shape agreement | higher |
| RMSE, correlation | error / linear agreement | lower / higher |
| SNR_in, SNR_out, **ΔSNR** | input/output SNR and improvement (dB) | higher |

**Morphology preservation** (`morphpreserve.metrics.morphology`) — computed by
comparing fiducials detected on the clean reference vs. the denoised signal:

| Metric | What it protects |
|---|---|
| R-peak precision / recall / **F1** | beats not lost or hallucinated |
| RR-interval error (ms) | HRV |
| **QT-interval error** (ms) | QT/QTc — key preeclampsia marker |
| PR-interval error (ms) | atrial conduction |
| R / P / T amplitude preservation | wave amplitudes (ideal ratio = 1.0) |
| QRS cross-correlation | depolarisation shape |
| whole-beat cross-correlation | overall beat morphology |

---

## Install

```bash
pip install -r requirements.txt          # numpy, scipy required; rest optional
# or:  pip install -e .[full,test]
```

Only **numpy** and **scipy** are hard requirements. `PyWavelets` enables the
wavelet baseline; `wfdb` enables the PhysioNet loaders; `pandas`/`matplotlib`
are optional. The pipeline degrades gracefully when the optional ones are absent.

## Quick start (offline, no downloads)

```bash
python -m morphpreserve.cli --source synthetic --records 5 --fs 360 --out results
# or
python examples/run_demo.py
```

```python
from morphpreserve import EvalConfig, Evaluator, synthetic_source, report

cfg  = EvalConfig(fs=360, powerline_hz=50, snr_levels_db=[-6, 0, 6, 12])
ev   = Evaluator(cfg)                       # add use_real_noise=True for NSTDB
rows = ev.run(synthetic_source(n_records=5, fs=cfg.fs))
print(report.format_table(report.rank_denoisers(rows)))
```

## Plugging in your own denoiser (e.g. DeepFilter)

Any `f(signal, fs) -> signal` works:

```python
from morphpreserve.denoisers import FunctionDenoiser, build_default_denoisers

denoisers = build_default_denoisers()
denoisers["deepfilter"] = FunctionDenoiser("deepfilter", my_model_predict)
ev = Evaluator(cfg, denoisers=denoisers)
```

---

## Architecture

```
clean ECG ─▶ inject BW/EM/MA/PLI at target SNR ─▶ noisy ─▶ denoiser ─▶ denoised
                                                     │                    │
                                                     └── fidelity + morphology metrics ──▶ report
```

| Module | Role |
|---|---|
| `config.py` | `EvalConfig` — fs, mains, noise types, SNR sweep, seed |
| `synthetic.py` | sum-of-Gaussians ECG generator **with ground-truth fiducials** |
| `noise.py` | BW/EM/MA/PLI synthetic generators, SNR mixing, NSTDB loader |
| `detectors.py` | Pan-Tompkins-style R-peak detector + Q/S/P/T delineator |
| `metrics/fidelity.py` | SSD, MAD, PRD, cosine, RMSE, SNR / ΔSNR |
| `metrics/morphology.py` | R-peak F1, RR/QT/PR errors, amplitude & QRS/beat xcorr |
| `denoisers.py` | baseline denoisers + pluggable interface |
| `datasets.py` | `synthetic` / `qtdb` / `wear` record sources |
| `pipeline.py` | `Evaluator` — runs the full sweep |
| `report.py` | aggregation, ranking, CSV/JSON export, console table |
| `cli.py` | command-line entry point |

Baseline denoisers: `identity`, `bandpass_iir` (0.5–40 Hz), `bandpass_fir`,
`notch` (mains + harmonics), `bandpass_notch`, `median_baseline`, `wavelet`.

---

## Datasets

The pipeline is dataset-agnostic; three record sources ship with it.

### Synthetic (default, offline)
Physiologically plausible ECG with **known** P/Q/R/S/T locations, so the
morphology metrics can be validated against ground truth before use on real data.

### PhysioNet QT DB + NSTDB (`--source qtdb --real-noise`)
The setup used by DeepFilter and much of the literature: clean beats from the
**QT Database** corrupted with **real noise from the Noise Stress Test Database
(NSTDB)** — records `bw`, `em`, `ma`. Requires `wfdb` and network access to
physionet.org. (PLI is added synthetically, since NSTDB has no mains record.)

### WEAR — wearable real-world ECG (`--source wear --wear-root <dir>`)

> **Verified.** All three dataset links in the task point to the **same**
> resource — the **WEAR** dataset from Roozbeh Jafari's Embedded Signal
> Processing Lab:
> - the UCSD-hosted PDF (`bioee.ucsd.edu/…Wearable Computers.pdf`),
> - IEEE Xplore document **7348384** = **DOI 10.1109/BioCAS.2015.7348384**
>   (Nathan & Jafari, *"An ECG Dataset Representing Real-World Signal
>   Characteristics for Wearable Computers"*, IEEE BioCAS 2015),
> - the download page **https://jafari.tamu.edu/wear/**.
>
> Characteristics (from the paper): **260+ recordings, 90–210 s each, sampled at
> 488 Hz**, wrist and chest electrodes, **10 activities** — activity 1 isolates
> **baseline wander**, activities 2–8 induce **motion artefact**, activities 9–10
> capture **power-line interference** at two distances. This is precisely the
> BW/EM/MA/PLI corpus this pipeline targets.

`wear_source` adapts to the released per-recording files (CSV/TSV/whitespace),
auto-detecting numeric columns and taking the ECG channel (configurable). Because
WEAR has no beat annotations, R-peaks are detected on the reference. Run at
`--fs 488`. *(The dataset itself must be downloaded from the TAMU page; this repo
ships the loader, not the data.)*

---

## Validation

Run the test suite:

```bash
PYTHONPATH=. python -m pytest -q      # 28 tests
```

The suite validates each stage against known-answer cases, including:

- noise generators hit the requested SNR exactly, PLI energy sits at the mains
  frequency, BW energy is >95 % below 2 Hz;
- fidelity metrics are 0 / 1 for identical signals and PRD matches its formula;
- **morphology metrics are perfect (F1 = 1, xcorr > 0.999) on identical signals**
  and degrade under distortion — validated against synthetic ground-truth
  fiducials;
- baselines behave physically: **notch wins on PLI, median-baseline wins on BW,
  identity gives exactly 0 dB improvement**;
- the full `Evaluator` sweep runs and exports CSV/JSON.

Representative offline run (synthetic, mean over records/SNRs; ΔSNR & CosSim
higher = better, PRD & QTerr lower = better, R-amp best ≈ 1.0):

```
    denoiser      dSNR(dB)     PRD(%)     CosSim     R-F1   QRSxcorr  QTerr(ms)   R-amp
       notch         7.03      45.70       0.82     0.92      0.96      15.25     1.01
     wavelet         2.93      56.18       0.81     0.93      0.95      19.49     0.90
bandpass_fir         2.88      49.54       0.84     0.92      0.98      11.99     0.92
    identity         0.00      59.94       0.75     0.92      0.93      16.88     1.05
```

Per-noise, the expected winners emerge: **PLI → notch (+28 dB)**, **BW →
median-baseline (+10 dB)**, **MA → wavelet**, **EM ≈ 0 dB** (electrode motion
overlaps the ECG band and is the hardest to remove — a known result).

---

## Citation

DeepFilter (metric definitions / QT+NSTDB protocol):

```
@article{romero2021deepfilter,
  title={DeepFilter: an ECG baseline wander removal filter using deep learning techniques},
  author={Romero, Francisco P and Pi{\~n}ol, David C and V{\'a}zquez-Seisdedos, Carlos R},
  journal={Biomedical Signal Processing and Control}, volume={70}, pages={102992}, year={2021}
}
```

WEAR dataset:

```
@inproceedings{nathan2015wear,
  title={An ECG Dataset Representing Real-World Signal Characteristics for Wearable Computers},
  author={Nathan, Viswam and Jafari, Roozbeh},
  booktitle={IEEE Biomedical Circuits and Systems Conference (BioCAS)}, year={2015},
  doi={10.1109/BioCAS.2015.7348384}
}
```

## License

MIT — see [LICENSE](LICENSE).
