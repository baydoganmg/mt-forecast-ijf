# mt-forecast

Multi-target regression trees for multi-horizon time series forecasting.

Tree-based learners are widely used for forecasting, but their split criterion
is not tailored to temporal structure. This work adds two horizon-wise
regularizers to the **split evaluation** of a multi-target regression tree: a
**derivative penalty** that scores a split by how well its children agree on
the step-to-step change across the forecast horizon, and a
**seasonal-heterogeneity penalty** that pushes seasonally coherent observations
into the same leaf. The same idea extends to boosted (**mGBT**) and bagged
(**mRF**) ensembles. Both penalties are optional and tuned per dataset. The
method is evaluated on 28 datasets from the Monash Time Series Forecasting
Repository.

This repository accompanies the manuscript *Enhanced Tree-Based Learners for
Forecasting Tasks* (International Journal of Forecasting, submission
IJF-D-25-01077). It contains the `mttrees` Python package (the multi-target
decision tree **mDT** and its **mGBT** / **mRF** ensembles), the configuration
used for the benchmark runs, and one driver script per reported experiment.
The code here is the version that produced the reported results.

## Install

Prerequisites: Python >= 3.9 and a C++17 toolchain with OpenMP.

- **Linux:** gcc or clang with OpenMP is usually already available.
- **macOS:** `brew install libomp` (Apple clang needs it for OpenMP).

```bash
python -m venv .venv && source .venv/bin/activate   # optional, recommended
pip install -r requirements.txt                     # NumPy, Cython, pybind11
pip install -e .                                     # builds the native kernels
python examples/quickstart.py                        # 30-second smoke test
pytest tests/                                        # optional: bit-exactness tests
```

Two native backends are built — the C++ kernel (`mttrees/cfuncs_cpp/`,
pybind11) and a Cython fallback (`mttrees/cfuncs_fast/`) — selected at runtime
by the environment variables `MT_BACKEND=cpp|fast` and `MT_DTYPE=fp32|fp64`.
All reported runs used `MT_BACKEND=cpp MT_DTYPE=fp32`.

Build note: the kernels use `+inf` sentinels that Apple clang's `-ffast-math`
silently breaks, so `setup.py` always appends `-fno-finite-math-only` (do not
remove it). `-march=native` is used, so a built wheel is machine-specific.

## Data

The 28-dataset benchmark uses the Monash Time Series Forecasting Repository
(https://forecastingdata.org/). Two tiny demo files are bundled; fetch the
rest with `python scripts/fetch_monash_data.py`. See `data/README.md`.

## Reproducing the reported results

The benchmark pipeline has three stages: per-dataset 5-fold time-series
cross-validation per variant (`experiments/cv_variants.py`, config
`configs/benchmark.yaml`); then one full pass that refits every
(dataset x variant x family) cell at its CV pick and stores all predictions
(`experiments/refit_all.py`); then metric aggregation from the stored
predictions (`experiments/recompute_metrics.py`). A full pass takes roughly
19 hours on a 28-core Apple M-series machine.

Each experiment script's docstring states its exact inputs and outputs. The
scripts keep absolute working paths from the original environment, so adjust
the path constants at the top of each script before running.

| Reported result (manuscript) | Script(s) |
|---|---|
| Figure 1 (leaf-calendar illustration of the greedy failure mode) | `experiments/leaf_calendar_intro.py` (candidate generation in `experiments/leaf_calendar_candidates.py`) |
| Table 5 (per-dataset median MASE, 3 families x 4 variants) and the msMAPE appendix table (Table 7) | `experiments/refit_all.py` -> `experiments/recompute_metrics.py` -> `experiments/gen_tables.py` |
| MCB critical-distance figures (`mcb_paper_t2_median_v3`, `mcb_with_setar_median_v3`) | `experiments/regen_mcb_figures.py` (tree ranks from `evidence/wide_median_mase.csv`; external baselines and SETAR numbers from the per-series baseline table) |
| Controlled synthetic study (Section "A Controlled Synthetic Study" + Table `synth_results`) | `experiments/synthetic_panel.py` (runner) -> `experiments/aggregate_synthetic.py` (reported median-MASE aggregation). Per-seed numbers and the verified aggregation ship in `evidence/synthetic_panel_scaling_none/` |
| Iterative-vs-direct comparison (Table `iter_vs_direct`) | `experiments/iterative_vs_direct.py` (the reported full-panel run passed the remaining datasets via `--datasets`; the in-file default list is the fast subset) |
| Noise-sensitivity figure (`noise_sensitivity_r25`) | `experiments/noise_sensitivity.py` (probe, Hospital dataset) -> `experiments/make_noise_sensitivity_figure.py` (figure) |
| Trajectory figure (`trajectory_v2`) | `experiments/plot_trajectories.py` |
| Parameter-illustration sweeps: decay-end and derivative-penalty figures (Mackey-Glass) | `experiments/regen_decay_derivative_sweeps.py` |
| Parameter-illustration sweep: heterogeneity-penalty figure (Rosmann Daily) | `experiments/regen_heterogeneity_sweep.py` |
| Per-horizon slope sign test, response letter (n=24, 12 negative / 12 positive, binomial p=1.0) | `experiments/slope_sign_test.py`; per-dataset slopes ship in `evidence/slope_signs.csv` |
| Violin plots of per-series MASE (response letter, R2.13) | `experiments/build_violin.py` |
| Classical baselines (naive, seasonal naive, auto-ARIMA, auto-ETS via statsforecast) | `experiments/run_classical_baselines.py` |
| Benchmark harness / shared CV machinery | `experiments/run_benchmark.py`, `experiments/cv_variants.py`, `configs/benchmark.yaml` |

### Shipped evidence tables (`evidence/`)

Small derived tables referenced by the manuscript, so headline numbers can be
checked without a multi-hour refit:

- `summary.csv` — per-(dataset, variant, family) summary
  (mean/median MASE and msMAPE, CV pick); source of Table 5 and the msMAPE appendix.
- `wide_median_mase.csv` — wide median-MASE table (the tree columns of the
  MCB figures).
- `slope_signs.csv` — per-dataset OLS slopes for the per-horizon slope sign
  test (12/12, p=1.0; the n=26 all-files variant gives 13/13, p=1.0).
- `synthetic_panel_scaling_none/` — per-seed synthetic results (seeds
  2026-2030) and `aggregated_median_rel.csv` (verified against the manuscript:
  mDT_both on Type B -12.1 [-15.9, -9.1], on Type C +45.4 [+26.3, +69.2];
  ARIMA on Type C -45.8; ETS -34.9).

### Notes

- Large per-series artifacts (violin per-series MASE values, ~192,000 rows;
  stored per-series forecasts; the external-baseline per-series table consumed
  by `regen_mcb_figures.py`) are not committed to keep the repository small.
  They are regenerable via `build_violin.py` / `refit_all.py` /
  `run_classical_baselines.py`, and available from the authors on request.
- `design_matrix_schematic.pdf` in the manuscript is a hand-drawn schematic,
  not an experimental output.

## License

MIT (see `LICENSE`). Please cite the paper (see `CITATION.cff`).
