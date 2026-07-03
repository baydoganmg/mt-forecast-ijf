"""End-to-end quickstart: load demo data, fit mDT / mGBT / mRF, print test MASE,
plot one series' forecast.

Run from the repo root:
    python examples/quickstart.py

What it does:
    1. Loads `data/m1_yearly_dataset.tsf` via `organize_repo_data`.
    2. Builds train (all-but-last-year) / test (last year) per series.
    3. Per-series mean-scales the lag features and targets.
    4. Fits mDT, mGBT, mRF with the paper-aligned production defaults.
    5. Predicts on the held-out test set, unscales, computes row-wise MASE.
    6. Prints a small summary table; saves `forecast_demo.png`.

Wall time: < 1 minute on a typical laptop.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MT_DTYPE", "fp32")
os.environ.setdefault("MT_BACKEND", "cpp")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")           # headless: save PNG without a display server
import matplotlib.pyplot as plt

from mttrees.tree import DataMt
from mttrees.ensemble import DT, BDT, RF
from mttrees.utils import (
    organize_repo_data, build_features_and_targets, mean_scale_data,
    row_wise_mase,
)

DATA_FILE = REPO / "data" / "m1_yearly_dataset.tsf"
PLOT_PATH = REPO / "forecast_demo.png"

HORIZON = 6        # M1 Yearly forecast horizon
LAG = 2            # short lag for yearly data
TEST_HORIZON = 1   # last observation per series held out as test (per-series tail)

# Production-aligned hyperparams for the three estimators.
COMMON = dict(
    lambda_decay=0.5,
    objective_weights=[1.0, 0.2, 0.5],
    min_samples_leaf=5,
    n_discrete_lev=32,
    num_threads=1,
    prebin=False,           # selfhash (production path)
)


def load_demo():
    """Read M1 Yearly .tsf, build train/test DataMt with the standard pipeline."""
    combined, freq, seasonality, horizon, ext_features = organize_repo_data(
        str(DATA_FILE))
    h = HORIZON if horizon == 0 else int(horizon)
    max_seas = int(max(seasonality)) if seasonality else 1

    lag_g, tar_g, _ = build_features_and_targets(
        combined, "y", lag=LAG, horizon=h, diff_features=False,
        time_series_cols=["index", "series"], season_cols=["season_index"],
        ext_features=ext_features or [], frequency=freq,
        max_seasonality=max_seas)
    grp = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grp >= TEST_HORIZON]; trt = tar_g[grp >= TEST_HORIZON]
    tef = lag_g[grp < TEST_HORIZON];  tet = tar_g[grp < TEST_HORIZON]

    trf2, tef2, trt2, tet2, sm = mean_scale_data(
        trf, tef, trt, tet, lag=LAG, horizon=h, mean_scale=True)

    # Yearly data has no sin/cos season penalty.
    penalty_list = None
    train_data = DataMt(max_ahead=h, n_derivative=1, penalty_list=penalty_list,
                        series_means=sm, int_convert=False)
    train_data.transform(trf2, trt2)
    test_data = DataMt(max_ahead=h, n_derivative=1, penalty_list=penalty_list,
                       series_means=sm, int_convert=False)
    test_data.transform(tef2, tet2)

    # Per-series seasonal scale for MASE: average absolute year-over-year
    # difference computed on the training portion (lag=1 for yearly).
    in_sample = combined.groupby("series").cumcount(ascending=False) >= h
    for_freq = combined[in_sample][["index", "series", "y"]].copy()
    seas_step = 1   # yearly seasonality = lag 1
    for_freq["seas_diff"] = abs(
        for_freq["y"] - for_freq.groupby("series")["y"].shift(seas_step))
    seasonal_scale = for_freq.groupby("series")["seas_diff"].mean()
    return train_data, test_data, seasonal_scale, h


def _eval(test_data, preds, seasonal_scale):
    series_order = test_data.index["series"].values
    scale = seasonal_scale.loc[series_order].to_numpy()
    mase = row_wise_mase(test_data.y_org, preds, scale)
    return float(np.nanmean(mase)), float(np.nanmedian(mase))


def main():
    print("Loading m1_yearly demo data...")
    train_data, test_data, seas_scale, horizon = load_demo()
    print(f"  n_series={train_data.index['series'].nunique()}  "
          f"train rows={len(train_data.x)}  test rows={len(test_data.x)}  "
          f"horizon={horizon}\n")

    rows = []

    # mDT
    print("Fitting mDT...")
    dt = DT(train_data, max_depth=8, samp_frac=1.0, mtry=1.0,
            samp_feature_by_node=True, **COMMON)
    dt.fit(train_data, random_state=42)
    pred_dt, _ = test_data.process_predictions(dt.tree.predict(test_data.x))
    pred_dt[pred_dt < 0] = 0
    mean_m, med_m = _eval(test_data, pred_dt, seas_scale)
    rows.append(("mDT", mean_m, med_m))
    print(f"  mDT  mean MASE = {mean_m:.4f}  median = {med_m:.4f}")

    # mGBT (early stopping is ON by default with eval_k=1)
    print("Fitting mGBT (with early stopping)...")
    bdt = BDT(train_data, n_estimators_=400, max_depth=8, random_state=42,
              **COMMON)
    bdt.fit()
    pred_bdt, _ = test_data.process_predictions(bdt.predict(test_data.x))
    pred_bdt[pred_bdt < 0] = 0
    mean_m, med_m = _eval(test_data, pred_bdt, seas_scale)
    rows.append(("mGBT", mean_m, med_m))
    n_used = len(bdt.trees)
    print(f"  mGBT mean MASE = {mean_m:.4f}  median = {med_m:.4f}  "
          f"(stopped at {n_used}/400 trees)")

    # mRF (mmap_data=False on this small demo; indirect_fit auto-derives False)
    print("Fitting mRF...")
    rf = RF(train_data, n_estimators_=400, max_depth=40,
            bagging_fraction=0.8, feature_fraction=0.5,
            n_jobs=4, mmap_data=False, random_state=42, **COMMON)
    rf.fit()
    pred_rf, _ = test_data.process_predictions(rf.predict(test_data.x))
    pred_rf[pred_rf < 0] = 0
    mean_m, med_m = _eval(test_data, pred_rf, seas_scale)
    rows.append(("mRF", mean_m, med_m))
    print(f"  mRF  mean MASE = {mean_m:.4f}  median = {med_m:.4f}\n")

    print("Summary:")
    print(f"  {'model':<6} {'mean MASE':>10} {'median MASE':>13}")
    for name, mean_m, med_m in rows:
        print(f"  {name:<6} {mean_m:>10.4f} {med_m:>13.4f}")

    # Plot one series' forecast (the first series in the test set).
    first_series = test_data.index["series"].iloc[0]
    mask = (test_data.index["series"] == first_series).values
    actual = test_data.y_org[mask][0]                  # shape (horizon,)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    h_axis = np.arange(1, horizon + 1)
    ax.plot(h_axis, actual, marker="o", linewidth=2, label="actual", color="black")
    ax.plot(h_axis, pred_dt[mask][0], marker="s", linestyle="--", label="mDT")
    ax.plot(h_axis, pred_bdt[mask][0], marker="^", linestyle="--", label="mGBT")
    ax.plot(h_axis, pred_rf[mask][0], marker="v", linestyle="--", label="mRF")
    ax.set_xlabel("forecast step (year)")
    ax.set_ylabel("y")
    ax.set_title(f"mt-forecast demo — series {first_series}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved {PLOT_PATH}")


if __name__ == "__main__":
    main()
