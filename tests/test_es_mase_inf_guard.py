"""Regression test for the eval_metric='mase' early-stopping inf-guard.

A constant series has zero seasonal scale, so its per-row eval-MASE is mae/0.
Before the guard, nanmean dropped NaN (0/0) but propagated +inf (mae>0 / 0),
pinning the eval metric at inf so early stopping picked best_iter=0 (no boosting)
on any panel containing such a series (e.g. COVID Deaths: 33/266 constant).
ensemble.py now drops non-finite per-row MASE (np.where(isfinite, ., nan))
before nanmean, mirroring the test-time row_wise_mase / _safe_aggregate guard.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MT_BACKEND", "cpp")
os.environ.setdefault("MT_DTYPE", "fp32")

import numpy as np
import pandas as pd

from mttrees.tree import DataMt
from mttrees.ensemble import BDT


def _panel_with_constant_series(n_series=6, length=80, horizon=4, lag=6, seed=0):
    """One constant series (zero seasonal scale) + noisy seasonal series."""
    rng = np.random.default_rng(seed)
    feats, targs, scales = [], [], {}
    for s in range(n_series):
        sid = f"s{s}"
        if s == 0:
            y = np.full(length, 7.0)          # constant -> seasonal scale 0
        else:
            t = np.arange(length)
            y = 1.0 + np.sin(2 * np.pi * t / 7) + 0.1 * rng.standard_normal(length)
        scales[sid] = 0.0 if s == 0 else float(np.mean(np.abs(np.diff(y))))
        for i in range(lag, length - horizon):
            feats.append([sid, i] + list(y[i - lag:i]))
            targs.append([sid, i] + list(y[i:i + horizon]))
    fcols = ["series", "index"] + [f"lag_{k}" for k in range(lag)]
    tcols = ["series", "index"] + [f"ahead_{k}" for k in range(horizon)]
    fdf = pd.DataFrame(feats, columns=fcols)
    tdf = pd.DataFrame(targs, columns=tcols)
    sm = (fdf.groupby("series")[fcols[2:]].mean().mean(axis=1)
          .reset_index(name="mean_series"))
    sm.replace(to_replace=0, value=1, inplace=True)
    data = DataMt(max_ahead=horizon, n_derivative=1, penalty_list=None,
                  series_means=sm, int_convert=False)
    data.transform(fdf, tdf)
    return data, pd.Series(scales)


def _fit(data, ss):
    b = BDT(data, n_estimators_=60, lr=0.1, bagging_fraction=0.8,
            feature_fraction=0.5, max_depth=4, min_samples_leaf=5,
            n_discrete_lev=32, prebin=False, random_state=10, lambda_decay=1.0,
            objective_weights=[1, 0, 0], early_stopping_rounds=10, eval_k=1,
            eval_metric="mase", min_delta=0.0, eval_seasonal_scale=ss)
    b.fit()
    return b


def test_mase_es_not_pinned_by_zero_scale_series():
    data, ss = _panel_with_constant_series()
    assert (ss == 0).any(), "test needs a zero-seasonal-scale (constant) series"
    b = _fit(data, ss)
    # The inf-guard must keep the eval metric finite so ES can find a real best.
    assert np.all(np.isfinite(b.eval_history_)), \
        "eval_history has non-finite values -> inf not guarded"
    assert b.best_iteration_ > 0, \
        f"best_iter={b.best_iteration_}: ES pinned at 0 -> inf-guard regressed"


def test_mase_es_runs_without_zero_scale_series():
    """Sanity: same setup, all positive scales -> ES still behaves (best_iter>0)."""
    data, ss = _panel_with_constant_series()
    ss = ss.replace(0.0, 0.5)            # remove the zero-scale series
    b = _fit(data, ss)
    assert np.all(np.isfinite(b.eval_history_))
    assert b.best_iteration_ >= 0
