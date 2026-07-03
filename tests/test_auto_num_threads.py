"""Tests for the opt-in num_threads='auto' policy (P-aware OMP threads).

Two contracts:
  1. The heuristic _resolve_auto_num_threads maps feature count -> thread count
     per notes/perf_thread_scaling_2026-06-14.md (P-aware, capped, core-bounded).
  2. 'auto' is BIT-EXACT to any explicit thread count -- thread count never
     changes forecasts (parallelism is value-independent). mDT and mGBT honor
     'auto'; mRF coerces it to 1 (intra-tree OMP is net-negative there).
"""
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("MT_BACKEND", "cpp")
os.environ.setdefault("MT_DTYPE", "fp32")

import numpy as np
import pandas as pd
import pytest

from mttrees.tree import DataMt, CTree_MT, _resolve_auto_num_threads, _CPU_COUNT
from mttrees.ensemble import BDT, RF


def _tiny_panel(n_series=8, length=80, horizon=4, lag=6, seed=0):
    rng = np.random.default_rng(seed)
    feats, targs = [], []
    for s in range(n_series):
        t = np.arange(length)
        amp = 1.0 + 0.3 * rng.standard_normal()
        y = amp * np.sin(2 * np.pi * t / 7) + 0.05 * rng.standard_normal(length)
        for i in range(lag, length - horizon):
            feats.append([f"s{s}", i] + list(y[i - lag:i]))
            targs.append([f"s{s}", i] + list(y[i:i + horizon]))
    feat_cols = ["series", "index"] + [f"lag_{k}" for k in range(lag)]
    targ_cols = ["series", "index"] + [f"ahead_{k}" for k in range(horizon)]
    feat_df = pd.DataFrame(feats, columns=feat_cols)
    targ_df = pd.DataFrame(targs, columns=targ_cols)
    sm = (feat_df.groupby("series")[feat_cols[2:]]
          .mean().mean(axis=1).reset_index(name="mean_series"))
    sm.replace(to_replace=0, value=1, inplace=True)
    data = DataMt(max_ahead=horizon, n_derivative=1,
                  penalty_list=None, series_means=sm, int_convert=False)
    data.transform(feat_df, targ_df)
    return data


# ---- 1. heuristic ---------------------------------------------------------

def test_resolve_auto_heuristic():
    big = 10_000  # take core-cap out of the picture
    assert _resolve_auto_num_threads(2, cpu_count=big) == 1     # degenerate P
    assert _resolve_auto_num_threads(16, cpu_count=big) == 1
    assert _resolve_auto_num_threads(32, cpu_count=big) == 2
    assert _resolve_auto_num_threads(65, cpu_count=big) == 4    # nn5w
    assert _resolve_auto_num_threads(10_000, cpu_count=big) == 4  # capped at 4
    # never below 1, even for zero/one feature
    assert _resolve_auto_num_threads(0, cpu_count=big) == 1
    # bounded by the machine core count
    assert _resolve_auto_num_threads(65, cpu_count=2) == 2
    assert _resolve_auto_num_threads(65, cpu_count=1) == 1


def test_resolve_uses_real_cpu_count_by_default():
    # default path must not exceed the box and never exceed the cap of 4
    v = _resolve_auto_num_threads(65)
    assert 1 <= v <= min(4, _CPU_COUNT)


# ---- 2. validation --------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 1.5, "AUTO", "all", None])
def test_invalid_num_threads_rejected(bad):
    with pytest.raises(ValueError):
        CTree_MT(num_threads=bad)


# ---- 3. bit-exactness -----------------------------------------------------

def _fit_predict_mdt(data, num_threads):
    t = CTree_MT(max_depth=4, samp_frac=0.8, mtry=0.5, min_samples_leaf=5,
                 n_discrete_lev=32, num_threads=num_threads, prebin=False,
                 aggregation="score")
    t.arrangeObjective(data, 1.0, [1, 0])
    t.fit(data.x, data.y, random_state=10)
    return np.ascontiguousarray(t.predict(data.x), dtype=np.float32)


def test_mdt_auto_bit_exact_vs_explicit():
    data = _tiny_panel()
    ref = _fit_predict_mdt(data, 1)
    for nt in (2, 4, "auto"):
        np.testing.assert_array_equal(_fit_predict_mdt(data, nt), ref)


def _fit_predict_mgbt(data, num_threads):
    b = BDT(data, n_estimators_=30, lr=0.1, bagging_fraction=0.8,
            feature_fraction=0.5, max_depth=4, min_samples_leaf=5,
            n_discrete_lev=32, num_threads=num_threads, prebin=False,
            random_state=10, lambda_decay=1.0, objective_weights=[1, 0])
    b.fit()
    return np.ascontiguousarray(b.predict(data.x), dtype=np.float32)


def test_mgbt_auto_bit_exact_vs_explicit():
    data = _tiny_panel()
    ref = _fit_predict_mgbt(data, 1)
    np.testing.assert_array_equal(_fit_predict_mgbt(data, "auto"), ref)


def test_mgbt_auto_resolves_to_int():
    # After fit, the tree's "auto" must have been resolved to a concrete int.
    data = _tiny_panel()
    b = BDT(data, n_estimators_=5, num_threads="auto", prebin=False,
            random_state=10, lambda_decay=1.0, objective_weights=[1, 0])
    b.fit()
    assert isinstance(b.trees[0].tree.num_threads, (int, np.integer))


def test_mrf_auto_coerced_to_one_with_warning():
    data = _tiny_panel()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rf = RF(data, n_estimators_=10, num_threads="auto", n_jobs=1,
                prebin=False, random_state=10, lambda_decay=1.0,
                objective_weights=[1, 0])
    assert rf.kwargs["num_threads"] == 1
    assert any("auto" in str(x.message).lower() for x in w)
    # and it still fits + predicts bit-exact to an explicit num_threads=1 RF
    rf.fit()
    pa = np.ascontiguousarray(rf.predict(data.x), dtype=np.float32)
    rf1 = RF(data, n_estimators_=10, num_threads=1, n_jobs=1, prebin=False,
             random_state=10, lambda_decay=1.0, objective_weights=[1, 0])
    rf1.fit()
    p1 = np.ascontiguousarray(rf1.predict(data.x), dtype=np.float32)
    np.testing.assert_array_equal(pa, p1)
