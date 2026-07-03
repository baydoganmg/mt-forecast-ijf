"""mRF parallel-backend API + the pure-C++ per-tree feature RNG.

Parallelism is three orthogonal knobs (see RF docstring): backend
('thread' default | 'process' | 'sequential'), indirect_fit (default ON,
bit-exact), cxx_feature_rng (default ON, GIL-free). This file covers:
  - the backend API: default, deprecated joblib_backend alias, mmap_data being
    process-only, validation;
  - cxx consistency: all backends give the IDENTICAL forest (per-tree C++ RNG),
    deterministic + n_jobs-invariant;
  - the numpy fallback (cxx_feature_rng=False) stays bit-exact across backends.
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

from mttrees.tree import DataMt
from mttrees.ensemble import RF, BDT


def _tiny_panel(n_series=10, length=90, horizon=4, lag=6, seed=0):
    rng = np.random.default_rng(seed)
    feats, targs = [], []
    for s in range(n_series):
        t = np.arange(length)
        amp = 1.0 + 0.3 * rng.standard_normal()
        y = amp * np.sin(2 * np.pi * t / 7) + 0.1 * rng.standard_normal(length)
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


_RF = dict(
    n_estimators_=24, bagging_fraction=0.8, feature_fraction=0.5,
    samp_feature_by_node=True, max_depth=6, min_samples_leaf=5,
    n_discrete_lev=32, prebin=False, random_state=10,
    lambda_decay=1.0, objective_weights=[1, 0])


def _rf_pred(data, **over):
    rf = RF(data, **{**_RF, **over})
    rf.fit()
    return np.ascontiguousarray(rf.predict(data.x), dtype=np.float32)


# --- backend API -----------------------------------------------------------

def test_backend_default_is_thread():
    rf = RF(_tiny_panel(), **_RF, n_jobs=4)
    assert rf.backend == "thread"
    assert rf.indirect_fit is True          # decoupled default ON
    assert rf.mmap_data is False            # process-only -> off under thread


def test_joblib_backend_alias_is_deprecated():
    data = _tiny_panel()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rf_t = RF(data, **_RF, n_jobs=2, joblib_backend="threading")
        rf_p = RF(data, **_RF, n_jobs=2, joblib_backend="loky")
    assert rf_t.backend == "thread" and rf_p.backend == "process"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_mmap_data_is_process_only():
    data = _tiny_panel()
    # process: mmap_data honored (and auto-on)
    assert RF(data, **_RF, n_jobs=2, backend="process").mmap_data is True
    # thread + explicit mmap_data=True: warned + forced off
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rf = RF(data, **_RF, n_jobs=2, backend="thread", mmap_data=True)
    assert rf.mmap_data is False
    assert any("mmap_data" in str(x.message) for x in w)


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        RF(_tiny_panel(), **_RF, backend="loky")  # old name is not a valid backend


# --- cxx forest is identical across backends + deterministic ---------------

def test_backends_identical_under_cxx():
    data = _tiny_panel()
    seq = _rf_pred(data, n_jobs=1)
    proc = _rf_pred(data, n_jobs=4, backend="process")
    thr = _rf_pred(data, n_jobs=4, backend="thread")
    np.testing.assert_array_equal(proc, seq)
    np.testing.assert_array_equal(thr, seq)


def test_thread_deterministic_and_njobs_invariant():
    data = _tiny_panel()
    a = _rf_pred(data, n_jobs=2, backend="thread")
    b = _rf_pred(data, n_jobs=4, backend="thread")
    c = _rf_pred(data, n_jobs=4, backend="thread")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(b, c)


def test_cxx_differs_from_numpy_but_close():
    data = _tiny_panel()
    cxx = _rf_pred(data, n_jobs=1)
    npy = _rf_pred(data, n_jobs=1, cxx_feature_rng=False)
    assert not np.array_equal(cxx, npy)
    assert np.corrcoef(cxx.ravel(), npy.ravel())[0, 1] > 0.99


def test_numpy_fallback_bit_exact_across_backends():
    data = _tiny_panel()
    seq = _rf_pred(data, n_jobs=1, cxx_feature_rng=False)
    proc = _rf_pred(data, n_jobs=4, backend="process", cxx_feature_rng=False)
    np.testing.assert_array_equal(proc, seq)


# --- mGBT also benefits from cxx (base-tree feature) -----------------------

def _bdt_pred(data, **over):
    kw = dict(n_estimators_=20, lr=0.1, bagging_fraction=0.8, feature_fraction=0.5,
              samp_feature_by_node=True, max_depth=4, min_samples_leaf=5,
              n_discrete_lev=32, prebin=False, random_state=10,
              lambda_decay=1.0, objective_weights=[1, 0])
    b = BDT(data, **{**kw, **over})
    b.fit()
    return np.ascontiguousarray(b.predict(data.x), dtype=np.float32)


def test_mgbt_cxx_deterministic_and_differs_from_numpy():
    data = _tiny_panel()
    a = _bdt_pred(data)
    np.testing.assert_array_equal(a, _bdt_pred(data))
    npy = _bdt_pred(data, cxx_feature_rng=False)
    assert not np.array_equal(a, npy)
    assert np.corrcoef(a.ravel(), npy.ravel())[0, 1] > 0.95
