"""Regression tests for the random_state-missing bug fixed in commit 7a9760d.

Without these tests, the bug pattern is easy to reintroduce: any new caller
of RF(...) or BDT(...) that forgets random_state silently falls through to
CTree_MT.fit's hard-coded bagging_seed=5 fallback AND skips np.random.seed,
making per-tree fits depend on numpy's entropy-initialized global RNG —
different across Python invocations.

Phase 2 test_mgbt_use_indirect_is_bit_exact_to_default verified bit-exact
behavior at tiny scale within a single process, but it doesn't catch the
cross-invocation entropy drift that this test guards against.
"""

import numpy as np
import pandas as pd
import pytest

from mttrees.tree import DataMt
from mttrees.ensemble import RF, BDT


def _panel(n_series=8, length=80, horizon=4, lag=6, seed=0):
    """Same pattern as tests/test_phase2.py::_tiny_panel."""
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


def test_rf_bit_deterministic_within_process():
    """Two RF fits with the same random_state must give bit-exact predictions."""
    data = _panel()
    # keep_train_predictions=True: final_predictions is opt-in (default OFF;
    # it is OOB-prediction groundwork, not a current deliverable). These
    # tests assert on it, so they enable it explicitly.
    common = dict(n_estimators_=16, bagging_fraction=0.8, feature_fraction=0.5,
                  max_depth=8, prebin=False, random_state=10,
                  n_jobs=1, mmap_data=False, keep_train_predictions=True)
    rf1 = RF(data, **common); rf1.fit()
    rf2 = RF(data, **common); rf2.fit()
    np.testing.assert_array_equal(rf1.final_predictions, rf2.final_predictions)


def test_bdt_bit_deterministic_within_process():
    """Two BDT fits with the same random_state must give bit-exact predictions."""
    data = _panel()
    common = dict(n_estimators_=16, lr=0.05, bagging_fraction=0.8,
                  feature_fraction=0.5, max_depth=4, prebin=False,
                  random_state=10, early_stopping_rounds=None)
    bdt1 = BDT(data, **common); bdt1.fit()
    bdt2 = BDT(data, **common); bdt2.fit()
    np.testing.assert_array_equal(bdt1.predict(data.x), bdt2.predict(data.x))


def test_rf_n_jobs_1_vs_2_bit_exact():
    """Same seed, different n_jobs, same answer (joblib doesn't change result)."""
    data = _panel()
    common = dict(n_estimators_=16, bagging_fraction=0.8, feature_fraction=0.5,
                  max_depth=8, prebin=False, random_state=10, mmap_data=False,
                  keep_train_predictions=True)
    rf1 = RF(data, n_jobs=1, **common); rf1.fit()
    rf2 = RF(data, n_jobs=2, **common); rf2.fit()
    np.testing.assert_array_equal(rf1.final_predictions, rf2.final_predictions)


@pytest.mark.parametrize("n_jobs,mmap", [(1, False), (2, False), (2, True)])
def test_rf_keep_train_predictions_flag_is_forecast_bit_exact(n_jobs, mmap):
    """The keep_train_predictions flag must NOT change the fitted forest:
    flag OFF (default) skips computing/shipping each tree's train-prediction
    array, but the trees themselves are fit identically (predict() does not
    touch the RNG), so rf.predict(X) is byte-identical ON vs OFF. OFF also
    leaves final_predictions as None (honest 'not computed' signal). Covers
    sequential, parallel, and parallel+mmap dispatch paths."""
    data = _panel()
    common = dict(n_estimators_=16, bagging_fraction=0.8, feature_fraction=0.5,
                  max_depth=8, prebin=False, random_state=10,
                  n_jobs=n_jobs, mmap_data=mmap)
    rf_on = RF(data, keep_train_predictions=True, **common); rf_on.fit()
    rf_off = RF(data, keep_train_predictions=False, **common); rf_off.fit()
    assert rf_off.final_predictions is None
    assert rf_on.final_predictions is not None
    np.testing.assert_array_equal(rf_on.predict(data.x), rf_off.predict(data.x))
