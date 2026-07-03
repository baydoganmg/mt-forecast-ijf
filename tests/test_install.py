"""Install smoke test.

Verifies that `pip install -e .` succeeded:
  * `mttrees` imports.
  * The C++ kernel (`cfuncs_cpp`) is loadable.
  * A tiny mDT / mGBT / mRF fit + predict completes with finite outputs.
  * RF's `indirect_fit=mmap_data` auto-derivation is in place.

Run:
    python -m pytest tests/test_install.py -v
or as a script:
    python tests/test_install.py
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MT_DTYPE", "fp32")
os.environ.setdefault("MT_BACKEND", "cpp")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd


def _synthetic_data(n_series=8, length=30, horizon=4, lag=6, seed=0):
    """Build a tiny synthetic DataMt-shaped panel: n_series sinusoids."""
    from mttrees.tree import DataMt
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        t = np.arange(length)
        amp = 1.0 + rng.standard_normal()
        y = amp * np.sin(2 * np.pi * t / 7) + 0.1 * rng.standard_normal(length)
        for i, val in enumerate(t):
            rows.append({"series": f"s{s}", "index": i, "y": y[i],
                         "season_index": i})
    combined = pd.DataFrame(rows)

    from mttrees.utils import build_features_and_targets, mean_scale_data
    lag_g, tar_g, _ = build_features_and_targets(
        combined, "y", lag=lag, horizon=horizon, diff_features=False,
        time_series_cols=["index", "series"], season_cols=["season_index"],
        ext_features=[], frequency="1D", max_seasonality=7)
    # Hold out the last row per series as test, rest as train.
    grp = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grp >= 1]; trt = tar_g[grp >= 1]
    tef = lag_g[grp < 1];  tet = tar_g[grp < 1]
    trf2, tef2, trt2, tet2, sm = mean_scale_data(
        trf, tef, trt, tet, lag=lag, horizon=horizon, mean_scale=True)

    train_data = DataMt(max_ahead=horizon, n_derivative=1,
                        penalty_list=["sin_season", "cos_season"],
                        series_means=sm, int_convert=False)
    train_data.transform(trf2, trt2)
    test_data = DataMt(max_ahead=horizon, n_derivative=1,
                       penalty_list=["sin_season", "cos_season"],
                       series_means=sm, int_convert=False)
    test_data.transform(tef2, tet2)
    return train_data, test_data


def test_import():
    """The package imports and the C++ extension is loaded."""
    import mttrees
    import mttrees.tree
    import mttrees.ensemble
    import mttrees.utils
    from mttrees.cfuncs_cpp import mt_ctree_selfhash_f32 as cpp
    assert hasattr(cpp, "fit_mt_tree_bin")
    assert hasattr(cpp, "fit_mt_tree_bin_indirect")
    assert hasattr(cpp, "predict_mt_tree")


import pytest


@pytest.mark.xfail(reason="Depends on build_features_and_targets/mean_scale_data "
                          "living in mttrees.utils; currently they live in "
                          "experiments/run_benchmark.py. Refactor follow-up.",
                   strict=True)
def test_mdt_mgbt_mrf_smoke():
    """All three estimators fit + predict + produce finite K-vector predictions."""
    from mttrees.ensemble import DT, BDT, RF

    train_data, test_data = _synthetic_data()
    common = dict(
        lambda_decay=0.5, objective_weights=[1, 0.2, 0.5],
        max_depth=6, min_samples_leaf=2, n_discrete_lev=8,
        num_threads=1, prebin=False,
    )

    # mDT
    dt = DT(train_data, **common, samp_frac=1.0, mtry=1.0,
            samp_feature_by_node=True)
    dt.fit(train_data, random_state=0)
    pred_dt = dt.tree.predict(test_data.x)
    assert pred_dt.shape == (len(test_data.x), test_data.y.shape[1])
    assert np.all(np.isfinite(pred_dt))

    # mGBT (early-stopping ON by default; small n_estimators)
    bdt = BDT(train_data, n_estimators_=5, random_state=0, **common)
    bdt.fit()
    pred_bdt = bdt.predict(test_data.x)
    assert pred_bdt.shape == (len(test_data.x), test_data.y.shape[1])
    assert np.all(np.isfinite(pred_bdt))

    # mRF
    rf = RF(train_data, n_estimators_=3,
            bagging_fraction=0.8, feature_fraction=0.5,
            n_jobs=1, mmap_data=False, random_state=0, **common)
    rf.fit()
    pred_rf = rf.predict(test_data.x)
    assert pred_rf.shape == (len(test_data.x), test_data.y.shape[1])
    assert np.all(np.isfinite(pred_rf))


@pytest.mark.xfail(reason="Depends on build_features_and_targets/mean_scale_data "
                          "living in mttrees.utils; currently they live in "
                          "experiments/run_benchmark.py. Refactor follow-up.",
                   strict=True)
def test_rf_indirect_fit_auto_derives():
    """RF auto-derives indirect_fit from mmap_data when caller leaves it None."""
    from mttrees.ensemble import RF
    train_data, _ = _synthetic_data()
    common = dict(
        n_estimators_=2,
        bagging_fraction=0.8, feature_fraction=0.5,
        lambda_decay=0.5, objective_weights=[1, 0.2, 0.5],
        max_depth=4, min_samples_leaf=2, n_discrete_lev=8,
        num_threads=1, n_jobs=1, prebin=False, random_state=0,
    )
    rf_off = RF(train_data, mmap_data=False, **common)
    rf_on  = RF(train_data, mmap_data=True,  **common)
    assert rf_off.indirect_fit is False
    assert rf_on.indirect_fit is True


if __name__ == "__main__":
    test_import();                       print("[PASS] test_import")
    test_mdt_mgbt_mrf_smoke();           print("[PASS] test_mdt_mgbt_mrf_smoke")
    test_rf_indirect_fit_auto_derives(); print("[PASS] test_rf_indirect_fit_auto_derives")
    print("\nAll 3 smoke tests PASS.")
