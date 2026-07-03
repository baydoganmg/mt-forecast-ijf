"""Per-node random objective-weights regression tests.

Covers the `obj_weight_mode` modes added per
notes/post_ijf_random_objective_weights.md:

    'fixed'     -- legacy default; caller-supplied objective_weights at every
                   node. Bit-exact to pre-change behavior.
    'dirichlet' -- per-node w ~ Dirichlet(alpha * ones(K)) draw.
    'onehot'    -- per-node uniform pick of exactly one objective.

Both new modes use the global numpy RNG via the cpp kernel (same path as
sample_features_np), so determinism follows from the M0 per-tree seeding
contract in `CTree_MT.fit`.
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

from mttrees.tree import DataMt, CTree_MT
from mttrees.ensemble import BDT, RF


def _tiny_panel(n_series=8, length=80, horizon=4, lag=6, seed=0):
    """Same synthetic panel as tests/test_phase2.py::_tiny_panel.

    Three objective groups (target / derivative / external are reduced to two
    here: target + derivative via n_derivative=1; no penalty_list so only
    target+derivative objective types are present). With n_derivative=1 the
    DataMt builds objective_types = ['target', 'derivative'], n_objective=2
    which still exercises the per-node Dirichlet/onehot resampling.
    """
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


def _fit_one(data, random_state=10, obj_weight_mode=None,
             obj_weight_dirichlet_alpha=1.0, max_depth=4):
    """Fit a single CTree_MT and return predictions on the training panel."""
    kwargs = dict(max_depth=max_depth, samp_frac=0.8, mtry=0.5,
                  min_samples_leaf=5, prebin=False, aggregation='score')
    if obj_weight_mode is not None:
        kwargs["obj_weight_mode"] = obj_weight_mode
        kwargs["obj_weight_dirichlet_alpha"] = obj_weight_dirichlet_alpha
    tree = CTree_MT(**kwargs)
    obj = [1, 1, 1][:data.n_objective] if hasattr(data, "n_objective") else [1, 1, 1]
    # arrangeObjective sets data.n_objective; build objective_weights to match.
    n_obj = len(data.objective_types)
    obj = [1.0] * n_obj
    tree.arrangeObjective(data, lambda_decay=None, objective_weights=obj)
    tree.fit(data.x, data.y, random_state=random_state)
    return tree.predict(data.x)


# -----------------------------------------------------------------------
# 1. BACK-COMPAT: fixed mode is bit-exact to the legacy default
# -----------------------------------------------------------------------

def test_fixed_mode_bit_exact_to_legacy():
    """Adding the obj_weight_mode kwarg with its default 'fixed' must not
    change predictions vs the prior code path (no kwarg). Bit-exact."""
    data = _tiny_panel()
    pred_legacy = _fit_one(data, random_state=10, obj_weight_mode=None)
    pred_fixed = _fit_one(data, random_state=10, obj_weight_mode='fixed')
    np.testing.assert_array_equal(pred_legacy, pred_fixed)


# -----------------------------------------------------------------------
# 2. DETERMINISM: same seed -> same result for each new mode
# -----------------------------------------------------------------------

def test_dirichlet_deterministic_same_seed():
    """Same random_state under obj_weight_mode='dirichlet' must give
    bit-exact predictions across two runs."""
    data = _tiny_panel()
    pred1 = _fit_one(data, random_state=10, obj_weight_mode='dirichlet')
    pred2 = _fit_one(data, random_state=10, obj_weight_mode='dirichlet')
    np.testing.assert_array_equal(pred1, pred2)


def test_onehot_deterministic_same_seed():
    """Same random_state under obj_weight_mode='onehot' must give
    bit-exact predictions across two runs."""
    data = _tiny_panel()
    pred1 = _fit_one(data, random_state=10, obj_weight_mode='onehot')
    pred2 = _fit_one(data, random_state=10, obj_weight_mode='onehot')
    np.testing.assert_array_equal(pred1, pred2)


# -----------------------------------------------------------------------
# 3. SANITY: new modes diverge from fixed mode (random sampling actually
#    affects the split scoring, not just a code path that quietly degenerates
#    to FIXED). Predictions must remain finite and correctly shaped.
# -----------------------------------------------------------------------

def test_dirichlet_differs_from_fixed():
    data = _tiny_panel()
    pred_fixed = _fit_one(data, random_state=10, obj_weight_mode='fixed')
    pred_dir = _fit_one(data, random_state=10, obj_weight_mode='dirichlet')
    assert pred_fixed.shape == pred_dir.shape
    assert np.all(np.isfinite(pred_dir))
    assert not np.array_equal(pred_fixed, pred_dir), (
        "Dirichlet predictions matched fixed bit-for-bit; per-node "
        "resampling does not appear to have taken effect.")


def test_onehot_differs_from_fixed():
    data = _tiny_panel()
    pred_fixed = _fit_one(data, random_state=10, obj_weight_mode='fixed')
    pred_oh = _fit_one(data, random_state=10, obj_weight_mode='onehot')
    assert pred_fixed.shape == pred_oh.shape
    assert np.all(np.isfinite(pred_oh))
    assert not np.array_equal(pred_fixed, pred_oh), (
        "Onehot predictions matched fixed bit-for-bit; per-node "
        "resampling does not appear to have taken effect.")


# -----------------------------------------------------------------------
# 4. MGBT / MRF integration smoke (kwargs flow through DT->CTree_MT)
# -----------------------------------------------------------------------

def test_bdt_with_dirichlet_runs():
    """BDT must accept obj_weight_mode via **kwargs (BDT->DT->CTree_MT) and
    produce finite predictions. Smoke test only — not a bit-exact check."""
    data = _tiny_panel()
    bdt = BDT(
        data,
        n_estimators_=8,
        lr=0.1,
        max_depth=3,
        bagging_fraction=1.0,
        feature_fraction=1.0,
        prebin=False,
        random_state=0,
        early_stopping_rounds=None,
        obj_weight_mode='dirichlet',
        obj_weight_dirichlet_alpha=1.0,
    )
    bdt.fit()
    pred = bdt.predict(data.x)
    assert np.all(np.isfinite(pred)), "non-finite values in mGBT dirichlet predictions"
    assert pred.shape == data.y.shape


def test_rf_with_onehot_runs():
    """RF must accept obj_weight_mode via **kwargs and produce finite
    predictions. Smoke test only."""
    data = _tiny_panel()
    rf = RF(
        data,
        n_estimators_=8,
        bagging_fraction=0.8,
        feature_fraction=0.5,
        max_depth=4,
        prebin=False,
        random_state=10,
        n_jobs=1,
        mmap_data=False,
        obj_weight_mode='onehot',
        keep_train_predictions=True,  # asserts on final_predictions, which is
        # opt-in since dc23254 (default OFF; OOB groundwork).
    )
    rf.fit()
    assert np.all(np.isfinite(rf.final_predictions))
    assert rf.final_predictions.shape == data.y.shape
