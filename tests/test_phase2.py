"""
Phase 2 committee-fix regression tests.

Each test pins one specific fix from the 3-specialist audit. Tests are written
red-first (added at the same commit as the fix), and run on synthetic data so
they do not depend on the larger data pipeline or external datasets.
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
import pytest

from mttrees.tree import DataMt


def _tiny_panel(n_series=8, length=80, horizon=4, lag=6, seed=0):
    """A handful of noisy sinusoids in DataMt form. Smallest viable training
    set for an ensemble: large enough that ES will actually trigger before the
    400-tree default but small enough to fit in <1s."""
    rng = np.random.default_rng(seed)
    feats = []
    targs = []
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
                  penalty_list=None,
                  series_means=sm,
                  int_convert=False)
    data.transform(feat_df, targ_df)
    return data


# -----------------------------------------------------------------------
# Fix #1 — ML committee finding "n_estimators_ stale after early stopping"
# ensemble.py:528-529 truncates self.trees but leaves self.n_estimators_
# at its original 400 (or whatever was passed in), so
# len(bdt.trees) != bdt.n_estimators_. The eval_refit_on_full branch already
# fixes this; the more common eval_refit_on_full=False path does not.
# -----------------------------------------------------------------------

def test_mgbt_n_estimators_matches_trees_after_early_stopping():
    """After ES truncates self.trees to best_iter+1, self.n_estimators_ must
    reflect that. Otherwise downstream code that inspects bdt.n_estimators_
    (logging, plotting, refit-budget calculation) sees a stale value."""
    from mttrees.ensemble import BDT

    data = _tiny_panel()
    # n_estimators_ generous so ES has room to trigger; ES rounds tight so it
    # fires inside the budget on synthetic data.
    bdt = BDT(
        data,
        n_estimators_=200,
        max_depth=3,
        lr=0.1,
        bagging_fraction=1.0,
        feature_fraction=1.0,
        early_stopping_rounds=5,
        eval_k=1,
        eval_metric="mse",
        eval_refit_on_full=False,
        random_state=0,
    )
    bdt.fit()

    # Sanity: ES should fire on this tiny noisy problem within the 200 budget.
    assert bdt.best_iteration_ < 199, (
        "ES did not trigger; test cannot validate the fix. Increase noise or "
        "tighten early_stopping_rounds.")
    assert len(bdt.trees) == bdt.best_iteration_ + 1, (
        "trees list was not truncated to best_iter+1; the truncation step "
        "itself is broken — fix that before re-running this regression.")

    # The actual regression: n_estimators_ must equal len(trees).
    assert bdt.n_estimators_ == len(bdt.trees), (
        f"stale n_estimators_={bdt.n_estimators_} but len(trees)={len(bdt.trees)}; "
        f"BDT._fit_with_early_stopping must update self.n_estimators_ after "
        f"truncation in the eval_refit_on_full=False branch.")


# -----------------------------------------------------------------------
# Fix #2 — SWE committee finding "loky pool persists across datasets"
# joblib.externals.loky keeps a reusable executor alive for the process
# lifetime. After mRF.fit() on a heavy dataset, the worker procs retain
# their high-water-mark RSS until the parent exits. On Mac (jetsam) this
# is the dominant trigger of silent kills mid-benchmark.
# Contract: mttrees.utils must expose `shutdown_worker_pool()` that
#   1) is safe to call when no pool exists yet,
#   2) actually terminates the current pool (new Parallel call gets new PIDs),
#   3) leaves joblib fully usable for subsequent Parallel calls.
# -----------------------------------------------------------------------

def _worker_pid():
    """Trivial joblib task that returns the worker's PID. Module-level so
    loky can pickle it for delivery to workers."""
    return os.getpid()


def test_shutdown_worker_pool_recycles_loky_workers():
    """After shutdown_worker_pool(), a fresh Parallel call must spawn new
    workers (different PIDs from before). This is the regression that
    prevents Mac RSS from accumulating across datasets."""
    pytest.importorskip("joblib")
    from joblib import Parallel, delayed

    # First Parallel call: warm the pool, capture worker PIDs.
    pids_before = set(Parallel(n_jobs=2, backend="loky")(
        delayed(_worker_pid)() for _ in range(8)))
    assert len(pids_before) >= 1, "expected at least one loky worker PID"

    # Import the helper — this is the contract test for the fix.
    from mttrees.utils import shutdown_worker_pool
    shutdown_worker_pool()

    # Calling it a second time must be safe (idempotent: shutdown of no pool).
    shutdown_worker_pool()

    # Second Parallel call: pool must be re-created with fresh worker PIDs.
    pids_after = set(Parallel(n_jobs=2, backend="loky")(
        delayed(_worker_pid)() for _ in range(8)))
    assert len(pids_after) >= 1, "joblib broken after shutdown_worker_pool()"
    assert pids_before.isdisjoint(pids_after), (
        f"loky pool NOT actually shut down: PIDs reused across the boundary "
        f"({pids_before} ∩ {pids_after}). shutdown_worker_pool() is a no-op.")


# -----------------------------------------------------------------------
# Fix #5 — ML committee finding "BDT never engages the cpp indirect path"
# RF.fit plumbs use_indirect=self.indirect_fit into the worker, but
# BDT.fit (both ES and non-ES branches) calls dt.fit(...) without the
# kwarg, so DT.fit's default use_indirect=False wins. Result: mGBT pays
# the per-tree gather cost even when the user opted in.
# Contract: BDT(use_indirect=True).fit() must be BIT-EXACT to
# BDT(use_indirect=False).fit() — the indirect path is a memory/time
# optimisation, not a different algorithm.
# -----------------------------------------------------------------------

def _bdt_kwargs(use_indirect):
    return dict(
        n_estimators_=8,
        max_depth=3,
        lr=0.1,
        bagging_fraction=0.8,
        feature_fraction=0.7,
        prebin=False,          # indirect path requires selfhash (no prebin)
        early_stopping_rounds=None,  # vanilla path, no ES branch
        boost_from_average=True,
        random_state=42,
        use_indirect=use_indirect,
    )


def test_mgbt_use_indirect_is_bit_exact_to_default():
    """When user opts into the cpp indirect-fit path on mGBT, the predictions
    must match the default (gather) path BIT-EXACTLY at fp32 precision. If
    they differ, either the plumbing is wrong or the indirect kernel has a
    silent algorithmic divergence."""
    from mttrees.ensemble import BDT

    data_a = _tiny_panel(seed=1)
    data_b = _tiny_panel(seed=1)  # identical inputs

    bdt_off = BDT(data_a, **_bdt_kwargs(use_indirect=False))
    bdt_off.fit()
    bdt_on  = BDT(data_b, **_bdt_kwargs(use_indirect=True))
    bdt_on.fit()

    # final_predictions is the gold reference: same trees → same per-row preds.
    np.testing.assert_array_equal(
        bdt_off.final_predictions, bdt_on.final_predictions,
        err_msg="BDT(use_indirect=True) diverges from default on training preds. "
                "Either the plumbing is incomplete (use_indirect not threaded "
                "from BDT.__init__ through BDT.fit -> DT.fit) or the cpp "
                "indirect kernel disagrees with the gather kernel.")

    assert bdt_on.n_estimators_ == bdt_off.n_estimators_ == 8


# -----------------------------------------------------------------------
# Fix #6 — ML committee finding "selfhash left-child guard at fit.cpp:226"
# In the bin-scan loop, the left-child min_data_in_leaf check is gated by
# an inner test `cumul_count + bin_instance_count[b+1] < n_sample`. When
# the next bin contains the rest of the rows, the inner condition is
# false and the `continue` is skipped — the split candidate is emitted
# with cumul_count < min_data_in_leaf. The right-child guard at line 236
# catches the other side, but the left-child path is broken.
#
# Tie-heavy synthetic case: a feature where 90% of rows share one value
# collapses quantile bins to a single dominant bin. Splitting on any
# other bin produces a left child below min_samples_leaf, exercising
# the buggy path.
# -----------------------------------------------------------------------

def _count_bad_children(tree_info, min_leaf):
    bad = 0
    for k in range(len(tree_info)):
        status = tree_info[k, 2]
        if status != -3:  # NODE_INTERIOR
            continue
        left_id = int(tree_info[k, 5])
        right_id = int(tree_info[k, 6])
        if 0 <= left_id < len(tree_info) and 0 <= right_id < len(tree_info):
            lcount = int(tree_info[left_id, 1])
            rcount = int(tree_info[right_id, 1])
            if lcount < min_leaf or rcount < min_leaf:
                bad += 1
    return bad


@pytest.mark.parametrize("bin_mode", ["adaptive", "fixed_valid"])
def test_selfhash_min_data_in_leaf_respected_on_tieheavy(bin_mode):
    """No interior selfhash node may have a child with count < min_samples_leaf.

    Reproduces the buggy-guard scenario via a 90%-tied feature; the inner
    `cumul_count + bin_instance_count[b+1] < n_sample` test in the bin scan
    fails to skip splits whose left child is below the leaf-size contract."""
    os.environ["MT_CPP_BIN_MODE"] = bin_mode
    from mttrees.tree import CTree_MT, DataMt

    rng = np.random.default_rng(0)
    n = 400
    k_target = 4
    x0 = np.where(rng.random(n) < 0.9, 0.0, rng.uniform(1.0, 2.0, n))
    x1 = rng.standard_normal(n).astype(np.float32)
    x2 = rng.standard_normal(n).astype(np.float32)
    x3 = np.where(rng.random(n) < 0.85, 1.5, rng.uniform(-1.0, 1.0, n))
    feats = pd.DataFrame({
        "lag_1": x0.astype(np.float32), "lag_2": x1, "lag_3": x2,
        "lag_4": x3.astype(np.float32)})
    targets = rng.standard_normal((n, k_target)).astype(np.float32)
    targets[:, 0] += 0.3 * x1
    targets[:, 1] += 0.2 * x2
    targets_df = pd.DataFrame(targets,
                              columns=[f"ahead_{i}" for i in range(k_target)])
    series_means = pd.DataFrame({"series": [0], "mean_series": [1.0]})

    data = DataMt(max_ahead=k_target, n_derivative=0, penalty_list=None,
                  series_means=series_means, int_convert=False)
    data.x = feats
    data.y = targets_df
    data.y_org = targets[:, :k_target].astype(np.float64)
    data.index = pd.DataFrame({"series": np.zeros(n, dtype=int),
                               "index": np.arange(n)})
    # Bypass .transform() because we built x/y by hand; populate the
    # objective metadata the kernel inspects.
    target_cols = [f"ahead_{i}" for i in range(k_target)]
    data.target_col_list = [target_cols]
    data.target_type_list = ["ahead"]
    data.flat_target_cols = target_cols
    data.objective_types = ["ahead"]

    min_leaf = 5
    tree = CTree_MT(max_depth=10, samp_frac=1.0, mtry=1.0,
                    samp_feature_by_node=True, min_samples_leaf=min_leaf,
                    n_discrete_lev=16, num_threads=1, verbose=0,
                    prebin=False, aggregation="rank")
    tree.arrangeObjective(data, lambda_decay=0.5,
                          objective_weights=[1.0] * k_target)
    tree.fit(data.x, data.y, random_state=42)

    bad = _count_bad_children(np.asarray(tree.tree_info), min_leaf)
    assert bad == 0, (
        f"selfhash cpp produced {bad} interior nodes with at least one child "
        f"below min_samples_leaf={min_leaf} (bin_mode={bin_mode}). Inner "
        f"left-child guard at fit.cpp:231 fails to skip splits where the "
        f"next bin would consume all remaining rows.")


# -----------------------------------------------------------------------
# Fix #7 — SWE committee finding "thread_local ScanWorkspace lifetime"
# The per-feature scan workspace is `static thread_local`, so its
# capacity grows to the largest n_sample ever seen on that thread and
# never shrinks. Across heterogeneous datasets that's a hold of
# `num_threads * 12 bytes * max(n_sample)` per worker process. The fix
# is an adaptive shrink at the end of fit_mt_tree_bin that frees memory
# only when the held-over capacity is >> the current fit's root size.
# Hot-path cost is one capacity comparison; cold-path cost is one free.
# -----------------------------------------------------------------------

def _make_synth_fit_input(n_sample, n_features=4, k_target=4, seed=0):
    """Construct the raw ndarrays that fit_mt_tree_bin takes. Avoids the
    DataMt wrapper so the test can scale n_sample directly."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_sample, n_features)).astype(np.float32)
    labels = rng.standard_normal((n_sample, k_target)).astype(np.float32)
    weights_target = np.ones(k_target, dtype=np.float32)
    target_weight_map = np.zeros(k_target, dtype=np.int32)
    objective_weights = np.ones(1, dtype=np.float32)
    return features, labels, weights_target, target_weight_map, objective_weights


def test_scanworkspace_shrinks_after_large_then_small_fit():
    """A large fit grows the per-thread workspace; a subsequent small fit
    must free the held-over capacity. Without the adaptive shrink at the
    end of fit_mt_tree_bin, the workspace stays at the large size forever
    and accumulates across datasets — the dominant on-Mac jetsam trigger
    once num_threads > 1."""
    from mttrees.cfuncs_cpp import mt_ctree_selfhash_f32 as cpp

    common_kwargs = dict(
        n_objective=1,
        max_depth=6,
        mtry=1.0,
        samp_feat_by_node=0,
        min_data_in_leaf=5,
        num_nodes=128,
        min_gain_to_split=0.0,
        num_bins=16,
        num_threads=1,
        verbose=0,
        agg_mode=1,
    )

    # 1. Large fit grows the workspace.
    feat_big, lab_big, wt, twm, ow = _make_synth_fit_input(n_sample=80_000)
    cpp.fit_mt_tree_bin(feat_big, lab_big, wt, twm,
                       objective_weights=ow, **common_kwargs)
    cap_after_big = cpp.ws_capacity_bytes()
    assert cap_after_big > 100_000, (
        f"Sanity: workspace should have grown after the 80k-row fit. "
        f"Got {cap_after_big} bytes. Either the fit was skipped or the "
        f"introspection helper is wrong.")

    # 2. Small fit. With the fix the workspace must shrink back close to
    #    the small fit's footprint.
    feat_small, lab_small, _, _, _ = _make_synth_fit_input(n_sample=200,
                                                            seed=1)
    cpp.fit_mt_tree_bin(feat_small, lab_small, wt, twm,
                       objective_weights=ow, **common_kwargs)
    cap_after_small = cpp.ws_capacity_bytes()

    # The adaptive guard uses 2x current N as the threshold; the small fit
    # has N=200 so the trigger is capacity > 400 elements. Anything left at
    # the big-fit scale (80k elements * 12 bytes ≈ 1 MB) would fail this.
    assert cap_after_small < cap_after_big / 4, (
        f"Workspace did NOT shrink after small fit: cap_before={cap_after_big}B, "
        f"cap_after={cap_after_small}B. The adaptive shrink at the end of "
        f"fit_mt_tree_bin is not firing — high-water-mark hold persists "
        f"across heterogeneous fits.")


# -----------------------------------------------------------------------
# Fix #8 — Forecaster committee finding "MASE seasonal-scale H-1 leak"
# The benchmark drops only test_horizon=1 raw observation per series
# when computing the seasonal scale, but the actual test set is the LAST
# horizon=H raw observations of each series. The H-1 raw test rows
# between (N-H) and (N-2) leak into the scale denominator, breaking the
# Monash MASE contract:
#   https://github.com/rakshitha123/TSForecasting utils/error_calculator.R
# uses mean(abs(diff(training_only_y, lag=m))), where training_only_y is
# the series MINUS the last horizon observations.
# -----------------------------------------------------------------------

def _make_clean_seasonal_panel(n_series=3, length=100, season=12, horizon=12):
    """Series whose training portion (first length-horizon obs) is a clean
    period-`season` sinusoid (zero seasonal-naive scale), and whose test
    portion (last `horizon` obs) is perturbed by a large constant offset.

    Monash MASE: training is clean → scale = 0 → falls back to lag-1 = 0.
    H-1 leak:    drops only 1 raw row → scale denominator includes the
                 perturbed test rows → non-zero scale.
    """
    rows = []
    for s in range(n_series):
        for t in range(length):
            y = float(np.sin(2 * np.pi * t / season))
            if t >= length - horizon:
                # Test horizon perturbation: shift by +10. Without leakage
                # the scale ignores these rows entirely.
                y += 10.0
            rows.append({"series": f"s{s}", "index": t, "y": y})
    return pd.DataFrame(rows)


def test_seasonal_scale_drops_full_horizon():
    """compute_seasonal_scale must drop the FULL forecast horizon from each
    series, not just the wide-format test_horizon. Otherwise the test-row
    perturbation leaks into the scale denominator."""
    from mttrees.utils import compute_seasonal_scale

    horizon = 12
    max_seasonality = 12
    panel = _make_clean_seasonal_panel(
        n_series=3, length=100, season=12, horizon=horizon)

    # Drop the full horizon: Monash-style. Training portion is clean
    # sinusoidal so seasonal diff at lag=12 is exactly 0 for every row,
    # the lag-1 fallback also gives small but non-zero (sin slope), so the
    # final scale is the lag-1 fallback (positive, but small).
    scale_correct = compute_seasonal_scale(
        panel, "y", drop_last=horizon, max_seasonality=max_seasonality)
    assert (scale_correct < 1.0).all(), (
        f"Sanity: with horizon dropped, scale should fall back to the lag-1 "
        f"slope of a clean sinusoid (< 1.0 in magnitude). Got {scale_correct}")

    # Drop only the wide-format test_horizon=1: H-1 = 11 perturbed rows leak
    # into the scale. The +10 perturbation produces seasonal diffs of
    # magnitude 10, dominating the average.
    scale_leaky = compute_seasonal_scale(
        panel, "y", drop_last=1, max_seasonality=max_seasonality)

    # The leaky scale should clearly diverge from the correct scale.
    # Concretely: 11 seasonal-diff entries of magnitude ~10, 76 of magnitude
    # ~0 → mean magnitude ≈ 11*10 / 87 ≈ 1.26. Far from correct (~ lag-1).
    assert (scale_leaky > scale_correct + 1.0).all(), (
        f"H-1 leak NOT detected: leaky scale {scale_leaky.tolist()} "
        f"vs correct {scale_correct.tolist()}. compute_seasonal_scale must "
        f"use drop_last=horizon, not drop_last=test_horizon.")
