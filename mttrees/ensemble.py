import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from . import tree as mt
# Route precompute_global_bins through tree's dispatch, so fp32 builds use the
# fp32 kernel automatically (controlled by MT_DTYPE env var).
from .tree import precompute_global_bins, MT_DTYPE


def _fit_one_rf_tree(seed_i, data, lambda_decay, objective_weights, kwargs,
                     bins, use_indirect=False, compute_pred=True):
    """Worker for joblib RF parallel fit (top-level so loky can pickle it).

    Returns (DT instance with strip_fit_data applied, predictions on data.x).
    Each worker receives its per-tree seed under the M0 RNG contract; this
    guarantees that with n_jobs=1 the result bit-matches the sequential RF.fit,
    and that with n_jobs>1 each tree is reproducible independently of
    worker-scheduling order.

    use_indirect: when True, CTree_MT.fit takes the no-gather cpp indirect
    path (cfuncs_cpp.fit_mt_tree_bin_indirect). Default False keeps the
    historical gather path. Only meaningful when prebin=False (selfhash).
    """
    dt = DT(data, lambda_decay=lambda_decay, objective_weights=objective_weights,
            **kwargs)
    dt.fit(data, precomputed_bins=bins, random_state=seed_i,
           use_indirect=use_indirect)
    # compute_pred=False (RF default): skip the train-prediction array so the
    # worker ships back only the fitted tree, not the N x K fp32 pred (the
    # dominant mRF orchestration cost). predict() does NOT touch the global
    # RNG, so skipping it leaves every subsequent draw / tree bit-identical.
    pred = dt.predictions(data) if compute_pred else None
    dt.tree.strip_fit_data()
    return dt, pred


def _fit_one_rf_tree_memmap(seed_i, x_arr, y_arr, x_columns,
                            x_index, y_columns, y_index,
                            objective_types, target_type_list, target_col_list,
                            lambda_decay, objective_weights, kwargs, bins,
                            use_indirect=True, compute_pred=True):
    """Memmap variant: receives numpy arrays at the top level so joblib
    auto-memmaps them across workers (shared physical memory) instead of
    pickling a per-worker copy. Worker reconstructs zero-copy pandas
    DataFrames from the memmapped arrays, preserving the original index so
    the bagging label-based `.sample(...).index` + `.loc[...]` pattern in
    tree.fit selects the same rows as the non-memmap path.

    NOTE: y_org is intentionally NOT passed to workers. RF tree fit /
    predict only consume data.x and data.y; y_org is used by tune_penalties
    for forecast evaluation, which runs in the parent.

    The worker also receives MINIMAL objective metadata
    (objective_types/target_type_list/target_col_list) instead of a
    stripped DataMt clone. the mmap-design review
    items #4 and #5 — avoids shipping DataMt.index (~15 MB on Rossmann)
    and other unused fields to every worker.
    """
    import pandas as pd
    import types
    x_df = pd.DataFrame(x_arr, columns=x_columns, index=x_index, copy=False)
    y_df = pd.DataFrame(y_arr, columns=y_columns, index=y_index, copy=False)
    data = types.SimpleNamespace(
        x=x_df, y=y_df, y_org=None,
        objective_types=objective_types,
        target_type_list=target_type_list,
        target_col_list=target_col_list,
    )
    dt = DT(data, lambda_decay=lambda_decay, objective_weights=objective_weights,
            **kwargs)
    dt.fit(data, precomputed_bins=bins, random_state=seed_i,
           use_indirect=use_indirect)
    pred = dt.predictions(data) if compute_pred else None
    dt.tree.strip_fit_data()
    return dt, pred


class DT:
    def __init__(self, data, lambda_decay=None, objective_weights=None, **kwargs):
        self.tree = mt.CTree_MT(**kwargs)
        self.tree.arrangeObjective(data, lambda_decay, objective_weights)

    def fit(self, data, precomputed_bins=None, random_state=None,
            use_indirect=False):
        # use_indirect plumbed through from RF.fit (mmap workers set True).
        # mDT and mGBT call DT.fit without this kwarg, so the indirect path
        # stays gated off for them. Default False preserves backward compat.
        self.tree.fit(data.x, data.y,
                      precomputed_bins=precomputed_bins,
                      random_state=random_state,
                      use_indirect=use_indirect)

    def predictions(self, data):
        return self.tree.predict(data.x)


class BDT:
    """Multi-target boosted decision trees (mGBT).

    Defaults are aligned with the production recommendation from the IJF
    revision experiments:
      lr=0.05, bagging_fraction=0.8, feature_fraction=0.5, prebin=False
        ... match the per-tree hyperparams used in the published Table:MASE.
      early_stopping_rounds=40, eval_k=1, eval_metric='mse', min_delta=0.0
        ... enable early stopping by default on the last row per series
        (mirrors test_horizon=1 holdout shape). Patience=20 is the standard
        boosting-library default.

    Recommended production call (matches the validated 8-dataset comparison
    in `experiments/compare_mgbt_paper_vs_rule_es_8ds.py`):
      BDT(data, n_estimators_=400, max_depth=min(mDT_CV_depth, 8),
          eval_metric='mase', eval_seasonal_scale=<per-series-Series>, ...)
    The depth rule and 'mase' metric require caller-supplied inputs
    (mDT CV result, seasonal scale Series) so they remain caller
    conventions rather than baked-in defaults.

    Set `early_stopping_rounds=None` to disable early stopping and
    restore the pre-revision behavior.
    """

    def __init__(self, data, n_estimators_,
                 lr=0.05, bagging_fraction=0.8, feature_fraction=0.5,
                 samp_feature_by_node=True,
                 boost_from_average=True, lambda_decay=None, objective_weights=None,
                 keep_loss=False, prebin=False, random_state=None,
                 early_stopping_rounds=40, eval_k=1,
                 eval_refit_on_full=False, min_delta=0.0,
                 eval_metric="mse", eval_seasonal_scale=None,
                 use_indirect=False,
                 **kwargs):

        self.data = copy.copy(data)
        self.data_ = copy.copy(data)
        self.n_estimators_ = n_estimators_
        self.lr = lr
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.samp_feature_by_node = samp_feature_by_node
        self.boost_from_average = boost_from_average
        self.lambda_decay = lambda_decay
        self.objective_weights = objective_weights
        self.prebin = prebin
        self.keep_loss = keep_loss
        self.random_state = random_state
        # Early stopping (mGBT only). Default early_stopping_rounds=40
        # enables ES on the last row per series (eval_k=1) with MSE on the
        # mean-scaled targets. Pass early_stopping_rounds=None to disable
        # entirely and restore the pre-revision behavior.
        #   eval_k: number of trailing rows per series to hold out as eval
        #           (defaults to data.max_ahead = forecast horizon).
        #   eval_refit_on_full: when True, after finding best_iter on
        #           (train, eval), refit on (train + eval) with that count
        #           fixed. Doubles wall but uses the eval signal in the
        #           final model. Default False truncates to best_iter only.
        #   min_delta: minimum absolute MSE improvement required to count as
        #           "improved". Matches LightGBM's `min_delta` (3.0+). With
        #           noisy validation, late-iter MSE can wiggle without real
        #           progress (e.g. 0.0652 -> 0.0651 -> 0.0652). Setting
        #           min_delta > 0 ignores those wiggles. Default 0.0 keeps
        #           the strict "any improvement" rule (only the 1e-12 FP
        #           noise guard remains).
        #   eval_metric: "mse" (default) or "mase".
        #     mse  -- raw mean-squared-error on mean-scaled targets. Cheap,
        #             no extra inputs required. The default.
        #     mase -- row-wise mean absolute scaled error matching the
        #             test-time reporting metric. Requires
        #             eval_seasonal_scale; unscales predictions via
        #             data.series_means then divides per-row MAE by the
        #             per-series seasonal-naive error.
        #   eval_seasonal_scale: pandas Series indexed by series id, value
        #     = mean seasonal-naive |y_t - y_{t-S}| over training. Required
        #     when eval_metric == "mase"; ignored otherwise. Same source as
        #     the test-eval scale, so eval-MASE and test-MASE share units.
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_k = eval_k
        self.eval_refit_on_full = eval_refit_on_full
        self.min_delta = float(min_delta)
        self.eval_metric = str(eval_metric).lower()
        # Indirect cpp fit (no-gather, int32 row positions). Default off
        # preserves the historical gather path. Only meaningful when
        # prebin=False (selfhash). Bit-equivalent by contract — see
        # tests/test_phase2.py::test_mgbt_use_indirect_is_bit_exact_to_default.
        self.use_indirect = bool(use_indirect)
        if self.eval_metric not in ("mse", "mase"):
            raise ValueError(
                f"eval_metric must be 'mse' or 'mase', got {eval_metric!r}")
        self.eval_seasonal_scale = eval_seasonal_scale
        self.best_iteration_ = None
        # best_eval_mse_ retained as attribute name for backward compat; it
        # actually holds whichever metric is in use (MSE or MASE).
        self.best_eval_mse_ = None
        self.eval_history_ = []

        given_params = kwargs.copy()
        given_params['samp_frac'] = bagging_fraction
        given_params['mtry'] = feature_fraction
        given_params['samp_feature_by_node'] = samp_feature_by_node
        given_params['prebin'] = prebin
        # mGBT benefits from the pure-C++ per-tree feature RNG (GIL-free, cheaper
        # per node). Default ON; caller can pass cxx_feature_rng=False to restore
        # the numpy/bit-exact path (e.g. to reproduce pre-change canonical).
        given_params.setdefault('cxx_feature_rng', True)
        self.kwargs = given_params

        self.trees = []
        self.losses = []
        # Use MT_DTYPE so residuals stay in the kernel's expected dtype across
        # boosting iterations (avoids cross-precision casts).
        self.final_predictions = np.zeros(self.data.y.shape, dtype=MT_DTYPE)
        self.residuals = []

    def fit(self):
        # Early-stopping branch: handled by _fit_with_early_stopping which
        # may call back into the standard fit path for the optional
        # refit-on-full step. The _in_refit_pass guard prevents recursion.
        if (self.early_stopping_rounds is not None
                and not getattr(self, "_in_refit_pass", False)):
            return self._fit_with_early_stopping()

        # DEBUG: enable per-iteration RSS prints by setting MTT_MEM_DEBUG=1.
        # No-op otherwise.
        import os as _os
        _mem_debug = _os.environ.get("MTT_MEM_DEBUG") == "1"
        def _rss_mb():
            try:
                with open("/proc/self/status") as f:
                    for ln in f:
                        if ln.startswith("VmRSS:"):
                            return int(ln.split()[1]) / 1024
            except Exception:
                return -1
            return -1
        def _hwm_mb():
            try:
                with open("/proc/self/status") as f:
                    for ln in f:
                        if ln.startswith("VmHWM:"):
                            return int(ln.split()[1]) / 1024
            except Exception:
                return -1
            return -1
        if _mem_debug:
            print(f"  [BDT.fit] entry  RSS={_rss_mb():.0f} MB  HWM={_hwm_mb():.0f} MB", flush=True)

        # precompute bins once for all trees when prebin is enabled
        n_discrete_lev = self.kwargs.get('n_discrete_lev', 256)
        if self.prebin:
            features_np = np.ascontiguousarray(self.data.x.to_numpy())
            bins = precompute_global_bins(features_np, n_discrete_lev)
        else:
            bins = None

        if _mem_debug:
            print(f"  [BDT.fit] post-bins  RSS={_rss_mb():.0f} MB  HWM={_hwm_mb():.0f} MB", flush=True)

        # OPT memory: pre-allocate ONE residual buffer; do in-place numpy
        # subtraction every iteration instead of allocating a fresh ~53 MB
        # pandas DataFrame each time. Wrap the buffer in a DataFrame with
        # copy=False so CTree_MT.fit (which calls .loc[], .index) still sees
        # the pandas interface, but the underlying numpy memory is shared and
        # mutated in-place.
        if hasattr(self.data.y, "to_numpy"):
            y_np = np.ascontiguousarray(self.data.y.to_numpy(), dtype=MT_DTYPE)
            y_columns = list(self.data.y.columns)
            y_index = self.data.y.index
        else:
            y_np = np.ascontiguousarray(self.data.y, dtype=MT_DTYPE)
            y_columns = None
            y_index = None

        residuals_buf = np.empty_like(y_np)
        if y_columns is not None:
            residuals_df = pd.DataFrame(
                residuals_buf, columns=y_columns, index=y_index, copy=False)
            self.data_.y = residuals_df
        else:
            self.data_.y = residuals_buf

        if _mem_debug:
            print(f"  [BDT.fit] post-bufs  RSS={_rss_mb():.0f} MB  HWM={_hwm_mb():.0f} MB"
                  f"  y_np={y_np.nbytes/1e6:.0f} MB  buf={residuals_buf.nbytes/1e6:.0f} MB",
                  flush=True)

        # M0 RNG contract: deterministic per-tree seeds from base random_state.
        if self.random_state is not None:
            from .rng import per_tree_seeds
            tree_seeds = per_tree_seeds(self.random_state, self.n_estimators_)
        else:
            tree_seeds = [None] * self.n_estimators_

        for i in tqdm(range(self.n_estimators_), leave=False):
            if i == 0:
                if self.boost_from_average:
                    kwargs_ = self.kwargs.copy()
                    kwargs_.update({"max_depth": 1, "samp_frac": 1.0, "mtry": 1.0})
                    dt = DT(data=self.data, lambda_decay=self.lambda_decay,
                            objective_weights=self.objective_weights, **kwargs_)
                else:
                    dt = DT(data=self.data, lambda_decay=self.lambda_decay,
                            objective_weights=self.objective_weights, **self.kwargs)
                dt.fit(self.data, precomputed_bins=bins, random_state=tree_seeds[i],
                       use_indirect=self.use_indirect)
                self.final_predictions += dt.predictions(self.data)
            else:
                dt = DT(data=self.data_, lambda_decay=self.lambda_decay,
                        objective_weights=self.objective_weights, **self.kwargs)
                dt.fit(self.data_, precomputed_bins=bins, random_state=tree_seeds[i],
                       use_indirect=self.use_indirect)
                self.final_predictions += self.lr * dt.predictions(self.data)

            # In-place: residuals_buf = y_np - final_predictions.
            np.subtract(y_np, self.final_predictions, out=residuals_buf)
            # If pandas BlockManager re-consolidated, re-wrap to ensure
            # the next DT.fit sees the updated values (cheap; no data copy
            # when single-block, single-dtype).
            if y_columns is not None:
                self.data_.y = pd.DataFrame(
                    residuals_buf, columns=y_columns, index=y_index, copy=False)
            self.residuals = residuals_buf

            if self.keep_loss:
                self.losses.append(self.get_loss())

            dt.tree.strip_fit_data()
            self.trees.append(dt)

            # Per-iteration RSS probe (every 10 trees, plus iters 0,1,2)
            if _mem_debug and (i < 3 or (i + 1) % 10 == 0):
                print(f"  [BDT.fit] iter {i+1:>4d}/{self.n_estimators_}  "
                      f"RSS={_rss_mb():.0f} MB  HWM={_hwm_mb():.0f} MB", flush=True)

        del self.data_, y_np
        self.data_ = None
        if _mem_debug:
            print(f"  [BDT.fit] exit  RSS={_rss_mb():.0f} MB  HWM={_hwm_mb():.0f} MB", flush=True)

    def _fit_with_early_stopping(self):
        """mGBT with early stopping driven by eval MSE on a per-series tail.

        Holdout strategy: last `self.eval_k` rows of each series (default
        data.max_ahead = forecast horizon). Series shorter than k+1 contribute
        fully to training and nothing to eval (avoids zero-train cases).

        Loop: per iteration, fit a tree on train residuals, accumulate
        predictions on both train (for next-iter residual) and eval
        (for stopping signal). Track best eval MSE; stop when
        `early_stopping_rounds` iterations elapse without improvement.

        After stop: trees are truncated to best_iter+1. self.final_predictions
        is recomputed across the full data so downstream consumers
        (BDT.predict, get_loss, etc.) see consistent state.

        If self.eval_refit_on_full is True, after the truncate step the
        whole BDT.fit is re-run on the full data (train + eval) with
        n_estimators_ fixed to best_iter+1. The eval signal is "used up"
        by the search and then folded back into the model. Doubles wall.
        """
        from .tree import _per_series_last_k_holdout, _subset_datamt
        from .tree import precompute_global_bins

        # 1. Per-series holdout split.
        k = self.eval_k if self.eval_k is not None else int(
            getattr(self.data, "max_ahead", 1))
        train_rows, eval_rows = _per_series_last_k_holdout(self.data, k=k)
        if len(eval_rows) == 0:
            raise ValueError(
                f"early stopping eval set is empty (eval_k={k}, all series "
                f"shorter than k+1). Reduce eval_k or disable early stopping.")
        if len(train_rows) == 0:
            raise ValueError("early stopping training set is empty.")
        self._eval_k_used = k
        self._n_eval_rows = int(len(eval_rows))
        self._n_train_rows = int(len(train_rows))

        # 2. Subset DataMt-like wrappers for train, plus a residuals-mutable copy.
        train_data = _subset_datamt(self.data, train_rows)
        train_data_resid = _subset_datamt(self.data, train_rows)
        eval_x_df = self.data.x.iloc[eval_rows]
        eval_y_np = (self.data.y.iloc[eval_rows].to_numpy()
                     if hasattr(self.data.y, "iloc")
                     else self.data.y[eval_rows])

        # 3. Precompute bins ONCE on train portion (when prebin).
        n_discrete_lev = self.kwargs.get("n_discrete_lev", 256)
        if self.prebin:
            train_features_np = np.ascontiguousarray(
                train_data.x.to_numpy(), dtype=MT_DTYPE)
            bins = precompute_global_bins(train_features_np, n_discrete_lev)
        else:
            bins = None

        # 4. Train residual scaffolding (in-place buffer, same pattern as fit).
        train_y_np = np.ascontiguousarray(
            train_data.y.to_numpy(), dtype=MT_DTYPE)
        residuals_buf = np.empty_like(train_y_np)
        y_columns = list(train_data.y.columns) if hasattr(train_data.y, "columns") else None
        y_index = train_data.y.index if hasattr(train_data.y, "index") else None
        if y_columns is not None:
            train_data_resid.y = pd.DataFrame(
                residuals_buf, columns=y_columns, index=y_index, copy=False)
        else:
            train_data_resid.y = residuals_buf

        # 5. Prediction accumulators. train_pred is in MT_DTYPE matching the
        # kernel; eval_pred matches eval_y_np dtype.
        train_pred = np.zeros(train_y_np.shape, dtype=MT_DTYPE)
        eval_pred = np.zeros(eval_y_np.shape, dtype=eval_y_np.dtype)
        # The eval metric focuses on the original-target columns only
        # (data.y has K target cols + derivative + external; only the K
        # target cols matter for forecast accuracy).
        n_target = (len(self.data.target_col_list[0])
                    if hasattr(self.data, "target_col_list")
                    else eval_y_np.shape[1])

        # 5b. Resolve MASE inputs once (no-op for MSE path).
        if self.eval_metric == "mase":
            if self.eval_seasonal_scale is None:
                raise ValueError(
                    "eval_metric='mase' requires eval_seasonal_scale to be "
                    "provided (pandas Series indexed by series id).")
            if not hasattr(self.data, "series_means") or self.data.series_means is None:
                raise ValueError(
                    "eval_metric='mase' requires data.series_means (set in "
                    "DataMt.__init__ via series_means=...).")
            if not hasattr(self.data, "y_org") or self.data.y_org is None:
                raise ValueError(
                    "eval_metric='mase' requires data.y_org (auto-computed "
                    "in DataMt.transform).")
            # Per-eval-row metadata.
            eval_series_ids = self.data.index.iloc[eval_rows]["series"].to_numpy()
            # series_means is a DataFrame indexed by series; .loc returns a
            # DataFrame; flatten to (n_eval,).
            eval_series_means = (
                self.data.series_means.loc[eval_series_ids].to_numpy().ravel())
            eval_y_org = np.asarray(self.data.y_org[eval_rows])  # (n_eval, n_target)
            # eval_seasonal_scale may be a Series, dict, or array. Resolve
            # to a (n_eval,) float array indexed by eval_series_ids.
            ess = self.eval_seasonal_scale
            if hasattr(ess, "loc"):
                eval_seas_scale = np.asarray(
                    ess.loc[eval_series_ids], dtype=float)
            elif isinstance(ess, dict):
                eval_seas_scale = np.array(
                    [ess[s] for s in eval_series_ids], dtype=float)
            else:
                # Assume positional alignment is the user's responsibility.
                eval_seas_scale = np.asarray(ess, dtype=float)
            if eval_seas_scale.shape != (len(eval_rows),):
                raise ValueError(
                    f"eval_seasonal_scale shape mismatch after lookup: "
                    f"got {eval_seas_scale.shape}, expected ({len(eval_rows)},)")
        else:
            eval_series_means = None
            eval_y_org = None
            eval_seas_scale = None

        # 6. M0 RNG contract.
        if self.random_state is not None:
            from .rng import per_tree_seeds
            tree_seeds = per_tree_seeds(self.random_state, self.n_estimators_)
        else:
            tree_seeds = [None] * self.n_estimators_

        # 7. Boost loop with stopping criterion.
        best_iter = 0
        best_mse = float("inf")
        no_improve = 0
        self.eval_history_ = []

        for i in tqdm(range(self.n_estimators_), leave=False):
            if i == 0:
                if self.boost_from_average:
                    kwargs_ = self.kwargs.copy()
                    kwargs_.update({"max_depth": 1, "samp_frac": 1.0, "mtry": 1.0})
                    dt = DT(data=train_data, lambda_decay=self.lambda_decay,
                            objective_weights=self.objective_weights, **kwargs_)
                else:
                    dt = DT(data=train_data, lambda_decay=self.lambda_decay,
                            objective_weights=self.objective_weights, **self.kwargs)
                dt.fit(train_data, precomputed_bins=bins,
                       random_state=tree_seeds[i],
                       use_indirect=self.use_indirect)
                tree_train_pred = dt.predictions(train_data)
                tree_eval_pred = dt.tree.predict(eval_x_df)
                train_pred += tree_train_pred
                eval_pred += tree_eval_pred
            else:
                dt = DT(data=train_data_resid, lambda_decay=self.lambda_decay,
                        objective_weights=self.objective_weights, **self.kwargs)
                dt.fit(train_data_resid, precomputed_bins=bins,
                       random_state=tree_seeds[i],
                       use_indirect=self.use_indirect)
                tree_train_pred = dt.predictions(train_data)
                tree_eval_pred = dt.tree.predict(eval_x_df)
                train_pred += self.lr * tree_train_pred
                eval_pred += self.lr * tree_eval_pred

            # In-place residual update: residuals = train_y - train_pred.
            np.subtract(train_y_np, train_pred, out=residuals_buf)
            if y_columns is not None:
                train_data_resid.y = pd.DataFrame(
                    residuals_buf, columns=y_columns, index=y_index, copy=False)

            # Stopping criterion: eval metric over the original-target cols.
            if self.eval_metric == "mse":
                metric = float(np.mean(
                    (eval_pred[:, :n_target] - eval_y_np[:, :n_target]) ** 2))
            else:
                # MASE: unscale predictions via per-series mean, take MAE
                # against unscaled target, divide by per-series seasonal
                # scale. Aggregated by nanmean over rows. Matches the
                # bench's test-time row_wise_mase exactly (minus optional
                # integer rounding / nonneg clipping, which are post-hoc
                # transforms not appropriate for an in-fit signal).
                pred_unscaled = (
                    eval_pred[:, :n_target] * eval_series_means[:, None])
                mae_per_row = np.mean(
                    np.abs(pred_unscaled - eval_y_org[:, :n_target]), axis=1)
                # Drop non-finite per-row MASE before aggregating. nanmean drops
                # NaN (0/0 on constant series) but NOT +inf (mae>0 over a zero
                # seasonal_scale), and a single +inf pins metric at inf so ES
                # never improves -> best_iter=0 (silent underfit on any dataset
                # with constant/zero-scale series, e.g. COVID Deaths: 33/266).
                # Mirror the test-time _safe_aggregate isfinite->nan guard.
                with np.errstate(divide="ignore", invalid="ignore"):
                    mase_per_row = mae_per_row / eval_seas_scale
                mase_per_row = np.where(
                    np.isfinite(mase_per_row), mase_per_row, np.nan)
                metric = float(np.nanmean(mase_per_row))
            self.eval_history_.append(metric)

            # Improvement test: must beat current best by at least max(min_delta, 1e-12).
            # The 1e-12 floor guards against fp32 noise on the eval metric
            # computation; min_delta is the user-controlled wiggle filter.
            threshold = max(self.min_delta, 1e-12)
            improved = metric < best_mse - threshold
            if improved:
                best_mse = metric
                best_iter = i
                no_improve = 0
            else:
                no_improve += 1

            dt.tree.strip_fit_data()
            self.trees.append(dt)

            if no_improve >= self.early_stopping_rounds:
                break

        # 8. Truncate to best_iter + 1 trees.
        n_best = best_iter + 1
        self.trees = self.trees[:n_best]
        self.n_estimators_ = n_best
        self.best_iteration_ = best_iter
        self.best_eval_mse_ = best_mse

        # 9. Recompute self.final_predictions on FULL data using the
        # truncated trees. This keeps BDT.predict() / get_loss() / etc.
        # consistent with the non-early-stopping path.
        self.final_predictions = np.zeros(self.data.y.shape, dtype=MT_DTYPE)
        for j, tree_dt in enumerate(self.trees):
            pred_j = tree_dt.tree.predict(self.data.x)
            if j == 0:
                self.final_predictions += pred_j
            else:
                self.final_predictions += self.lr * pred_j

        # 10. Optional refit on full data (train + eval) with n_best trees.
        if self.eval_refit_on_full:
            # Reset state to use the standard fit path, then guard against
            # recursion via _in_refit_pass.
            self._in_refit_pass = True
            self.trees = []
            self.final_predictions = np.zeros(self.data.y.shape, dtype=MT_DTYPE)
            original_n = self.n_estimators_
            self.n_estimators_ = n_best
            try:
                self.fit()
            finally:
                # Restore the discovered n_estimators_; leave _in_refit_pass
                # cleared so subsequent .fit() calls behave as expected
                # (rare for a user to re-fit, but be defensive).
                self.n_estimators_ = n_best
                self._in_refit_pass = False

    def predict(self, X, n_tree=None):
        if n_tree is None:
            n_tree = self.n_estimators_

        for i, gbt in enumerate(self.trees):
            if i > n_tree:
                break
            if i == 0:
                predicted = gbt.tree.predict(X)
            else:
                predicted += self.lr * gbt.tree.predict(X)
        return predicted

    def score(self, data):
        return mean_squared_error(data.y, self.predict(data.x), multioutput="raw_values")

    def get_loss(self):
        return mean_squared_error(self.data.y, self.final_predictions, multioutput="raw_values")

    def get_pred_df(self):
        preds = self.data.y.copy()
        pred_df = pd.DataFrame(
            self.final_predictions,
            columns=[c + "_pred" for c in preds.columns],
            index=preds.index)
        return pd.concat([preds, pred_df], axis=1)


class RF:
    """Multi-target Random Forest (mRF).

    Parallelism is described by THREE orthogonal knobs (each means one thing):

      1. WHERE trees run  -- `n_jobs` (count) + `backend`:
           'thread'      shared-memory threads. The cpp kernel frees the GIL
                         (cxx_feature_rng), so trees run in parallel with FLAT
                         memory in n_jobs. The default and recommended choice.
           'process'     loky processes. Memory grows per worker; use
                         `mmap_data=True` to share arrays instead of pickling a
                         copy into each worker.
           'sequential'  one tree at a time (also used whenever n_jobs == 1).
      2. HOW the kernel reads data -- `indirect_fit` (default True): read the
         bagging rows by position from the shared base array (no per-tree
         gather). Bit-exact to the gather path; better memory + threading.
      3. SPLIT RNG -- `cxx_feature_rng` (default True): pure-C++ per-tree feature
         sampling (GIL-free). =False restores the numpy path (bit-exact to the
         pre-2026-06 canonical, but GIL-bound under threads).

    `mmap_data` is a PROCESS-only optimisation (ignored, with a warning, under
    thread/sequential). Recommended mRF config: backend='thread',
    n_jobs=<physical cores>, num_threads=1 (everything else optimal by default).

    Back-compat: joblib_backend='threading'/'loky' map to backend (deprecated).
    """

    def __init__(self, data, n_estimators_,
                 bagging_fraction=1, feature_fraction=1, samp_feature_by_node=True,
                 lambda_decay=None, objective_weights=None, keep_loss=False,
                 prebin=True, random_state=None, n_jobs=1,
                 backend=None, joblib_batch_size=None,
                 indirect_fit=None, keep_train_predictions=False,
                 mmap_data=None, joblib_backend=None,
                 **kwargs):

        self.data = copy.copy(data)
        self.n_estimators_ = n_estimators_
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.samp_feature_by_node = samp_feature_by_node
        self.lambda_decay = lambda_decay
        self.objective_weights = objective_weights
        self.prebin = prebin
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.joblib_batch_size = joblib_batch_size
        self.keep_loss = keep_loss

        # --- Parallelism backend: one knob, three values (see class docstring).
        #   'thread'     shared-memory threads; the cpp kernel frees the GIL so
        #                trees run in parallel with FLAT memory. The default.
        #   'process'    loky processes; memory grows per worker (use mmap_data
        #                to share the arrays instead of pickling a copy each).
        #   'sequential' one tree at a time (also forced whenever n_jobs == 1).
        # Back-compat: the old joblib_backend='threading'/'loky' is mapped here.
        if backend is None and joblib_backend is not None:
            warnings.warn(
                "RF(joblib_backend=...) is deprecated; use "
                "backend='thread'|'process'|'sequential'.",
                DeprecationWarning, stacklevel=2)
            backend = {"threading": "thread", "loky": "process"}.get(
                joblib_backend, joblib_backend)
        if backend is None:
            backend = "thread"
        if backend not in ("thread", "process", "sequential"):
            raise ValueError("backend must be 'thread', 'process' or "
                             f"'sequential', got {backend!r}")
        self.backend = backend

        # mmap_data is a PROCESS-only optimisation (memory-map the shared arrays
        # so loky workers don't each pickle a copy). Meaningless under thread
        # (already shared) / sequential. Default: on for process, off otherwise.
        if mmap_data is None:
            mmap_data = (backend == "process")
        elif mmap_data and backend != "process":
            warnings.warn(
                f"mmap_data=True is ignored under backend={backend!r}; only the "
                "'process' backend memory-maps data.", stacklevel=2)
            mmap_data = False
        self.mmap_data = bool(mmap_data)
        # keep_train_predictions: opt-in (default OFF). When OFF, RF workers do
        # NOT compute/ship each tree's N x K train-prediction array and
        # final_predictions is left None -- this removes the dominant mRF
        # orchestration cost (the per-tree result transfer). The train-pred
        # average is groundwork for future out-of-bag predictions, not a
        # current deliverable (run_benchmark consumes RF only via predict()).
        # keep_loss needs the accumulator, so it forces this on.
        self.keep_train_predictions = bool(keep_train_predictions) or keep_loss
        # indirect_fit: route per-tree fit through cpp fit_mt_tree_bin_indirect
        # (read the bagging rows by position from the shared base array -- no
        # per-tree contiguous gather). It is the better path everywhere (less
        # transient memory, less GIL-held pandas prep under threads) and is
        # BIT-EXACT to the contiguous path (tests/test_cpp_fit_indirect.py), so
        # it defaults ON -- decoupled from mmap_data. No effect when prebin=True
        # or MT_BACKEND != cpp. Set False only to force the legacy gather path.
        self.indirect_fit = True if indirect_fit is None else bool(indirect_fit)

        given_params = kwargs.copy()
        given_params['samp_frac'] = bagging_fraction
        given_params['mtry'] = feature_fraction
        given_params['samp_feature_by_node'] = samp_feature_by_node
        given_params['prebin'] = prebin
        # mRF parallelises ACROSS trees (n_jobs processes); intra-tree OMP on
        # top of that oversubscribes and is net-negative (a 5-thread worker ran
        # SLOWER than fully serial -- notes/perf_thread_scaling_2026-06-14.md).
        # So num_threads='auto' is meaningless here: coerce to 1 and steer the
        # user to n_jobs, which is the real mRF scaling lever.
        if given_params.get('num_threads') == "auto":
            warnings.warn(
                "RF ignores num_threads='auto' (intra-tree OMP is net-negative "
                "under process parallelism); using num_threads=1. Scale mRF via "
                "n_jobs instead.", stacklevel=2)
            given_params['num_threads'] = 1
        # mRF benefits most from the pure-C++ per-tree feature RNG: it makes the
        # cpp kernel GIL-free, so the threading backend (flat memory) runs at
        # full parallel speed. Default ON; cxx_feature_rng=False restores the
        # numpy/bit-exact path (reproduces pre-change canonical mRF numbers).
        given_params.setdefault('cxx_feature_rng', True)
        self.kwargs = given_params

        self.trees = []
        self.losses = []
        self.final_predictions = np.zeros(data.y.shape, dtype=MT_DTYPE)
        self.residuals = []

        # Lightweight metadata captured at construction so post-fit APIs
        # (predict, get_loss, get_pred_df) remain usable even when the
        # memmap path cleans up self.data.x/y. design note:
        # review_mmap_design_input_2026_06_04.md item #2.
        self.n_targets_ = int(data.y.shape[1])
        self.train_n_rows_ = int(data.y.shape[0])
        self.y_columns_ = (list(data.y.columns)
                           if hasattr(data.y, "columns") else None)
        self.y_index_ = (data.y.index.copy()
                         if hasattr(data.y, "index") else None)

    def fit(self):
        import os as _os
        _mem_debug = _os.environ.get("MTT_MEM_DEBUG") == "1"
        def _rss_mb():
            try:
                with open("/proc/self/status") as f:
                    for ln in f:
                        if ln.startswith("VmRSS:"):
                            return int(ln.split()[1]) / 1024
            except Exception:
                return -1
            return -1
        if _mem_debug:
            print(f"  [RF.fit] entry  RSS={_rss_mb():.0f} MB", flush=True)

        # precompute bins once for all trees when prebin is enabled
        n_discrete_lev = self.kwargs.get('n_discrete_lev', 256)
        if self.prebin:
            features_np = np.ascontiguousarray(self.data.x.to_numpy())
            bins = precompute_global_bins(features_np, n_discrete_lev)
        else:
            bins = None
        if _mem_debug:
            print(f"  [RF.fit] post-bins  RSS={_rss_mb():.0f} MB", flush=True)

        # M0 RNG contract: deterministic per-tree seeds from base random_state.
        if self.random_state is not None:
            from .rng import per_tree_seeds
            tree_seeds = per_tree_seeds(self.random_state, self.n_estimators_)
        else:
            tree_seeds = [None] * self.n_estimators_

        # M1.5: optional joblib across trees. n_jobs=1 stays on the sequential
        # path (preserves bit-identical legacy semantics + supports keep_loss).
        # n_jobs>1 fans trees out to a loky pool. Inner num_threads stays 1
        # unless the caller explicitly opted into nested parallelism via the
        # num_threads kwarg.
        if self.backend in ("thread", "process") and self.n_jobs != 1 \
                and not self.keep_loss:
            from joblib import Parallel, delayed
            # Map our backend name to joblib's.
            _bkwargs = {"backend": "threading" if self.backend == "thread" else "loky"}
            if self.joblib_batch_size is not None:
                _bkwargs["batch_size"] = self.joblib_batch_size
            # 'process' + mmap_data: memory-map the shared arrays so loky
            # workers read them instead of pickling a per-worker copy. Every
            # other case ('thread', or 'process' without mmap) passes the
            # DataMt directly -- threads share it natively; non-mmap processes
            # pickle it once. (mmap_data is forced False off the process backend
            # in __init__, so this condition is just the explicit form.)
            if self.backend == "process" and self.mmap_data:
                # Extract numpy arrays from data.x/data.y once. joblib's
                # auto-memmap detects top-level numpy arrays > max_nbytes and
                # shares them across loky workers as memory-mapped views,
                # instead of pickling a per-worker copy. Pandas DataFrames
                # nested inside DataMt do NOT get this treatment, hence the
                # explicit conversion path.
                x_obj = self.data.x
                y_obj = self.data.y
                x_arr = np.ascontiguousarray(
                    x_obj.to_numpy() if hasattr(x_obj, "to_numpy") else x_obj)
                y_arr = np.ascontiguousarray(
                    y_obj.to_numpy() if hasattr(y_obj, "to_numpy") else y_obj)
                x_columns = list(x_obj.columns) if hasattr(x_obj, "columns") else None
                y_columns = list(y_obj.columns) if hasattr(y_obj, "columns") else None
                x_index = x_obj.index if hasattr(x_obj, "index") else None
                y_index = y_obj.index if hasattr(y_obj, "index") else None
                # y_org is intentionally NOT shipped to workers (design review
                # item #4): RF workers don't consume y_org; only parent-side
                # tune_penalties does.

                # design note: pass only the minimal objective
                # metadata needed by arrangeObjective. DataMt.index,
                # series_means, max_ahead, etc. are NOT shipped — workers
                # don't use them.
                _objective_types = self.data.objective_types
                _target_type_list = self.data.target_type_list
                _target_col_list = self.data.target_col_list

                # Option 1 (senior-eng recommendation): drop this RF
                # instance's references to the pandas frames + force GC
                # before joblib dispatch. The original BlockManager memory
                # is released ONLY if no other holder exists (a caller-
                # provided DataMt may still hold the same blocks via the
                # shallow-copy semantics of self.data = copy.copy(data));
                # in that case this is a no-op for the parent peak but
                # still cheap. Reduces the parent peak by ~one raw-data
                # worth when callers drop their reference before .fit().
                self.data.x = None
                self.data.y = None
                del x_obj, y_obj
                import gc as _gc
                _gc.collect()

                # max_nbytes=1 forces memmap on any top-level numpy array
                # >= 1B. mmap_mode='c' (copy-on-write) is the default:
                # workers see shared physical memory for reads, writes go
                # to private COW pages. Required because the Cython
                # predict path uses writable memoryviews. Override via
                # MT_MMAP_MODE env var (e.g. 'r' for the read-only audit
                # path that finds where the writable buffer is required).
                _mmap_mode = _os.environ.get("MT_MMAP_MODE", "c")
                # return_as='generator' streams results as workers finish in
                # DISPATCH ORDER, so each (dt, pred) is consumed and dropped
                # before the next is materialised. Without this, Parallel(...)
                # builds a list of all n_estimators_ tuples in parent RSS
                # simultaneously: on Favorita Daily (N=1.66M, K=1, fp32)
                # that's 400 trees x 6.6 MB pred = ~2.6 GB held in the parent
                # during the accumulation loop, which was the dominant cause
                # of the 2026-06-07 jetsam kills on the Mac benchmark.
                # 'generator' (ordered) is required for fp32 bit-determinism
                # of the final_predictions sum.
                results = Parallel(n_jobs=self.n_jobs, return_as='generator',
                                   max_nbytes=1, mmap_mode=_mmap_mode, **_bkwargs)(
                    delayed(_fit_one_rf_tree_memmap)(
                        tree_seeds[i], x_arr, y_arr, x_columns,
                        x_index, y_columns, y_index,
                        _objective_types, _target_type_list, _target_col_list,
                        self.lambda_decay, self.objective_weights,
                        self.kwargs, bins,
                        use_indirect=self.indirect_fit,
                        compute_pred=self.keep_train_predictions)
                    for i in range(self.n_estimators_))
            else:
                # max_nbytes=None disables joblib's silent auto-mmap of any
                # numpy array > 1 MB via the per-pool scratch dir under
                # $TMPDIR/joblib_memmapping_folder_*. On heavy datasets
                # (Favorita Daily, Kaggle, Rossmann) those scratch files
                # accumulate ~100s of MB per dispatch, get page-cached +
                # wired by the kernel, and push macOS into jetsam. With
                # mmap_data=False the user has explicitly opted out of the
                # shared-memmap path; respecting that means also opting out
                # of joblib's auto-mmap (otherwise data still goes through
                # the scratch dir, defeating the point).
                # return_as='generator': see mmap-branch comment above for
                # why streaming is required to avoid the parent-side ~2.6 GB
                # pred-accumulator on heavy datasets.
                results = Parallel(n_jobs=self.n_jobs, return_as='generator',
                                   max_nbytes=None, **_bkwargs)(
                    delayed(_fit_one_rf_tree)(
                        tree_seeds[i], self.data, self.lambda_decay,
                        self.objective_weights, self.kwargs, bins,
                        use_indirect=self.indirect_fit,
                        compute_pred=self.keep_train_predictions)
                    for i in range(self.n_estimators_))
            for dt, pred in results:
                if pred is not None:
                    self.final_predictions += pred
                self.trees.append(dt)
                del pred  # explicit drop so refcount falls to 0 immediately;
                # without this, CPython keeps the just-yielded value live
                # until the next loop iteration overwrites the local.
            if self.keep_train_predictions:
                self.final_predictions /= self.n_estimators_
            else:
                self.final_predictions = None
            if _mem_debug:
                print(f"  [RF.fit] parallel exit  RSS={_rss_mb():.0f} MB", flush=True)
            return

        for i in tqdm(range(self.n_estimators_), leave=False):
            dt = DT(self.data, lambda_decay=self.lambda_decay,
                    objective_weights=self.objective_weights, **self.kwargs)
            dt.fit(self.data, precomputed_bins=bins, random_state=tree_seeds[i],
                   use_indirect=self.indirect_fit)
            if self.keep_train_predictions:
                self.final_predictions += dt.predictions(self.data)
            if self.keep_loss:
                self.losses.append(self.get_loss(i + 1))
            dt.tree.strip_fit_data()
            self.trees.append(dt)

            if _mem_debug and (i < 3 or (i + 1) % 10 == 0):
                print(f"  [RF.fit] iter {i+1:>4d}/{self.n_estimators_}  "
                      f"RSS={_rss_mb():.0f} MB", flush=True)

        if self.keep_train_predictions:
            self.final_predictions /= self.n_estimators_
        else:
            self.final_predictions = None

    def predict(self, X, n_tree=None):
        if n_tree is None:
            n_tree = self.n_estimators_

        # Use n_targets_ (saved at construction) instead of self.data.y.shape
        # so this still works after RF.fit(mmap_data=True) nulls self.data.y.
        predicted = np.zeros([X.shape[0], self.n_targets_])
        for i, rf in enumerate(self.trees):
            if i > n_tree:
                break
            predicted += rf.tree.predict(X)
        return predicted / (i + 1)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X), multioutput="raw_values")

    def get_loss(self, current_iteration):
        if self.final_predictions is None:
            raise RuntimeError(
                "RF.get_loss() requires the train-prediction average, which is "
                "opt-in (default OFF). Re-fit with keep_train_predictions=True "
                "(or keep_loss=True).")
        if self.data.y is None:
            raise RuntimeError(
                "RF.get_loss() requires the original training data, which "
                "was released by mmap_data=True cleanup. Re-fit with "
                "mmap_data=False if you need this method.")
        return mean_squared_error(
            self.data.y, self.final_predictions / current_iteration, multioutput="raw_values")

    def get_pred_df(self):
        if self.final_predictions is None:
            raise RuntimeError(
                "RF.get_pred_df() requires the train-prediction average, which "
                "is opt-in (default OFF). Re-fit with keep_train_predictions=True "
                "(or keep_loss=True).")
        if self.data.y is None:
            raise RuntimeError(
                "RF.get_pred_df() requires the original training data, "
                "which was released by mmap_data=True cleanup. Re-fit with "
                "mmap_data=False if you need this method.")
        preds = self.data.y.copy()
        pred_df = pd.DataFrame(
            self.final_predictions,
            columns=[c + "_pred" for c in preds.columns],
            index=preds.index)
        return pd.concat([preds, pred_df], axis=1)
