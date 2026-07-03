import ast
import gc
import copy
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, ParameterGrid
from tqdm import tqdm

from .utils import rmse, evaluate_forecast

# Logical core count, sampled once at import (cheap; avoids an os call per
# tree in ensemble loops). Falls back to 1 if undeterminable.
_CPU_COUNT = os.cpu_count() or 1


def _resolve_auto_num_threads(n_features, cpu_count=None):
    """Pick an OMP thread count for the split-search scan from feature count.

    The kernel parallelises the per-node split search OVER FEATURES, so the
    benefit is bounded by how many features there are AND how memory-bandwidth
    -bound the (radix-sorted) scan is. Measured on Apple Silicon
    (notes/perf_thread_scaling_2026-06-14.md): P=2 -> 1 thread optimal (4
    threads ran mDT/mGBT ~2-3x SLOWER, pure fork/join overhead); P=65 -> 4
    optimal, 8 regressed (bandwidth saturates). Heuristic: ~1 thread per 16
    features, capped at 4 (kernel is BW-bound, saturates by ~4 even at high P)
    and at the machine's core count. Crossovers are box-specific; this is a
    deliberately conservative default, overridden by any explicit int.

    NOTE: this is for INTRA-tree feature parallelism only. mRF parallelises
    ACROSS trees via processes (n_jobs) and must keep num_threads=1 -- in-worker
    OMP there is net-negative (see the notes); RF coerces 'auto' -> 1.
    """
    cores = _CPU_COUNT if cpu_count is None else cpu_count
    by_features = max(1, int(n_features) // 16)
    return max(1, min(4, by_features, cores))


def _time_series_split_per_series(data, nfolds=5, block=None, min_train=None):
    """Rolling-origin CV splits per series, sized to the forecast horizon.

    Per-series adaptive folds: each series contributes as many folds as its
    history can support without violating ``min_train``. The most-recent
    fold (closest to the deployment forecast origin) is included for every
    series that can fit at least one test block; earlier folds are added
    progressively for longer series.

    Parameters
    ----------
    data : DataMt
        Source data; uses ``data.index`` for series/time information and
        ``data.max_ahead`` as the default block size.
    nfolds : int
        Maximum number of folds to produce per series.
    block : int, optional
        Test-block size per series. Defaults to ``data.max_ahead`` (the
        forecast horizon), so each fold's CV objective matches the
        deployment-time forecast horizon.
    min_train : int, optional
        Minimum number of training rows required per series for a fold
        to be valid. Defaults to ``block / 2``. A series's individual fold
        is dropped if its training portion would be smaller than this; the
        fold is still produced for other series that satisfy the rule.

    Returns
    -------
    list of (train_rows, test_rows) tuples
        Each entry is a pair of ``np.ndarray`` of integer row indices.
        Folds are ordered earliest-to-latest in time. A fold may include
        only a subset of series (the long ones); the most-recent fold
        includes all series that satisfy ``min_train``.
    """
    if block is None:
        block = max(1, int(getattr(data, "max_ahead", 1)))
    block = max(1, int(block))
    if min_train is None:
        min_train = max(1, block // 2)

    df_idx = data.index.reset_index(drop=True).copy()
    df_idx["_row"] = np.arange(len(df_idx))
    df_idx = df_idx.sort_values(["series", "index"]).reset_index(drop=True)

    # For each series, precompute (test_start, test_end) for every potential
    # fold; keep only those fitting the min_train rule.
    series_groups = list(df_idx.groupby("series", sort=False))

    splits = []
    # fold 0 = oldest test block; fold nfolds-1 = most-recent test block
    for fold in range(nfolds):
        test_rows, train_rows = [], []
        for _, grp in series_groups:
            n = len(grp)
            test_end = n - (nfolds - 1 - fold) * block
            test_start = test_end - block
            if test_start < min_train or test_end > n:
                # This particular series can't support this fold; skip the
                # series for THIS fold but keep the fold for others.
                continue
            row_arr = grp["_row"].to_numpy()
            test_rows.extend(row_arr[test_start:test_end].tolist())
            train_rows.extend(row_arr[:test_start].tolist())
        if not train_rows or not test_rows:
            continue
        splits.append((np.asarray(train_rows), np.asarray(test_rows)))
    return splits


def _per_series_last_k_holdout(data, k=None):
    """Per-series tail holdout for mGBT early stopping.

    For each series in `data`, the last `k` rows (ordered by `data.index["index"]`)
    are returned as the eval set; the rest are training. Series shorter than
    `k + 1` are skipped from eval (they contribute fully to train) so we never
    leave a series with zero training rows. Returns global row positions
    (np.int64) into `data.x` / `data.y` / `data.y_org` / `data.index`.

    Parameters
    ----------
    data : DataMt
        Source with `data.index` carrying `series` and `index` columns.
    k : int, optional
        Number of trailing rows per series to hold out. Defaults to
        `data.max_ahead` so the eval block matches the forecast horizon.

    Returns
    -------
    (train_rows, eval_rows) : tuple of np.ndarray
        Row positions for training and eval, respectively.
    """
    if k is None:
        k = int(getattr(data, "max_ahead", 1))
    k = max(1, int(k))

    df_idx = data.index.reset_index(drop=True).copy()
    df_idx["_row"] = np.arange(len(df_idx))
    df_idx = df_idx.sort_values(["series", "index"]).reset_index(drop=True)

    train_rows, eval_rows = [], []
    for _, grp in df_idx.groupby("series", sort=False):
        row_arr = grp["_row"].to_numpy()
        n = len(row_arr)
        if n <= k:
            # Series too short: keep all rows in train, contribute nothing to
            # eval. Avoids leaving the series with no training history.
            train_rows.extend(row_arr.tolist())
            continue
        train_rows.extend(row_arr[: n - k].tolist())
        eval_rows.extend(row_arr[n - k:].tolist())
    return np.asarray(train_rows, dtype=np.int64), np.asarray(eval_rows, dtype=np.int64)


def _subset_datamt(data, rows):
    """Return a SimpleNamespace mimicking DataMt at the given row positions.

    Used by mGBT early stopping to isolate the training portion from the
    held-out eval portion. Carries over all metadata fields used downstream
    by `CTree_MT.arrangeObjective` and `DT.fit`.
    """
    import types
    rows = np.asarray(rows)
    ns = types.SimpleNamespace()
    ns.x = data.x.iloc[rows].copy()
    ns.y = data.y.iloc[rows].copy() if hasattr(data.y, "iloc") else data.y[rows].copy()
    if hasattr(data, "y_org") and data.y_org is not None:
        try:
            ns.y_org = data.y_org[rows]
        except Exception:
            ns.y_org = None
    else:
        ns.y_org = None
    if hasattr(data, "index") and data.index is not None:
        ns.index = data.index.iloc[rows].copy()
    for attr in ("max_ahead", "n_derivative", "penalty_list", "series_means",
                 "series_index", "int_convert",
                 "target_col_list", "target_type_list",
                 "flat_target_cols", "objective_types"):
        if hasattr(data, attr):
            setattr(ns, attr, getattr(data, attr))
    return ns
# --- Precision + backend dispatch.
# MT_DTYPE selects fp32 vs fp64 kernels (data-matrix memory tradeoff).
# MT_BACKEND selects legacy (cfuncs/) vs fast (cfuncs_fast/) Cython kernels.
# Both default to the legacy-equivalent path through M6 acceptance.
import os
_MT_DTYPE_STR = os.environ.get("MT_DTYPE", "fp64").lower()
_MT_BACKEND_STR = os.environ.get("MT_BACKEND", "legacy").lower()
if _MT_BACKEND_STR not in ("legacy", "fast", "cpp"):
    raise ValueError(
        f"MT_BACKEND must be 'legacy', 'fast', or 'cpp', got: {_MT_BACKEND_STR!r}")
# Path B narrow scope: MT_BACKEND=cpp overrides ONLY the selfhash kernel
# (where mGBT-selfhash needs the kernel-level win). Prebin and prediction
# stay on cfuncs_fast since cfuncs_fast already wins via M1.5/M4 there.
_BACKEND_PKG = "cfuncs_fast" if _MT_BACKEND_STR == "fast" else (
    "cfuncs" if _MT_BACKEND_STR == "legacy" else "cfuncs_fast")
_SELFHASH_PKG = ("cfuncs_cpp" if _MT_BACKEND_STR == "cpp"
                 else _BACKEND_PKG)

if _MT_DTYPE_STR == "fp32":
    _selfhash = __import__(f"mttrees.{_SELFHASH_PKG}.mt_ctree_selfhash_f32",
                           fromlist=["fit_mt_tree_bin"])
    _prebin = __import__(f"mttrees.{_BACKEND_PKG}.mt_ctree_prebin_f32",
                         fromlist=["precompute_global_bins", "fit_mt_tree_prebin",
                                   "predict_mt_tree", "predict_mt_tree_node"])
    MT_DTYPE = np.float32
elif _MT_DTYPE_STR == "fp64":
    if _MT_BACKEND_STR == "cpp":
        raise ValueError(
            "MT_BACKEND=cpp only supports fp32 in narrow-scope Path B. "
            "Use MT_DTYPE=fp32.")
    _selfhash = __import__(f"mttrees.{_BACKEND_PKG}.mt_ctree_selfhash",
                           fromlist=["fit_mt_tree_bin"])
    _prebin = __import__(f"mttrees.{_BACKEND_PKG}.mt_ctree_prebin",
                         fromlist=["precompute_global_bins", "fit_mt_tree_prebin",
                                   "predict_mt_tree", "predict_mt_tree_node"])
    MT_DTYPE = np.float64
else:
    raise ValueError(f"MT_DTYPE must be 'fp32' or 'fp64', got: {_MT_DTYPE_STR!r}")

fit_mt_tree_bin = _selfhash.fit_mt_tree_bin
# Indirected-access entry (cpp-only). When MT_INDIRECT_FIT=1 is set and
# MT_BACKEND=cpp + prebin=False, CTree_MT.fit takes the no-gather path:
# pass base features/labels + int32 positions, skip the per-tree
# `features.loc[sample_index].to_numpy()` materialization. Saves ~80% x
# N_base x (P+K) x 4 bytes per worker on mmap workloads
# (the mmap-design review, item #3).
_fit_mt_tree_bin_indirect = getattr(
    _selfhash, "fit_mt_tree_bin_indirect", None)
precompute_global_bins = _prebin.precompute_global_bins
fit_mt_tree_prebin = _prebin.fit_mt_tree_prebin
# Predict path: prefer the cpp implementation when MT_BACKEND=cpp. The cpp
# predict accepts read-only buffers (const py::array_t<...>), enabling
# mmap_mode='r' end-to-end. The Cython predict_mt_tree is retained as a
# fallback for fast/legacy backends. predict_mt_tree_node is only needed
# for the leaf-id-return path (not used in mRF/mGBT/mDT main flow) and
# stays on the cython implementation.
if _MT_BACKEND_STR == "cpp" and hasattr(_selfhash, "predict_mt_tree"):
    predict_mt_tree = _selfhash.predict_mt_tree
else:
    predict_mt_tree = _prebin.predict_mt_tree
predict_mt_tree_node = _prebin.predict_mt_tree_node


class DataMt:
    def __init__(self,
                 max_ahead=1,
                 n_derivative=None,
                 penalty_list=None,
                 series_index=None,
                 series_means=None,
                 int_convert=None):

        self.max_ahead = max_ahead
        self.n_derivative = n_derivative
        self.penalty_list = penalty_list
        self.series_index = series_index
        self.series_means = series_means.set_index('series')
        self.int_convert = int_convert

    def transform(self, feature_df, target_df=None, drop_penalty=True):
        if target_df is not None:
            target_col_list = []
            target_col_list.append(['ahead_' + str(i) for i in range(self.max_ahead)])
            target_type_list = ['target']

            if self.n_derivative is not None:
                # OPT: removed `tmp_target = target_df.copy()` and dead `set_index`.
                # The diff is computed from the original target_df ahead_* columns,
                # which are unchanged across derivative iterations (we only append
                # new der_* columns; the ahead_* slice stays the same).
                ahead_slice = target_df.iloc[:, 2:(self.max_ahead + 2)].to_numpy()
                for i in range(self.n_derivative):
                    diff_target = pd.DataFrame(np.diff(ahead_slice, i + 1))
                    diff_target.columns = [
                        'der_' + str(i + 1) + '_ahead_' + str(a)
                        for a in range(self.max_ahead - i - 1)
                    ]
                    target_df = pd.concat([target_df, diff_target], axis=1)
                    target_col_list.append(list(diff_target.columns))
                    target_type_list.append('derivative')
                del ahead_slice
                gc.collect()

            if self.penalty_list is not None:
                penalty_labels = [feature_df[m] for m in self.penalty_list]
                penalty_labels = pd.concat(penalty_labels, axis=1)
                target_col_list.append(list(penalty_labels.columns))
                target_df = pd.concat([target_df, penalty_labels], axis=1)
                target_type_list.append('external')

            used_targets = list(itertools.chain(*target_col_list))
            # Cast y to MT_DTYPE (fp64 or fp32 depending on MT_DTYPE env var).
            # Use copy=False so this is a no-op when the source dtype already
            # matches MT_DTYPE (which is the case when build_features_and_targets
            # used MT_DTYPE upstream).
            _y_src = target_df[used_targets]
            if (_y_src.dtypes == MT_DTYPE).all():
                self.y = _y_src
            else:
                self.y = _y_src.astype(MT_DTYPE)
                # Free the fp64 source before allocating y_org below.
                del _y_src
                gc.collect()
            # OPT: compute y_org via direct numpy multiply (no pandas .multiply
            # which creates a transient DataFrame in fp64). target_col_list[0]
            # is the first len(target_col_list[0]) columns of self.y.
            n_target = len(target_col_list[0])
            _y_target_np = self.y.to_numpy()[:, :n_target]
            _means = self.series_means.loc[target_df.series].values.reshape(-1, 1)
            self.y_org = (_y_target_np * _means).astype(np.float64, copy=False)
            del _y_target_np, _means

            self.target_col_list = target_col_list
            self.target_type_list = target_type_list
            self.flat_target_cols = list(itertools.chain(*target_col_list))
            self.objective_types = list(dict.fromkeys(target_type_list))

        self.index = feature_df[['index', 'series']]
        # Cast x to MT_DTYPE (matches kernel signature, avoids per-tree casts).
        # No-op when feature_df is already in MT_DTYPE.
        _x_src = feature_df.drop(['index', 'series'], axis=1)
        if (_x_src.dtypes == MT_DTYPE).all():
            self.x = _x_src
        else:
            self.x = _x_src.astype(MT_DTYPE)
            del _x_src
            gc.collect()

        if drop_penalty and self.penalty_list is not None:
            self.x = self.x.drop(self.penalty_list, axis=1)

    def process_predictions(self, predicted, derivative=False):
        if derivative:
            if 'derivative' in self.target_type_list:
                st_index = len(self.target_col_list[0])
                en_index = st_index + len(self.target_col_list[1])
                diff_prediction = pd.DataFrame(
                    predicted[:, st_index:en_index], columns=self.target_col_list[1])
            else:
                diff_prediction = None
        else:
            diff_prediction = None

        prediction = predicted[:, range(self.max_ahead)]
        if self.series_means is not None:
            prediction = prediction * self.series_means.loc[self.index.series].values

        if self.int_convert == 1:
            prediction = np.round(prediction)
            prediction[prediction < 0] = 0

        return prediction, diff_prediction

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj


# M5: top-level worker for joblib loky CV parallelization. Each call gets a
# fresh CTree_MT (no shared state between tasks) and a deterministic task seed
# under the M0 RNG contract so n_jobs_cv=1 sequential matches n_jobs_cv>1
# parallel bit-identically.
def _fit_one_cv_task(task, data, init_kwargs, depth_levels):
    from .utils import rmse, evaluate_forecast
    param = task["param"]
    fold_idx = task["fold_idx"]
    train_index = task["train_index"]
    test_index = task["test_index"]
    task_seed = task["task_seed"]
    eval_fun = task["eval_fun"]

    obj = [1, param["lambda"], param["beta"]]
    tree = CTree_MT(**init_kwargs)
    tree.arrangeObjective(data, param["lambda_decay"], obj)
    tree.fit(data.x.iloc[train_index], data.y.iloc[train_index],
             random_state=task_seed)

    rows = []
    for current_depth in depth_levels:
        prediction = tree.predict(data.x, current_depth)
        target_predictions, _ = data.process_predictions(prediction)
        all_errors = evaluate_forecast(eval_fun, data.y_org, target_predictions)
        all_errors_rmse = evaluate_forecast(rmse, data.y_org, target_predictions)
        test_errors = all_errors[test_index]
        train_errors = all_errors[train_index]
        test_errors_rmse = all_errors_rmse[test_index]
        train_errors_rmse = all_errors_rmse[train_index]
        rows.append({
            "fold": fold_idx + 1,
            "depth": current_depth,
            "lambda_decay": tree.lambda_decay,
            "objective_weights": str(obj),
            "train_error": np.nanmean(train_errors),
            "test_error": np.nanmean(test_errors),
            "train_std": np.nanstd(train_errors),
            "test_std": np.nanstd(test_errors),
            "train_rmse": np.nanmean(train_errors_rmse),
            "test_rmse": np.nanmean(test_errors_rmse),
        })
    return rows


class CTree_MT:
    def __init__(self,
                 max_depth=5,
                 samp_frac=1.0,
                 mtry=1.0,
                 samp_feature_by_node=False,
                 min_samples_leaf=5,
                 gain_threshold=0.0,  # min-gain split pruning; 0.0 = OFF (default, preserves reproducibility), >0 enables
                 n_discrete_lev=None,
                 num_threads=1,
                 verbose=0,
                 prebin=True,
                 aggregation='rank',
                 obj_weight_mode='fixed',
                 obj_weight_dirichlet_alpha=1.0,
                 cxx_feature_rng=False):

        self.max_depth = max_depth
        self.samp_frac = samp_frac
        self.mtry = mtry
        self.samp_feature_by_node = samp_feature_by_node
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        # cxx_feature_rng: use the pure-C++ per-tree std::mt19937 for per-node
        # feature subsampling instead of numpy's RNG. Lets the cpp kernel run
        # GIL-free (full threading-backend parallelism) and is cheaper per node.
        # NOT bit-exact to the numpy path (different RNG -> different valid
        # subsamples -> statistically-equivalent ensemble); deterministic per
        # tree + identical on Linux/Mac. Only meaningful on the cpp selfhash
        # path (prebin=False). Default False preserves the numpy/bit-exact path.
        self.cxx_feature_rng = bool(cxx_feature_rng)
        self.n_discrete_lev = 256 if n_discrete_lev is None else n_discrete_lev
        # num_threads: a positive int (used verbatim) or the string "auto"
        # (resolved from the feature count at fit time -- see
        # _resolve_auto_num_threads). "auto" is opt-in; the default stays 1 so
        # existing callers are bit-for-bit and timing-for-timing unchanged.
        if not (num_threads == "auto"
                or (isinstance(num_threads, (int, np.integer)) and num_threads >= 1)):
            raise ValueError(
                f"num_threads must be a positive int or 'auto', got {num_threads!r}")
        self.num_threads = num_threads
        self.verbose = verbose
        self.prebin = prebin
        if aggregation not in ('rank', 'score'):
            raise ValueError(f"aggregation must be 'rank' or 'score', got {aggregation!r}")
        self.aggregation = aggregation
        # Per-node random objective-weight modes (post-IJF follow-up; see
        # notes/post_ijf_random_objective_weights.md). 'fixed' preserves the
        # caller-supplied objective_weights at every node (bit-exact to prior
        # behavior). 'dirichlet' resamples w ~ Dirichlet(alpha * ones(K)) at
        # each node. 'onehot' picks one objective uniformly at each node. All
        # random draws go through the global numpy RNG so the M0 per-tree
        # seeding contract (np.random.seed(seed) in CTree_MT.fit) makes the
        # tree deterministic for a given random_state.
        if obj_weight_mode not in ('fixed', 'dirichlet', 'onehot'):
            raise ValueError(
                f"obj_weight_mode must be one of 'fixed', 'dirichlet', "
                f"'onehot'; got {obj_weight_mode!r}")
        self.obj_weight_mode = obj_weight_mode
        self.obj_weight_dirichlet_alpha = float(obj_weight_dirichlet_alpha)
        self.num_nodes = None
        self.tree_info = None
        self.split_vals = None
        self.node_means = None

    def arrangeObjective(self, data, lambda_decay=None, objective_weights=[1]):
        self.lambda_decay = lambda_decay
        self.n_objective = int(len(data.objective_types))
        self.objective_weights = np.array(objective_weights).astype(MT_DTYPE)

        # Build per-target weights and the target -> objective-group map.
        # weights_target[t] is the per-target weight applied to the t-th column
        # of data.y when accumulating its contribution to the
        # objective-group score in the split criterion. target_weight_map[t]
        # tells which objective-group (level / derivative / external) target t
        # belongs to. The two arrays must have the same length n_target.
        weights_target_per_group = {}
        for obj_type in data.objective_types:
            label_indices = [i for i, j in enumerate(data.target_type_list) if j == obj_type]
            for idx in label_indices:
                n_cols = len(data.target_col_list[idx])
                if obj_type == 'target' and self.lambda_decay is not None:
                    if self.lambda_decay <= 0:
                        self.lambda_decay = 0.001
                    if n_cols > 1:
                        lambda_transformed = np.log(self.lambda_decay) / (n_cols - 1)
                        w = np.exp(lambda_transformed * np.arange(n_cols)).astype(MT_DTYPE)
                    else:
                        w = np.ones(n_cols, dtype=MT_DTYPE)
                elif obj_type == 'derivative' and self.lambda_decay is not None:
                    # Same horizon-decay scheme applied to derivative columns
                    if self.lambda_decay <= 0:
                        self.lambda_decay = 0.001
                    if n_cols > 1:
                        lambda_transformed = np.log(self.lambda_decay) / (n_cols - 1)
                        w = np.exp(lambda_transformed * np.arange(n_cols)).astype(MT_DTYPE)
                    else:
                        w = np.ones(n_cols, dtype=MT_DTYPE)
                else:
                    # External (sin/cos season etc.) and any other group: uniform weight
                    w = np.ones(n_cols, dtype=MT_DTYPE)
                weights_target_per_group[idx] = w

        # Concatenate in the original target_col_list order
        weights_target = []
        target_weight_map = []
        for idx, terms in enumerate(data.target_col_list):
            weights_target.append(weights_target_per_group[idx])
            target_weight_map.append(np.repeat(idx, len(terms)))

        self.weights_target = np.array(
            [item for sublist in weights_target for item in sublist],
            dtype=MT_DTYPE)
        self.target_weight_map = np.array(
            [item for sublist in target_weight_map for item in sublist]).astype(np.intc)

    def fit(self, features, labels, precomputed_bins=None, random_state=None,
            use_indirect=False):
        # M0 RNG contract: random_state seeds both the pandas bagging draw and
        # the kernel's per-node feature sampling. The cpp kernel receives an
        # explicit per-tree RandomState `rng` (thread-safe -> enables the mRF
        # threading backend, see notes/mrf_leak_investigation_2026-06-15.md);
        # the legacy prebin (Cython) kernel still reads numpy's GLOBAL RNG, so
        # that path is seeded as before. Bit-exact either way:
        # RandomState(s).permutation == np.random.seed(s)+np.random.permutation.
        # random_state=None preserves legacy behavior (fixed random_state=5 for
        # bagging, ambient global RNG for feature sampling -> rng=None fallback).
        if random_state is None:
            bagging_seed = 5
            rng = None
            cxx_rng_seed = -1
        else:
            bagging_seed = int(random_state)
            if self.cxx_feature_rng and not self.prebin:
                # Pure-C++ per-tree RNG path: GIL-free, fast, non-bit-exact.
                cxx_rng_seed = bagging_seed
                rng = None
            else:
                cxx_rng_seed = -1
                rng = np.random.RandomState(bagging_seed)
                if self.prebin:
                    np.random.seed(bagging_seed)

        # Resolve "auto" num_threads now that the feature count is known.
        # Bit-exact regardless of the value (parallelism is value-independent;
        # proven in notes/perf_thread_scaling_2026-06-14.md). Overwrite so the
        # kernel calls below and predict() reuse the resolved int.
        if self.num_threads == "auto":
            self.num_threads = _resolve_auto_num_threads(features.shape[1])

        if self.samp_frac < 1:
            # Subagging (subsample WITHOUT replacement) at samp_frac of rows,
            # NOT bootstrap-with-replacement. Each tree sees n unique rows
            # drawn uniformly; no row appears twice in the same tree. This is
            # the per-tree row randomisation that drives the RF variance
            # reduction in this codebase; out-of-bag accounting and bootstrap
            # double-weighting are not part of the contract.
            n = int(np.ceil(len(features) * self.samp_frac))
            sample_index = features.sample(n, random_state=bagging_seed).index
        else:
            sample_index = features.index

        potential_node_cnt = int(len(sample_index) / self.min_samples_leaf)
        if self.max_depth >= 31:
            self.num_nodes = potential_node_cnt
        else:
            self.num_nodes = np.min([potential_node_cnt, 2 ** self.max_depth])

        # Per-node random objective-weight modes (post-IJF follow-up). The
        # string mode is translated to an int enum the cpp kernel understands:
        #   'fixed'     -> 0  (pass caller objective_weights through; default)
        #   'dirichlet' -> 1  (sample Dirichlet(alpha * ones) at each node)
        #   'onehot'    -> 2  (sample one objective uniformly at each node)
        # The legacy prebin (Cython) kernel only supports FIXED; the cpp paths
        # implement all three.
        _OBJ_W_MODE_MAP = {'fixed': 0, 'dirichlet': 1, 'onehot': 2}
        obj_weight_mode_int = _OBJ_W_MODE_MAP[self.obj_weight_mode]
        if self.prebin and obj_weight_mode_int != 0:
            raise NotImplementedError(
                "obj_weight_mode != 'fixed' is only supported on the cpp "
                "selfhash backend (prebin=False). Got obj_weight_mode="
                f"{self.obj_weight_mode!r} with prebin=True.")

        # use_indirect + cpp backend + selfhash: skip the per-tree
        # contiguous gather. Pass base ndarrays + int32 positions into the
        # indirect cpp entry. Bit-exact gate: tests/test_cpp_fit_indirect.py.
        # The flag is plumbed explicitly (not via env var) so the indirect
        # path is scoped to RF mmap workers and does NOT affect mDT/mGBT,
        # which call CTree_MT.fit with the default use_indirect=False.
        _use_indirect = (
            (not self.prebin)
            and use_indirect
            and (_fit_mt_tree_bin_indirect is not None))

        if _use_indirect:
            features_np = None
            labels_np = None
            features_base_np = features.to_numpy(copy=False)
            labels_base_np = labels.to_numpy(copy=False)
            if features_base_np.dtype != MT_DTYPE:
                features_base_np = np.ascontiguousarray(
                    features_base_np, dtype=MT_DTYPE)
            elif not features_base_np.flags["C_CONTIGUOUS"]:
                features_base_np = np.ascontiguousarray(features_base_np)
            if labels_base_np.dtype != MT_DTYPE:
                labels_base_np = np.ascontiguousarray(
                    labels_base_np, dtype=MT_DTYPE)
            elif not labels_base_np.flags["C_CONTIGUOUS"]:
                labels_base_np = np.ascontiguousarray(labels_base_np)
            indirect_positions = np.ascontiguousarray(
                features.index.get_indexer(sample_index), dtype=np.int32)
        elif sample_index is features.index and labels.index.equals(features.index):
            # sprint-6: samp_frac >= 1 sets sample_index = features.index
            # (same object), making .loc[sample_index] an identity reindex
            # that costs a full pandas take per fit (~10% of a small mDT
            # fit). Skip it — same rows, same order, bit-exact. The
            # labels.index.equals check guards the (never-seen) case of a
            # labels frame whose row order differs from features.
            features_np = np.ascontiguousarray(features.to_numpy())
            labels_np = np.ascontiguousarray(labels.to_numpy())
        else:
            features_np = np.ascontiguousarray(features.loc[sample_index].to_numpy())
            labels_np = np.ascontiguousarray(labels.loc[sample_index].to_numpy())

        if self.prebin:
            if precomputed_bins is not None:
                g_bins, g_bmax, g_bmin, g_unique = precomputed_bins
                # subset bin assignments to match sampled rows
                positions = features.index.get_indexer(sample_index)
                g_bins = np.ascontiguousarray(g_bins[positions])
            else:
                g_bins, g_bmax, g_bmin, g_unique = precompute_global_bins(
                    features_np, self.n_discrete_lev)

            tree_info, split_vals, node_means = fit_mt_tree_prebin(
                features_np, labels_np,
                self.weights_target, self.target_weight_map, self.n_objective,
                self.objective_weights, self.max_depth, self.mtry,
                self.samp_feature_by_node,
                self.min_samples_leaf, self.num_nodes, self.gain_threshold,
                self.n_discrete_lev, self.num_threads, self.verbose,
                np.ascontiguousarray(g_bins),
                np.ascontiguousarray(g_bmax),
                np.ascontiguousarray(g_bmin),
                g_unique,
                0 if self.aggregation == 'rank' else 1)
        elif _use_indirect:
            tree_info, split_vals, node_means = _fit_mt_tree_bin_indirect(
                features_base_np, labels_base_np, indirect_positions,
                self.weights_target, self.target_weight_map, self.n_objective,
                self.objective_weights, self.max_depth, self.mtry,
                self.samp_feature_by_node,
                self.min_samples_leaf, self.num_nodes, self.gain_threshold,
                self.n_discrete_lev, self.num_threads, self.verbose,
                0 if self.aggregation == 'rank' else 1,
                obj_weight_mode_int,
                self.obj_weight_dirichlet_alpha, rng, cxx_rng_seed)
        else:
            tree_info, split_vals, node_means = fit_mt_tree_bin(
                features_np, labels_np,
                self.weights_target, self.target_weight_map, self.n_objective,
                self.objective_weights, self.max_depth, self.mtry,
                self.samp_feature_by_node,
                self.min_samples_leaf, self.num_nodes, self.gain_threshold,
                self.n_discrete_lev, self.num_threads, self.verbose,
                0 if self.aggregation == 'rank' else 1,
                obj_weight_mode_int,
                self.obj_weight_dirichlet_alpha, rng, cxx_rng_seed)

        self.tree_info = tree_info
        self.split_vals = split_vals
        self.node_means = node_means

    def strip_fit_data(self):
        """Remove attributes only needed during fit to reduce memory."""
        for attr in ('weights_target', 'target_weight_map', 'objective_weights',
                     'n_objective', 'lambda_decay', 'num_nodes',
                     'samp_frac', 'mtry', 'samp_feature_by_node',
                     'min_samples_leaf', 'gain_threshold', 'n_discrete_lev', 'prebin'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def predict(self, features, depth=None, leaf_id=False):
        if depth is None:
            depth = self.max_depth
        # Normally fit() already resolved "auto"; guard the rare predict-first
        # path (e.g. a deserialized tree). predict parallelism is over rows, but
        # reusing the feature-count heuristic is a safe, conservative choice.
        if self.num_threads == "auto":
            self.num_threads = _resolve_auto_num_threads(features.shape[1])

        if leaf_id:
            prediction = predict_mt_tree_node(
                np.ascontiguousarray(features.to_numpy()),
                self.tree_info, self.split_vals, self.node_means, depth, self.verbose,
                self.num_threads)
        else:
            prediction = predict_mt_tree(
                np.ascontiguousarray(features.to_numpy()),
                self.tree_info, self.split_vals, self.node_means, depth, self.verbose,
                self.num_threads)
        return np.asarray(prediction)

    def tune_penalties(self, data, decay_set=None, lambda_set=None, beta_set=None,
                       nfolds=5, eval_fun=rmse, eval_depths=[], cv_type='grouped', seed=5,
                       n_jobs_cv=1):
        if cv_type == 'grouped':
            kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
            split = list(kfold.split(data.x, data.index.series))
        elif cv_type == 'timeseries':
            split = _time_series_split_per_series(data, nfolds=nfolds)
            if len(split) == 0:
                # Series too short for any rolling-origin fold; fall back to
                # series-grouped k-fold (random) so tuning can proceed. This
                # is logged so the caller knows the protocol degraded.
                print(f"  [tune_penalties] TSCV produced 0 folds; falling back "
                      f"to grouped k-fold (n_per_series may be too short for K).",
                      flush=True)
                kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
                split = list(kfold.split(data.x, data.index.series))
        else:
            kfold = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
            split = list(kfold.split(data.x))

        decay_set = decay_set[::-1]
        lambda_set = lambda_set[::-1]
        beta_set = beta_set[::-1]

        param_grid = {'lambda_decay': decay_set, 'lambda': lambda_set, 'beta': beta_set}
        eval_params = list(ParameterGrid(param_grid))

        depth_levels = [int(x) for x in eval_depths] if len(eval_depths) > 0 else [self.max_depth]

        # M5: build the flat task list (param × fold). Each task gets a
        # deterministic seed under the M0 RNG contract so n_jobs_cv=1 matches
        # n_jobs_cv>1 bit-identically.
        from .rng import per_tree_seeds
        all_tasks_count = len(eval_params) * len(split)
        task_seeds_flat = per_tree_seeds(seed, all_tasks_count)
        tasks = []
        for p_idx, param in enumerate(eval_params):
            for f_idx, (train_index, test_index) in enumerate(split):
                tasks.append({
                    "param": param,
                    "fold_idx": f_idx,
                    "train_index": train_index,
                    "test_index": test_index,
                    "task_seed": task_seeds_flat[p_idx * len(split) + f_idx],
                    "eval_fun": eval_fun,
                })

        init_kwargs = dict(
            max_depth=self.max_depth, samp_frac=self.samp_frac,
            mtry=self.mtry, samp_feature_by_node=self.samp_feature_by_node,
            min_samples_leaf=self.min_samples_leaf,
            gain_threshold=self.gain_threshold,
            n_discrete_lev=self.n_discrete_lev, num_threads=self.num_threads,
            verbose=self.verbose, prebin=self.prebin, aggregation=self.aggregation,
        )

        if n_jobs_cv != 1:
            from joblib import Parallel, delayed
            # return_as='generator' streams per-task results so the parent
            # doesn't materialise all (27 hyperparams x 5 folds = 135 tasks)
            # before consuming. Per-task return is small (~10 rows x 5 cols),
            # but applying the same fix as RF.fit:749/770 is good hygiene and
            # keeps CV memory profile flat under heavy datasets.
            results = Parallel(n_jobs=n_jobs_cv, backend="loky",
                               return_as='generator')(
                delayed(_fit_one_cv_task)(task, data, init_kwargs, depth_levels)
                for task in tasks)
            cv_results = []
            for sublist in results:
                cv_results.extend(sublist)
                del sublist
        else:
            cv_results = []
            for task in tqdm(tasks, leave=False):
                cv_results.extend(
                    _fit_one_cv_task(task, data, init_kwargs, depth_levels))

        cv_log = pd.DataFrame(cv_results)
        summary = cv_log.groupby(['depth', 'lambda_decay', 'objective_weights'])[
            ['train_error', 'test_error']].mean()
        best_params = list(summary['test_error'].idxmin())
        best_params = {"depth": best_params[0], "lambda": best_params[1], "weights": ast.literal_eval(best_params[2])}
        best_params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in best_params.items()}

        gc.collect()
        return best_params, cv_log, eval_params, summary
