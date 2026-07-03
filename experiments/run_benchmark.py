"""
Main benchmark experiment script.

Usage:
    python experiments/run_benchmark.py [config_file]

Reads configuration from configs/benchmark.yaml by default, or from the
specified YAML file.  Datasets must be placed in the directory specified by
data_path in the config.  Results are written to the directory specified by
results_path.
"""

import gc
import json
import os
import sys
import timeit
from os.path import exists

import numpy as np
import pandas as pd
import yaml

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mttrees.tree import DataMt, CTree_MT, MT_DTYPE
from mttrees.ensemble import BDT, RF
from mttrees.utils import (
    organize_repo_data, cyclic_transform, msmape, row_wise_mase, rmse,
    write_list_to_file, shutdown_worker_pool, compute_seasonal_scale,
)

pd.set_option('mode.chained_assignment', None)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_features_and_targets(combined_data, target_name, lag, horizon,
                                diff_features, time_series_cols, season_cols, ext_features,
                                frequency, max_seasonality):
    """OPT: vectorized numpy lag/target construction.

    Replaces the original list-of-Series + pd.concat pattern. For high-horizon
    datasets (e.g., Kaggle Daily with Cov, horizon=59) this saves ~3.5 GB of
    peak memory during data prep. Bit-identical to the original; see
    smoke_test_build_features.py.
    """
    # Requires combined_data sorted by ['series', 'timestamp'] — guaranteed by
    # organize_repo_data. We exploit the contiguous-by-series layout for direct
    # numpy indexing instead of pandas groupby/shift.
    # OPT memory: allocate lag/target matrices in MT_DTYPE so downstream
    # DataMt.transform doesn't need to cast (avoids holding fp64 + fp32 copies
    # simultaneously).
    y = combined_data[target_name].to_numpy(dtype=MT_DTYPE)
    n = len(y)

    row_in_series = combined_data.groupby('series').cumcount().to_numpy()
    sizes = combined_data.groupby('series').size().to_numpy()
    if hasattr(combined_data['series'], 'cat'):
        codes = combined_data['series'].cat.codes.to_numpy()
    else:
        codes = pd.Categorical(combined_data['series']).codes
    series_len_per_row = sizes[codes]

    # Lag features: lag_i[j] = y[j-(i+1)] iff row_in_series[j] >= (i+1)
    lag_matrix = np.full((n, lag), np.nan, dtype=MT_DTYPE)
    for i in range(lag):
        k = i + 1
        valid = row_in_series >= k
        if valid.any():
            idx = np.flatnonzero(valid)
            lag_matrix[idx, i] = y[idx - k]
    lag_feature_names = ['lag_' + str(i + 1) for i in range(lag)]

    if diff_features:
        diff_matrix = np.full((n, lag - 1), np.nan, dtype=MT_DTYPE)
        for i in range(1, lag):
            k1, k2 = i + 1, i
            valid = row_in_series >= k1
            if valid.any():
                idx = np.flatnonzero(valid)
                diff_matrix[idx, i - 1] = y[idx - k1] - y[idx - k2]
        diff_feature_names = ['diff_lag_' + str(i + 1) for i in range(1, lag)]
    else:
        diff_matrix = None

    # Target features: ahead_i[j] = y[j+i] iff (row_in_series[j] + i) < series_len
    target_matrix = np.full((n, horizon), np.nan, dtype=MT_DTYPE)
    for i in range(horizon):
        valid = (row_in_series + i) < series_len_per_row
        if valid.any():
            idx = np.flatnonzero(valid)
            target_matrix[idx, i] = y[idx + i]
    target_names = ['ahead_' + str(i) for i in range(horizon)]

    # Mask for rows where ALL lag and target columns are non-NaN
    full_valid = (~np.isnan(lag_matrix).any(axis=1)) & (~np.isnan(target_matrix).any(axis=1))

    base_index = combined_data.index
    meta_cols_lag = time_series_cols + season_cols + ext_features
    lag_features_df = pd.DataFrame(lag_matrix, columns=lag_feature_names, index=base_index)
    if diff_matrix is not None:
        lag_features_df = pd.concat(
            [lag_features_df,
             pd.DataFrame(diff_matrix, columns=diff_feature_names, index=base_index)],
            axis=1)
    lag_features_global = pd.concat(
        [combined_data[meta_cols_lag], lag_features_df], axis=1).loc[full_valid].dropna()

    target_features_df = pd.DataFrame(target_matrix, columns=target_names, index=base_index)
    target_features_global = pd.concat(
        [combined_data[time_series_cols], target_features_df], axis=1).loc[full_valid].dropna()

    # OPT memory: free the intermediate numpy / DataFrame views now that the
    # global frames are built. Without this, the wide DataFrames remain alive
    # alongside the matrices through downstream operations.
    del lag_matrix, target_matrix, lag_features_df, target_features_df
    if diff_matrix is not None:
        del diff_matrix
    gc.collect()

    if frequency != '1Y':
        lag_features_global['season'] = (lag_features_global['season_index'] % max_seasonality) + 1
        lag_features_global = cyclic_transform(lag_features_global, 'season')
        lag_features_global.drop(['season', 'season_index'], axis=1, inplace=True)
    else:
        lag_features_global.drop(['season_index'], axis=1, inplace=True)

    return lag_features_global, target_features_global, lag_feature_names


def mean_scale_data(train_features, test_features, train_targets, test_targets,
                    lag, horizon, mean_scale):
    series_means = train_features.groupby('series')['lag_1'].mean().reset_index()
    if mean_scale:
        series_means.loc[series_means['lag_1'] == 0, 'lag_1'] = np.mean(series_means['lag_1'])
    else:
        series_means['lag_1'] = 1
    series_means.rename(columns={'lag_1': 'mean_series'}, inplace=True)

    train_features = train_features.merge(series_means, on='series')
    test_features = test_features.merge(series_means, on='series')
    train_targets = train_targets.merge(series_means, on='series')
    test_targets = test_targets.merge(series_means, on='series')

    train_features.iloc[:, range(2, 2 + lag)] = \
        train_features.iloc[:, range(2, 2 + lag)].div(train_features.mean_series, axis=0)
    test_features.iloc[:, range(2, 2 + lag)] = \
        test_features.iloc[:, range(2, 2 + lag)].div(test_features.mean_series, axis=0)
    train_targets.iloc[:, range(2, 2 + horizon)] = \
        train_targets.iloc[:, range(2, 2 + horizon)].div(train_targets.mean_series, axis=0)
    test_targets.iloc[:, range(2, 2 + horizon)] = \
        test_targets.iloc[:, range(2, 2 + horizon)].div(test_targets.mean_series, axis=0)

    for df in [train_features, test_features, train_targets, test_targets]:
        df.drop(['mean_series'], axis=1, inplace=True)

    return train_features, test_features, train_targets, test_targets, series_means


def _safe_aggregate(arr):
    """Monash-style aggregation: drop inf and NaN, then mean + median.

    Matches the filter in https://github.com/rakshitha123/TSForecasting
    utils/error_calculator.R:
        mase_per_series <- mase_per_series[!is.infinite(mase_per_series)
                                           & !is.na(mase_per_series)]
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.median(a))


def _load_family_metrics_from_csv(forecast_file, actual_file, seasonal_scale):
    """Recompute (msmape, mase) for a single family from its saved CSVs.

    Used by the per-family resume path: when a family's forecast CSV already
    exists from a prior partial run, we skip the (expensive) fit+predict and
    just re-derive the metrics so the timing/accuracy log stays complete.
    """
    f_df = pd.read_csv(forecast_file)
    a_df = pd.read_csv(actual_file)
    hcols = [c for c in f_df.columns if str(c).isdigit()]
    yhat = f_df[hcols].values.astype(np.float64)
    y    = a_df[hcols].values.astype(np.float64)
    s = seasonal_scale.loc[f_df['series']]
    fam_msmape = _safe_aggregate(msmape(y, yhat))
    fam_mase   = list(_safe_aggregate(row_wise_mase(y, yhat, s)))
    return fam_msmape, fam_mase


def run_experiment(cfg, dataset_filter=None):
    experiment_folder = cfg['results_path']
    forecast_folder = os.path.join(experiment_folder, 'forecasts')
    data_path = cfg['data_path']
    tuning_folder = os.path.join(experiment_folder, 'tuning')
    os.makedirs(tuning_folder, exist_ok=True)

    result_prefix = cfg.get('result_prefix', 'run')
    prebin = cfg.get('prebin', True)
    force_rerun = cfg.get('force_rerun', False)
    info_table = pd.read_excel(cfg['info_table'], 'repo_data')

    if dataset_filter is not None:
        info_table = info_table[info_table.Dataset == dataset_filter].reset_index(drop=True)
        if len(info_table) == 0:
            sys.exit(f"--dataset {dataset_filter!r} not found in info_table")
        print(f"[run_benchmark] restricted to dataset: {dataset_filter}", flush=True)

    nthreads = cfg.get('nthreads', 1)
    # CV parallelism over the (param x fold) task list. tune_penalties is
    # embarrassingly parallel by design — each task is independent and uses a
    # deterministic per-task seed (M0 RNG contract) so n_jobs_cv > 1 is
    # bit-identical to n_jobs_cv == 1.
    n_jobs_cv = cfg.get('n_jobs_cv', 1)
    test_horizon = cfg.get('test_horizon', 1)
    n_diff = cfg.get('n_diff', 1)
    target_name = 'y'
    diff_features = cfg.get('diff_features', False)
    max_depth = cfg['max_depth']
    min_samples_leaf = cfg['min_samples_leaf']
    n_level = cfg['n_discrete_lev']
    decay_set = cfg['decay_set']
    lambda_set = cfg['lambda_set']
    beta_org_set = cfg['beta_set']
    nfolds = cfg['nfolds']
    rnd_seed = cfg.get('seed', 10)
    cv_type = cfg.get('cv_type', 'grouped')
    cv_metric = cfg.get('cv_metric', 'rmse').lower()
    aggregation = cfg.get('aggregation', 'rank').lower()
    if cv_metric not in ('rmse', 'mase', 'msmape'):
        raise ValueError(f"cv_metric must be 'rmse', 'mase', or 'msmape'; got {cv_metric!r}")
    if aggregation not in ('rank', 'score'):
        raise ValueError(f"aggregation must be 'rank' or 'score'; got {aggregation!r}")
    print(f"[run_benchmark] cv_metric={cv_metric}  aggregation={aggregation}", flush=True)
    min_depth = cfg['min_depth']
    depth_increment = cfg['depth_increment']
    eval_depth_lev = np.arange(min_depth, max_depth + 1, depth_increment).tolist()

    default_bdt_params = {
        "random_state": rnd_seed,
        "n_estimators_": cfg['bdt']['n_estimators'],
        "lr": cfg['bdt']['lr'],
        "bagging_fraction": cfg['bdt']['bagging_fraction'],
        "feature_fraction": cfg['bdt']['feature_fraction'],
        "max_depth": cfg['bdt']['max_depth'],
        "min_samples_leaf": min_samples_leaf,
        "n_discrete_lev": n_level,
        "num_threads": cfg['bdt'].get('num_threads', nthreads),
        "prebin": prebin,
        "aggregation": aggregation,
        # ES eval metric (default 'mse'); 'mase' matches the paper and needs
        # eval_seasonal_scale, passed unconditionally at the BDT call site below.
        "eval_metric": cfg['bdt'].get('eval_metric', 'mse'),
    }

    rf_default_params = {
        "random_state": rnd_seed,
        "n_estimators_": cfg['rf']['n_estimators'],
        "bagging_fraction": cfg['rf']['bagging_fraction'],
        "feature_fraction": cfg['rf']['feature_fraction'],
        "max_depth": cfg['rf']['max_depth'],
        "min_samples_leaf": min_samples_leaf,
        "n_discrete_lev": n_level,
        "num_threads": cfg['rf'].get('num_threads', nthreads),
        "prebin": prebin,
        "n_jobs": cfg['rf'].get('n_jobs', 1),
        # backend: 'thread' (default, flat memory) | 'process' | 'sequential'.
        # Legacy mmap_data is only meaningful under the process backend.
        "backend": cfg['rf'].get('backend', 'thread'),
        "aggregation": aggregation,
    }
    if rf_default_params["backend"] == "process" and 'mmap_data' in cfg['rf']:
        rf_default_params["mmap_data"] = cfg['rf']['mmap_data']

    time_series_cols = ['index', 'series']
    season_cols = ['season_index']
    n_datasets = cfg.get('n_datasets', len(info_table))

    for i in range(n_datasets):
        mean_scale = True
        print(info_table.Dataset[i])

        if info_table.run[i] == 0:
            continue

        if info_table.Dataset[i] == 'Carparts':
            mean_scale = False

        integer_conversion = info_table.integer_conversion[i]
        data_set = info_table.Dataset[i]
        data_file = os.path.join(data_path, info_table.file_name[i])

        forecast_file = os.path.join(forecast_folder, 'dt', f'{data_set}_{result_prefix}_forecast.csv')
        forecast_file_rf = os.path.join(forecast_folder, 'rf', f'{data_set}_{result_prefix}_forecast.csv')
        forecast_file_gbt = os.path.join(forecast_folder, 'gbt', f'{data_set}_{result_prefix}_forecast.csv')

        if not force_rerun and exists(forecast_file) and exists(forecast_file_rf) and exists(forecast_file_gbt):
            continue

        combined_data, frequency, seasonality, horizon, ext_features = organize_repo_data(data_file)

        if info_table.horizon[i] is not np.nan:
            horizon = int(info_table.horizon[i])

        max_seasonality = max(seasonality)

        if info_table.predetermined_lag[i] is not np.nan:
            lag = int(info_table.predetermined_lag[i])
        else:
            lag = int(np.ceil(max_seasonality * 1.25))

        max_ahead = int(horizon)
        max_seasonality = int(max_seasonality)
        print([lag, horizon, seasonality, frequency])

        # Seasonal scale for MASE -- Monash-style. Drop the FULL forecast
        # horizon from each series (the test set is the last `horizon` raw
        # observations, not the last `test_horizon` wide-format rows).
        # Pre-fix this used `test_horizon`, leaking horizon-1 test rows into
        # the scale denominator. Exercised by
        # tests/test_phase2.py::test_seasonal_scale_drops_full_horizon.
        seasonal_scale = compute_seasonal_scale(
            combined_data, target_name,
            drop_last=int(horizon),
            max_seasonality=max_seasonality)
        gc.collect()

        # Build features and targets
        lag_features_global, target_features_global, lag_feature_names = build_features_and_targets(
            combined_data, target_name, lag, horizon, diff_features,
            time_series_cols, season_cols, ext_features, frequency, max_seasonality)
        del combined_data
        gc.collect()

        # Train/test split
        grouped = lag_features_global.groupby('series').cumcount(ascending=False)
        train_features = lag_features_global[grouped >= test_horizon]
        test_features = lag_features_global[grouped < test_horizon]

        grouped = target_features_global.groupby('series').cumcount(ascending=False)
        train_targets = target_features_global[grouped >= test_horizon]
        test_targets = target_features_global[grouped < test_horizon]

        # Mean scaling
        train_features, test_features, train_targets, test_targets, series_means = \
            mean_scale_data(train_features, test_features, train_targets, test_targets,
                            lag, horizon, mean_scale)
        # Quick win #2: release the pre-split global feature/target frames now
        # that the train/test partitions and series_means are independent.
        del lag_features_global, target_features_global
        gc.collect()

        # Seasonal penalty targets
        if frequency == '1Y':
            seasonal_params = None
            beta_set = [0]
        else:
            seasonal_params = ['sin_season', 'cos_season']
            beta_set = beta_org_set

        train_data = DataMt(max_ahead=max_ahead, n_derivative=n_diff,
                            penalty_list=seasonal_params, series_means=series_means,
                            int_convert=integer_conversion)
        train_data.transform(train_features, train_targets)

        test_data = DataMt(max_ahead=max_ahead, n_derivative=n_diff,
                           penalty_list=seasonal_params, series_means=series_means,
                           int_convert=integer_conversion)
        test_data.transform(test_features, test_targets)

        # Tune hyperparameters (or load cached)
        tuned_params_file = os.path.join(tuning_folder, f'{data_set}_{result_prefix}.json')
        # CV phase: pin per-task OpenMP to 1 so the joblib n_jobs_cv parallelism
        # across (param x fold) tasks does not race with per-fit OpenMP threads.
        # Each CV task is deterministic via per_tree_seeds (M0 RNG contract), so
        # n_jobs_cv > 1 is bit-identical to n_jobs_cv == 1.
        tree = CTree_MT(max_depth=max_depth, verbose=0, min_samples_leaf=min_samples_leaf,
                        n_discrete_lev=n_level, num_threads=1, prebin=prebin,
                        aggregation=aggregation)

        # Build the CV eval function. tune_penalties expects fn(y_true, y_pred) -> per-row vector
        # of length N; the function aggregates via np.nanmean across fold rows and across folds,
        # so any NaN slots are correctly dropped (matches the Monash test-time _safe_aggregate).
        # row_wise_mase needs the per-row seasonal scale; index it once here and close over it.
        # For series with zero seasonal_scale (e.g. truly constant series like 33 of 266 in
        # COVID Deaths), the per-row MASE is inf; convert to NaN here so nanmean drops them.
        if cv_metric == 'mase':
            _row_scale = seasonal_scale.loc[train_data.index.series].values.astype(np.float64)
            def cv_eval_fun(target, prediction, _scale=_row_scale):
                out = row_wise_mase(target, prediction, _scale)
                return np.where(np.isfinite(out), out, np.nan)
            cv_eval_fun.__name__ = 'mase'
        elif cv_metric == 'msmape':
            cv_eval_fun = msmape
        else:
            cv_eval_fun = rmse

        if force_rerun or not exists(tuned_params_file):
            best_params, cv_log, _, _ = tree.tune_penalties(
                train_data, decay_set=decay_set, lambda_set=lambda_set, beta_set=beta_set,
                eval_depths=eval_depth_lev, cv_type=cv_type, nfolds=nfolds, seed=rnd_seed,
                eval_fun=cv_eval_fun, n_jobs_cv=n_jobs_cv)
            with open(tuned_params_file, "w") as f:
                json.dump(best_params, f)
            cv_log.to_csv(
                os.path.join(tuning_folder, f'{data_set}_{result_prefix}_cv_log.csv'), index=False)
        else:
            with open(tuned_params_file) as f:
                best_params = json.load(f)

        print(best_params)
        best_depth = best_params['depth']
        obj_weights = best_params['weights']
        best_lambda = best_params['lambda']

        # --- mDT ---
        actual_file_dt = os.path.join(forecast_folder, 'dt', f'{data_set}_{result_prefix}_actual.csv')
        if force_rerun or not exists(forecast_file):
            tree = CTree_MT(max_depth=best_depth, verbose=0, min_samples_leaf=min_samples_leaf,
                            n_discrete_lev=n_level, num_threads=nthreads, prebin=prebin,
                            aggregation=aggregation)
            tree.arrangeObjective(train_data, best_lambda, obj_weights)
            start_time = timeit.default_timer()
            tree.fit(train_data.x, train_data.y, random_state=rnd_seed)
            dt_time = timeit.default_timer() - start_time
            print(f'Tree build time: {dt_time:.2f}s')

            test_predictions = tree.predict(test_data.x)
            test_predictions, _ = test_data.process_predictions(test_predictions)

            os.makedirs(os.path.join(forecast_folder, 'dt'), exist_ok=True)
            pd.concat([pd.DataFrame(test_predictions), test_data.index], axis=1).to_csv(
                forecast_file, index=False)
            pd.concat([pd.DataFrame(test_data.y_org), test_data.index], axis=1).to_csv(
                actual_file_dt, index=False)

            dt_msmape = _safe_aggregate(msmape(test_data.y_org, test_predictions))
            dt_mase = list(_safe_aggregate(row_wise_mase(
                test_data.y_org, test_predictions,
                seasonal_scale.loc[test_data.index.series])))
            print('mDT msMAPE:', dt_msmape)
            print('mDT MASE:', dt_mase)
            del tree
            gc.collect()
        else:
            print(f'mDT skipped (forecast exists at {forecast_file})')
            dt_time = float('nan')
            dt_msmape, dt_mase = _load_family_metrics_from_csv(
                forecast_file, actual_file_dt, seasonal_scale)
            print('mDT msMAPE (from saved CSV):', dt_msmape)
            print('mDT MASE (from saved CSV):', dt_mase)

        # --- mGBT ---
        actual_file_gbt = os.path.join(forecast_folder, 'gbt', f'{data_set}_{result_prefix}_actual.csv')
        if force_rerun or not exists(forecast_file_gbt):
            # eval_seasonal_scale is required when yaml/bdt sets eval_metric='mase';
            # ignored otherwise. Pass unconditionally so the yaml knob works.
            bdt = BDT(train_data, lambda_decay=best_lambda, objective_weights=obj_weights,
                      eval_seasonal_scale=seasonal_scale,
                      **default_bdt_params)
            start_time = timeit.default_timer()
            bdt.fit()
            bdt_time = timeit.default_timer() - start_time
            print(f'mGBT build time: {bdt_time:.2f}s')
            test_predictions = bdt.predict(test_data.x)
            test_predictions, _ = test_data.process_predictions(test_predictions)

            os.makedirs(os.path.join(forecast_folder, 'gbt'), exist_ok=True)
            pd.concat([pd.DataFrame(test_predictions), test_data.index], axis=1).to_csv(
                forecast_file_gbt, index=False)
            pd.concat([pd.DataFrame(test_data.y_org), test_data.index], axis=1).to_csv(
                actual_file_gbt, index=False)

            bdt_msmape = _safe_aggregate(msmape(test_data.y_org, test_predictions))
            bdt_mase = list(_safe_aggregate(row_wise_mase(
                test_data.y_org, test_predictions,
                seasonal_scale.loc[test_data.index.series])))
            del bdt
            gc.collect()
        else:
            print(f'mGBT skipped (forecast exists at {forecast_file_gbt})')
            bdt_time = float('nan')
            bdt_msmape, bdt_mase = _load_family_metrics_from_csv(
                forecast_file_gbt, actual_file_gbt, seasonal_scale)
            print('mGBT msMAPE (from saved CSV):', bdt_msmape)
            print('mGBT MASE (from saved CSV):', bdt_mase)

        # --- mRF ---
        actual_file_rf = os.path.join(forecast_folder, 'rf', f'{data_set}_{result_prefix}_actual.csv')
        if force_rerun or not exists(forecast_file_rf):
            rf = RF(train_data, lambda_decay=best_lambda, objective_weights=obj_weights,
                    **rf_default_params)
            start_time = timeit.default_timer()
            rf.fit()
            rf_time = timeit.default_timer() - start_time
            print(f'mRF build time: {rf_time:.2f}s')
            test_predictions = rf.predict(test_data.x)
            test_predictions, _ = test_data.process_predictions(test_predictions)

            os.makedirs(os.path.join(forecast_folder, 'rf'), exist_ok=True)
            pd.concat([pd.DataFrame(test_predictions), test_data.index], axis=1).to_csv(
                forecast_file_rf, index=False)
            pd.concat([pd.DataFrame(test_data.y_org), test_data.index], axis=1).to_csv(
                actual_file_rf, index=False)

            rf_msmape = _safe_aggregate(msmape(test_data.y_org, test_predictions))
            rf_mase = list(_safe_aggregate(row_wise_mase(
                test_data.y_org, test_predictions,
                seasonal_scale.loc[test_data.index.series])))
            del rf
            gc.collect()
        else:
            print(f'mRF skipped (forecast exists at {forecast_file_rf})')
            rf_time = float('nan')
            rf_msmape, rf_mase = _load_family_metrics_from_csv(
                forecast_file_rf, actual_file_rf, seasonal_scale)
            print('mRF msMAPE (from saved CSV):', rf_msmape)
            print('mRF MASE (from saved CSV):', rf_mase)

        def _fmt_t(t):
            return 'skipped' if np.isnan(t) else f'{t:.2f}s'

        print(f'\n--- {data_set} results ---')
        print(f'mDT  msMAPE: {dt_msmape}  MASE: {dt_mase}  time: {_fmt_t(dt_time)}')
        print(f'mGBT msMAPE: {bdt_msmape}  MASE: {bdt_mase}  time: {_fmt_t(bdt_time)}')
        print(f'mRF  msMAPE: {rf_msmape}  MASE: {rf_mase}  time: {_fmt_t(rf_time)}')

        # Save timing and accuracy log
        timing_record = {
            'dataset': data_set,
            'prebin': prebin,
            'dt_time': dt_time,
            'bdt_time': bdt_time,
            'rf_time': rf_time,
            'dt_msmape_mean': dt_msmape[0], 'dt_msmape_median': dt_msmape[1],
            'dt_mase_mean': dt_mase[0], 'dt_mase_median': dt_mase[1],
            'bdt_msmape_mean': bdt_msmape[0], 'bdt_msmape_median': bdt_msmape[1],
            'bdt_mase_mean': bdt_mase[0], 'bdt_mase_median': bdt_mase[1],
            'rf_msmape_mean': rf_msmape[0], 'rf_msmape_median': rf_msmape[1],
            'rf_mase_mean': rf_mase[0], 'rf_mase_median': rf_mase[1],
        }
        timing_log_file = os.path.join(experiment_folder, f'{result_prefix}_timing.csv')
        timing_df = pd.DataFrame([timing_record])
        if exists(timing_log_file):
            timing_df.to_csv(timing_log_file, mode='a', header=False, index=False)
        else:
            timing_df.to_csv(timing_log_file, index=False)

        # Quick win #3: explicit cleanup of per-dataset heavy objects before
        # the next iteration. Without this the python allocator keeps the
        # high-water-mark pages until natural GC; on the heaviest datasets
        # (Favorita, Kaggle, M5) that's hundreds of MB held unnecessarily.
        del train_data, test_data
        del train_features, test_features, train_targets, test_targets
        del series_means, seasonal_scale
        del timing_df, timing_record
        gc.collect()
        # Recycle the joblib loky pool so worker high-water-mark RSS does not
        # accumulate across datasets. Critical on macOS where the carried-over
        # footprint trips jetsam on heavy datasets (Favorita / M5 / Rossmann).
        shutdown_worker_pool()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="mt-forecast benchmark runner")
    parser.add_argument("config", nargs="?", default=None,
                        help="path to YAML config (default: configs/benchmark.yaml)")
    parser.add_argument("--dataset", default=None,
                        help="restrict the run to a single dataset by name (e.g. 'M1 Yearly')")
    parser.add_argument("--cv-metric", default=None, choices=['rmse', 'mase', 'msmape'],
                        help="override cfg.cv_metric (rmse, mase, or msmape)")
    parser.add_argument("--aggregation", default=None, choices=['rank', 'score'],
                        help="override cfg.aggregation (rank or score)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.config is not None:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(repo_root, config_path)
    else:
        config_path = os.path.join(repo_root, 'configs', 'benchmark.yaml')
    cfg = load_config(config_path)
    if args.cv_metric is not None:
        cfg['cv_metric'] = args.cv_metric
    if args.aggregation is not None:
        cfg['aggregation'] = args.aggregation
    run_experiment(cfg, dataset_filter=args.dataset)
