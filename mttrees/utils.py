from statsmodels.tsa.tsatools import lagmat
try:
    from distutils.util import strtobool
except ImportError:
    # distutils removed in Python 3.12+
    def strtobool(val):
        val = val.strip().lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        if val in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        raise ValueError(f"invalid truth value {val!r}")
from datetime import datetime
from os.path import exists
import itertools
import gc

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


SEASONALITY_MAP = {
   "minutely": [1440, 10080, 525960],
   "10_minutes": [144, 1008, 52596],
   "half_hourly": [48, 336, 17532],
   "hourly": [24, 168, 8766],
   "daily": [7],
   "weekly": [365.25/7],
   "monthly": [12],
   "quarterly": [4],
   "yearly": [1]
}

FREQUENCY_MAP = {
   "minutely": "1min",
   "10_minutes": "10min",
   "half_hourly": "30min",
   "hourly": "1H",
   "daily": "1D",
   "weekly": "1W",
   "monthly": "1M",
   "quarterly": "1Q",
   "yearly": "1Y"
}

UNIT_MAP = {
    "minutely": "minutes",
    "10_minutes": "minutes",
    "half_hourly": "minutes",
    "hourly": "hours",
    "daily": "days",
    "weekly": "weeks",
    "monthly": "months",
    "quarterly": "months",
    "yearly": "years"
}

CNT_MAP = {
    "minutely": 1,
    "10_minutes": 10,
    "half_hourly": 30,
    "hourly": 1,
    "daily": 1,
    "weekly": 1,
    "monthly": 1,
    "quarterly": 3,
    "yearly": 1
}


def final_evaluation_from_file(forecast_file, actual_file, method, integer_conversion,
                                horizon, seasonal_scale, series_index,
                                divide_periods=1, type="csv"):
    temp_result = []

    if not exists(forecast_file):
        for metric in ['msMAPE', 'MASE', 'RMSE']:
            for agg in ['mean', 'median']:
                temp_result.append({'method': method, 'type': metric, 'agg': agg, 'value': None})
    else:
        if type == 'csv':
            forecasts_df = pd.read_csv(forecast_file)
            forecasts = forecasts_df[[str(x) for x in range(horizon)]].to_numpy()
        elif type == 'txt':
            np_arr = np.loadtxt(forecast_file)
            if np_arr.ndim == 1:
                np_arr = np.reshape(np_arr, (-1, len(np_arr)))
            forecasts_df = pd.DataFrame(np_arr)
            forecasts_df['series'] = ['T' + str(i + 1) for i in range(forecasts_df.shape[0])]
            forecasts_df = forecasts_df.sort_values(['series'])
            forecasts = forecasts_df[[x for x in range(horizon)]].to_numpy()

        actual_df = pd.read_csv(actual_file)

        if integer_conversion:
            forecasts = np.round(forecasts)
        forecasts[forecasts < 0] = 0

        actual = actual_df[[str(x) for x in range(horizon)]].to_numpy()

        multi_period = int(np.ceil(horizon / divide_periods))
        i = 0
        if actual.shape == forecasts.shape:
            for i in range(divide_periods):
                start_x = int(np.ceil(i * multi_period))
                end_x = int(min(np.ceil(i * multi_period) + multi_period, horizon))

                smape_metric_all = msmape(actual[:, start_x:end_x], forecasts[:, start_x:end_x])
                msmape_stats = np.nanmean(smape_metric_all), np.nanmedian(smape_metric_all)

                mase_metric_all = row_wise_mase(actual[:, start_x:end_x], forecasts[:, start_x:end_x],
                                                seasonal_scale.loc[series_index])
                mase_stats = [np.nanmean(mase_metric_all), np.nanmedian(mase_metric_all)]

                rmse_metric_all = rmse(actual[:, start_x:end_x], forecasts[:, start_x:end_x])
                rmse_stats = [np.nanmean(rmse_metric_all), np.nanmedian(rmse_metric_all)]

                temp_result.append({'method': method, 'horizon_period': i+1, 'type': 'msMAPE', 'agg': 'mean', 'value': msmape_stats[0]})
                temp_result.append({'method': method, 'horizon_period': i+1, 'type': 'msMAPE', 'agg': 'median', 'value': msmape_stats[1]})
                temp_result.append({'method': method, 'horizon_period': i+1, 'type': 'MASE', 'agg': 'mean', 'value': mase_stats[0]})
                temp_result.append({'method': method, 'horizon_period': i+1, 'type': 'MASE', 'agg': 'median', 'value': mase_stats[1]})
                temp_result.append({'method': method, 'horizon_period': i+1, 'type': 'RMSE', 'agg': 'mean', 'value': rmse_stats[0]})
                temp_result.append({'method': method, 'horizon_period': i+1, 'type': 'RMSE', 'agg': 'median', 'value': rmse_stats[1]})
        else:
            for metric in ['msMAPE', 'MASE', 'RMSE']:
                for agg in ['mean', 'median']:
                    temp_result.append({'method': method, 'horizon_period': i+1,
                                        'type': metric, 'agg': agg, 'value': None})

    return temp_result


def normalize_data(train_features, test_features, train_targets, test_targets, lag_feature_names,
                   target_names, external_mean_norm_features, mean_scale=False, zero_mean_series=[]):

    series_means = train_features.groupby('series')['lag_1'].mean().reset_index()
    series_means.rename(columns={'lag_1': 'mean_series'}, inplace=True)

    zero_mean_series = list(series_means[series_means['mean_series'] == 0].series)
    series_means.replace(to_replace=0, value=1, inplace=True)
    if not mean_scale:
        series_means['mean_series'] = 1
    else:
        if len(external_mean_norm_features):
            external_means = train_features.groupby('series')[external_mean_norm_features].mean().reset_index()

        train_features = train_features.merge(series_means, on='series')
        test_features = test_features.merge(series_means, on='series')
        train_targets = train_targets.merge(series_means, on='series')
        test_targets = test_targets.merge(series_means, on='series')
        train_features[lag_feature_names] = train_features[lag_feature_names].div(train_features.mean_series, axis=0)
        test_features[lag_feature_names] = test_features[lag_feature_names].div(test_features.mean_series, axis=0)
        train_targets[target_names] = train_targets[target_names].div(train_targets.mean_series, axis=0)
        test_targets[target_names] = test_targets[target_names].div(test_targets.mean_series, axis=0)
        train_features.drop(['mean_series'], axis=1, inplace=True)
        test_features.drop(['mean_series'], axis=1, inplace=True)
        train_targets.drop(['mean_series'], axis=1, inplace=True)
        test_targets.drop(['mean_series'], axis=1, inplace=True)

        if len(external_mean_norm_features):
            train_features = train_features.merge(external_means, on='series', suffixes=('', '_mean'))
            test_features = test_features.merge(external_means, on='series', suffixes=('', '_mean'))
            for feature in external_mean_norm_features:
                train_features[feature] /= train_features[feature + '_mean']
                test_features[feature] /= test_features[feature + '_mean']
            mean_cols = [feature + '_mean' for feature in external_mean_norm_features]
            train_features.drop(columns=mean_cols, inplace=True)
            test_features.drop(columns=mean_cols, inplace=True)

    gc.collect()
    return train_features, test_features, train_targets, test_targets, series_means, zero_mean_series


def discretize_features(X, num_levels):
    X_discrete = np.zeros_like(X, dtype=np.int64)
    bin_list = []
    level_counts = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        unique_values = np.unique(feature)
        if unique_values.size == 1:
            bins = (np.zeros_like(feature), unique_values)
        else:
            levels = min(unique_values.size, num_levels)
            bins = pd.qcut(feature, q=levels, labels=False, duplicates='drop', retbins=True)
        bin_list.append(bins[1])
        unique_bins = np.unique(bins[0])
        level_counts.append(len(unique_bins))
        mapping = {bin: i for i, bin in enumerate(unique_bins)}
        X_discrete[:, i] = np.vectorize(mapping.get)(bins[0])
    return X_discrete, bin_list, max(level_counts)


def calculate_quantiles(X, num_levels):
    percentiles = np.linspace(0, 100, num_levels + 1)[1:-1]
    quantiles = np.percentile(X, percentiles, axis=0)
    return quantiles


def reconstruct_features(val, variable, mapping_list):
    mapped = mapping_list[variable][val]
    return mapped


def ctype_conversion(A):
    return np.ctypeslib.as_ctypes(np.ascontiguousarray(A.T))


def write_list_to_file(to_write, fname):
    file = open(fname, 'w')
    for item in to_write:
        file.write(str(item) + "\t")
    file.close()


def std_agg(cnt, s1, s2):
    return np.sqrt((s2 / cnt) - (s1 / cnt) ** 2)


def cv(x):
    return np.std(x, ddof=1) / np.mean(x) * 100


def evaluate_forecast(fun, *args):
    return fun(*args)


def calculate_msmape(actual, forecast):
    epsilon = 0.1
    forecasts = np.array(forecast)
    test_set = np.array(actual)
    comparator = np.repeat((0.5 + epsilon), len(test_set))
    denom = np.maximum(comparator, (np.abs(forecasts) + np.abs(test_set) + epsilon))
    smape = 2 * np.abs(forecasts - test_set) / denom
    return 100 * np.nanmean(smape)


def rmse(target, target_prediction):
    mse = mean_squared_error(target.T, target_prediction.T,
                             multioutput='raw_values')
    return np.sqrt(mse)


def msmape(target, target_prediction, epsilon=0.1):
    assert target.shape == target_prediction.shape, \
        "Target and target_prediction arrays must have the same shape"
    comparator = np.full(target.shape, 0.5 + epsilon)
    sum_ = np.maximum(comparator, (np.abs(target) + np.abs(target_prediction) + epsilon))
    smape = 2 * np.abs(target - target_prediction) / sum_
    return 100 * np.mean(smape, axis=1, dtype=np.float64)


def MASE(test_set, forecasts, scale):
    return np.mean(np.abs(test_set - forecasts)) / scale


def row_wise_mase(actual, prediction, scale):
    assert prediction.shape == actual.shape, \
        "Prediction and actual arrays must have the same shape"
    assert len(scale) == prediction.shape[0], \
        "Scale vector length must match the number of instances"
    mae = np.mean(np.abs(prediction - actual), axis=1)
    return mae / scale


def calculate_mase(actual, forecast, training_actual, seasonality):
    if forecast.ndim == 1:
        forecasts = forecast.reshape(1, -1)
        test_set = actual.reshape(1, -1)
        training_set = training_actual.reshape(1, -1)
    else:
        forecasts = np.array(forecast)
        test_set = np.array(actual)
        training_set = np.array(training_actual)

    mase_per_series = np.empty((forecasts.shape[0],))
    for k in range(forecasts.shape[0]):
        training_series = training_set[k]
        seas = min(seasonality)
        if seas > len(training_series):
            seas = 1
        seas_diff = training_series[seas:] - training_series[:-seas]
        mean_abs_diff = np.mean(np.abs(seas_diff))
        mase_per_series[k] = MASE(test_set[k, :], forecasts[k, :], mean_abs_diff)
        if np.isnan(mase_per_series[k]):
            seas_diff = training_series[seas:] - training_series[:-seas]
            mean_abs_diff = np.mean(np.abs(seas_diff))
            mase_per_series[k] = MASE(test_set[k, :], forecasts[k, :], mean_abs_diff)
        if np.isnan(mase_per_series[k]):
            seas = 1
            seas_diff = training_series[seas:] - training_series[:-seas]
            mean_abs_diff = np.mean(np.abs(seas_diff))
            mase_per_series[k] = MASE(test_set[k, :], forecasts[k, :], mean_abs_diff)
    return mase_per_series


def calculate_mase_repo(actual, forecast, training_actual, seasonality):
    if forecast.ndim == 1:
        forecasts = forecast.reshape(1, -1)
        test_set = actual.reshape(1, -1)
        training_set = training_actual.reshape(1, -1)
    else:
        forecasts = np.array(forecast)
        test_set = np.array(actual)
        training_set = np.array(training_actual)

    mase_per_series = np.empty((forecasts.shape[0],))
    for k in range(forecasts.shape[0]):
        te = test_set[k,]
        te = te[~np.isnan(te)]
        tr = training_set[k]
        tr = tr[~np.isnan(tr)]
        f = forecasts[k,]
        f = f[~np.isnan(f)]
        mean_abs_diff = np.mean(np.abs(np.diff(tr, n=min(seasonality), prepend=[np.NaN])))
        mase = MASE(te, f, mean_abs_diff)
        if np.isnan(mase):
            mean_abs_diff = np.mean(np.abs(np.diff(tr, n=1, prepend=[np.NaN])))
            mase = MASE(te, f, mean_abs_diff)
        mase_per_series[k] = mase

    mase_per_series = mase_per_series[~np.isinf(mase_per_series) & ~np.isnan(mase_per_series)]
    return mase_per_series


def cyclic_transform(df, column):
    max_val = df[column].max()
    df['sin_' + column] = np.sin((2 * np.pi * df[column].values) / max_val)
    df['cos_' + column] = np.cos((2 * np.pi * df[column].values) / max_val)
    return df


def multi_target_transform(df, target_name, max_ahead, n_derivative=None, penalty_list=None):
    target_col_list = []
    target = df[target_name]

    if n_derivative is not None and n_derivative >= max_ahead:
        required_ahead = max_ahead + n_derivative
    else:
        required_ahead = max_ahead

    target_df = lagmat(target, maxlag=required_ahead - 1, trim="both", original='in', use_pandas=True)
    target_df.index = target_df.index - required_ahead + 1
    target_df = target_df[target_df.columns[::-1]]
    target_df.columns = ['stepid_' + str(a) for a in range(required_ahead)]
    target_col_list.append(list(target_df.columns[:max_ahead]))
    target_type_list = ['target']

    if n_derivative is not None:
        diff_target = target_df.copy()
        for i in range(n_derivative):
            if diff_target.shape[1] > 1:
                diff_target = diff_target.diff(axis=1)
                diff_target.drop(diff_target.columns[0], axis=1, inplace=True)
                diff_target.columns = ['der_' + str(i+1) + '_stepid_' + str(a)
                                        for a in range(diff_target.shape[1])]
                target_col_list.append(list(diff_target.columns))
                target_df = pd.concat([target_df, diff_target], axis=1)
                target_type_list.append('derivative')

    if penalty_list is not None:
        penalty_labels = [df[m] for m in penalty_list]
        penalty_labels = pd.concat(penalty_labels, axis=1)
        target_col_list.append(list(penalty_labels.columns))
        target_df = pd.concat([target_df, penalty_labels], axis=1)
        target_type_list.append('external')

    used_targets = list(itertools.chain(*target_col_list))
    return target_df[used_targets], target_col_list, target_type_list


def add_time(df, timestamp_col, period_col, period):
    if period == 'days':
        df['timestamp'] = df[timestamp_col] + pd.to_timedelta(df[period_col], unit='d')
    elif period == 'months':
        df['timestamp'] = df.apply(
            lambda row: row[timestamp_col] + pd.DateOffset(months=row[period_col]), axis=1)
    elif period == 'years':
        df['timestamp'] = df.apply(
            lambda row: row[timestamp_col] + pd.DateOffset(years=row[period_col]), axis=1)
    elif period == 'hours':
        df['timestamp'] = df[timestamp_col] + pd.to_timedelta(df[period_col], unit='h')
    elif period == 'minutes':
        df['timestamp'] = df[timestamp_col] + pd.to_timedelta(df[period_col], unit='m')
    elif period == 'seconds':
        df['timestamp'] = df[timestamp_col] + pd.to_timedelta(df[period_col], unit='s')
    elif period == 'quarters':
        df['timestamp'] = df.apply(
            lambda row: row[timestamp_col] + pd.DateOffset(months=row[period_col]), axis=1)
    elif period == 'weeks':
        df['timestamp'] = df[timestamp_col] + pd.to_timedelta(df[period_col], unit='W')
    else:
        raise ValueError(f"Unsupported period: {period}")
    return df


def organize_repo_data(data_file, sep='_', ignore_timestamp=False):
    loaded_data, frequency, horizon, contain_missing_values, contain_equal_length, \
        datetime_index = convert_tsf_to_dataframe(data_file)

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
        unit = UNIT_MAP[frequency]
        multiplier = CNT_MAP[frequency]
        if ignore_timestamp:
            freq = "1Y"
            seasonality = [1]
            unit = UNIT_MAP["minutely"]
            multiplier = 1
    else:
        freq = "1Y"
        seasonality = [1]
        unit = UNIT_MAP["minutely"]
        multiplier = 1

    unique_series_types = []
    if datetime_index is not None:
        list_of_series = [pd.DataFrame({
            'series': loaded_data.loc[x, 'series_name'],
            'start_timestamp': loaded_data.loc[x, datetime_index],
            'y': loaded_data.loc[x, 'series_value']
        }) for x in loaded_data.index]
    else:
        list_of_series = [pd.DataFrame({
            'series': loaded_data.loc[x, 'series_name'],
            'start_timestamp': datetime.strptime("1900-01-01 00-00-00", "%Y-%m-%d %H-%M-%S"),
            'y': loaded_data.loc[x, 'series_value']
        }) for x in loaded_data.index]
    list_of_series = pd.concat(list_of_series)

    if sep in list_of_series['series'].iloc[0]:
        unique_series = pd.DataFrame({'series': list_of_series['series'].unique()})
        unique_series[['type', 'id']] = unique_series['series'].str.split(sep, n=1, expand=True)
        unique_series_types = unique_series['type'].unique()

        if len(unique_series_types) > 1:
            raw_series = list_of_series[
                list_of_series['series'].isin(
                    unique_series.loc[unique_series['type'] == 'T', 'series'].values)]
            unique_series_types = unique_series_types[unique_series_types != 'T']
            for index_type in unique_series_types:
                temp_series = list_of_series[
                    list_of_series['series'].isin(
                        unique_series.loc[unique_series['type'] == index_type, 'series'].values)]
                raw_series[index_type] = temp_series['y']
            list_of_series = raw_series.copy()
            del raw_series, unique_series, temp_series
            gc.collect()

    list_of_series['cnt'] = multiplier * list_of_series.groupby(['series']).cumcount()
    list_of_series = add_time(list_of_series, 'start_timestamp', 'cnt', unit)
    unique_timestamp = pd.DataFrame({'timestamp': list_of_series['timestamp'].unique()})
    unique_timestamp['season_index'] = np.arange(len(unique_timestamp))
    list_of_series = list_of_series.merge(unique_timestamp, on='timestamp')
    list_of_series.drop(labels=['cnt', 'start_timestamp'], axis=1, inplace=True)
    list_of_series = list_of_series.sort_values(by=['series', 'timestamp'])
    list_of_series = list_of_series.reset_index()
    # OPT: convert series column to category — saves ~5x memory on the string keys
    # and speeds up all downstream groupby('series') operations.
    list_of_series['series'] = list_of_series['series'].astype('category')
    unique_series_types = list(unique_series_types)
    del unique_timestamp
    gc.collect()
    return list_of_series, freq, seasonality, horizon, unique_series_types


def convert_tsf_to_dataframe(full_file_path_and_name,
                              replace_missing_vals_with="NaN",
                              value_column_name="series_value"):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False
    datetime_index = None

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            line = line.strip()
            if line:
                if line.startswith("@"):
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:
                                raise Exception("Invalid meta-data specification.")
                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:
                                raise Exception("Invalid meta-data specification.")
                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))
                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data.")
                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []
                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")
                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")
                        if len(series) == 0:
                            raise Exception("A given series should contain at least one numeric value.")

                        numeric_series = []
                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                            raise Exception("All series values are missing.")

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                                datetime_index = col_names[i]
                            else:
                                raise Exception("Invalid attribute type.")
                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

            line_count += 1

    if line_count == 0:
        raise Exception("Empty file.")
    if len(col_names) == 0:
        raise Exception("Missing attribute section.")
    if not found_data_section:
        raise Exception("Missing series information under data section.")

    all_data[value_column_name] = all_series
    loaded_data = pd.DataFrame(all_data)
    return (loaded_data, frequency, forecast_horizon,
            contain_missing_values, contain_equal_length, datetime_index)


def compute_seasonal_scale(combined_data, target_name, drop_last,
                           max_seasonality):
    """Per-series seasonal-naive scale for MASE.

    For each series, drop the LAST ``drop_last`` raw observations (these are
    held out for forecast evaluation, so they must not enter the scale), then
    compute the mean absolute seasonal-lag-``max_seasonality`` difference over
    the remaining training portion. Series with zero or non-finite seasonal
    scale fall back to the lag-1 (non-seasonal naive) scale, matching
    https://github.com/rakshitha123/TSForecasting utils/error_calculator.R.

    Monash-style MASE requires ``drop_last == horizon`` (the test set is the
    last H raw observations). Using ``drop_last == test_horizon`` (the number
    of wide-format test rows) silently leaks ``horizon - test_horizon`` raw
    test rows into the scale denominator. Exercised by
    tests/test_phase2.py::test_seasonal_scale_drops_full_horizon.
    """
    original_series = combined_data[['index', 'series', target_name]]
    grouped = original_series.groupby('series').cumcount(ascending=False)
    training_only = original_series[grouped >= drop_last].copy()
    training_only['seas_diff'] = abs(
        training_only[target_name]
        - training_only.groupby('series')[target_name].shift(max_seasonality))
    seasonal_scale = training_only.groupby('series')['seas_diff'].mean()
    training_only['lag1_diff'] = abs(
        training_only[target_name]
        - training_only.groupby('series')[target_name].shift(1))
    lag1_scale = training_only.groupby('series')['lag1_diff'].mean()
    seasonal_scale = seasonal_scale.where(
        (seasonal_scale > 0) & np.isfinite(seasonal_scale), lag1_scale)
    return seasonal_scale


def shutdown_worker_pool():
    """Terminate the joblib loky reusable executor, if any.

    Why: joblib keeps its reusable worker pool alive for the process lifetime.
    After mRF.fit() finishes, the worker procs retain their high-water-mark
    RSS -- on the heaviest datasets that's several GB per worker. On macOS the
    accumulated footprint trips jetsam and the benchmark dies silently. Call
    this between datasets to recycle the pool back to zero before the next
    load.

    Safe to call before any pool has been created and safe to call repeatedly.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
    except ImportError:
        return
    try:
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
