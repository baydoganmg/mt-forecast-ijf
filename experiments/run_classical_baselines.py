"""Classical baselines for the IJF-D-25-01077 revision (R2.12).

Runs Naive, SeasonalNaive, AutoARIMA, AutoETS on all 27 datasets using
statsforecast (Nixtla). Matches your evaluation protocol exactly:
fixed-origin split with the LAST K observations of each series held
out as test, where K is the dataset's `horizon` from
`data/repo_data_info.xlsx` (overridden for M1 Yearly: H=6).

Outputs per-method, per-dataset CSVs in the same format as the
mDT/mGBT/mRF forecasts:
    results/forecasts_baselines/<method>/<dataset>_run_forecast.csv
    results/forecasts_baselines/<method>/<dataset>_run_actual.csv
columns: [0, 1, ..., K-1, index, series]

This script is fully self-contained: it does not import anything from
the `mttrees` package (which requires the Cython extensions and is
incompatible with the venv's numpy 2.x). It includes a minimal .tsf
parser inlined.

Run with the dedicated venv:
    .venv/bin/python experiments/run_classical_baselines.py [dataset_name]

If a dataset name is given as an argument, only that dataset is processed.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ----- Frequency / seasonality maps (mirrors mttrees.utils) -----
SEASONALITY_MAP = {
    "minutely":    [1440],
    "10_minutes":  [144],
    "half_hourly": [48],
    "hourly":      [24],
    "daily":       [7],
    "weekly":      [52],   # statsforecast uses int seasons; week ~ 52
    "monthly":     [12],
    "quarterly":   [4],
    "yearly":      [1],
}
SF_FREQ = {
    "minutely":    "min",
    "10_minutes":  "10min",
    "half_hourly": "30min",
    "hourly":      "h",
    "daily":       "D",
    "weekly":      "W",
    "monthly":     "MS",
    "quarterly":   "QS",
    "yearly":      "YS",
}


def parse_tsf(path: Path) -> tuple[list[dict], str | None, int | None]:
    """Minimal .tsf parser. Returns (records, frequency, horizon).

    Each record: {'series_name': str, 'series_value': list[float],
                  'start_timestamp': datetime | None}.
    """
    from datetime import datetime
    col_names: list[str] = []
    col_types: list[str] = []
    frequency: str | None = None
    horizon: int | None = None
    datetime_col: str | None = None
    records: list[dict] = []
    started_data = False
    found_data_tag = False

    with open(path, "r", encoding="cp1252") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                if line.startswith("@data"):
                    found_data_tag = True
                    continue
                parts = line.split(" ")
                if line.startswith("@attribute"):
                    col_names.append(parts[1]); col_types.append(parts[2])
                elif line.startswith("@frequency"):
                    frequency = parts[1]
                elif line.startswith("@horizon"):
                    horizon = int(parts[1])
                continue
            if not found_data_tag:
                raise RuntimeError(f"{path.name}: data line before @data")
            if not started_data:
                started_data = True
                for i, c in enumerate(col_types):
                    if c == "date":
                        datetime_col = col_names[i]
            full = line.split(":")
            series_str = full[-1].split(",")
            values = [float(v) for v in series_str if v != "?"]
            rec = {"series_value": values}
            for i, c in enumerate(col_types):
                if c == "string":
                    rec[col_names[i]] = full[i]
                elif c == "numeric":
                    rec[col_names[i]] = int(full[i])
                elif c == "date":
                    rec[col_names[i]] = datetime.strptime(full[i], "%Y-%m-%d %H-%M-%S")
            if "series_name" not in rec:
                raise RuntimeError(f"{path.name}: no series_name attribute")
            records.append(rec)
    return records, frequency, horizon


def tsf_to_long_df(records: list[dict], frequency: str | None,
                   datetime_col: str | None = "start_timestamp") -> pd.DataFrame:
    """Build long-format DataFrame with columns [unique_id, ds, y].

    `ds` is monotonic per series: if start_timestamp exists, use it;
    else use a synthetic per-series integer index converted to dates
    via a per-frequency offset starting 1900-01-01.
    """
    from datetime import datetime
    sf_freq = SF_FREQ.get(frequency, "D") if frequency else "D"
    rows = []
    fallback_start = pd.Timestamp("1900-01-01")
    for rec in records:
        series_name = rec["series_name"]
        values = rec["series_value"]
        start = rec.get(datetime_col, None)
        if start is None:
            start = fallback_start
        ds = pd.date_range(start=pd.Timestamp(start), periods=len(values),
                           freq=sf_freq)
        rows.append(pd.DataFrame({
            "unique_id": series_name,
            "ds": ds,
            "y": np.asarray(values, dtype=float),
        }))
    df = pd.concat(rows, ignore_index=True)
    return df


def split_last_K(df: pd.DataFrame, K: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-series: last K rows -> test, prior rows -> train. Mirrors
    Monash and the mDT pipeline."""
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["_rank_back"] = df.groupby("unique_id").cumcount(ascending=False)
    train = df[df["_rank_back"] >= K].drop(columns="_rank_back").copy()
    test  = df[df["_rank_back"] <  K].drop(columns="_rank_back").copy()
    return train, test


def run_for_dataset(ds_name: str, info: pd.Series, data_dir: Path,
                    out_root: Path, only_methods: list[str] | None = None) -> None:
    from statsforecast import StatsForecast
    from statsforecast.models import Naive, SeasonalNaive, AutoARIMA, AutoETS

    tsf_path = data_dir / info["file_name"]
    if not tsf_path.exists():
        print(f"  !! {ds_name}: {tsf_path.name} missing; skip"); return

    records, freq, _h_from_tsf = parse_tsf(tsf_path)
    horizon = int(info["horizon"]) if not pd.isna(info["horizon"]) else _h_from_tsf
    season_len = SEASONALITY_MAP.get(freq, [1])[0]
    n_series = len(records)
    print(f"  {ds_name}: freq={freq} season={season_len} K={horizon} series={n_series}", flush=True)

    long_df = tsf_to_long_df(records, freq)
    train, test = split_last_K(long_df, horizon)

    # Build the test_actual matrix (n_series x K) in series-sorted order
    series_order = sorted(test["unique_id"].unique())
    test_pivot = test.pivot_table(index="unique_id", columns=test.groupby("unique_id").cumcount(),
                                  values="y", aggfunc="first").reindex(series_order)
    actual_arr = test_pivot.to_numpy()
    if actual_arr.shape[1] != horizon:
        print(f"  !! {ds_name}: pivoted test shape {actual_arr.shape} != ({n_series},{horizon})")

    # statsforecast freq
    sf_freq = SF_FREQ.get(freq, "D")

    methods_all = {
        "naive":   Naive(),
        "snaive":  SeasonalNaive(season_length=season_len),
        "ets":     AutoETS(season_length=season_len),
        "arima":   AutoARIMA(season_length=season_len),
    }
    methods = {k: v for k, v in methods_all.items()
               if (only_methods is None or k in only_methods)}

    for method_name, model in methods.items():
        forecast_file = out_root / method_name / f"{ds_name}_run_forecast.csv"
        actual_file   = out_root / method_name / f"{ds_name}_run_actual.csv"
        if forecast_file.exists() and actual_file.exists():
            print(f"    [skip] {method_name}: already saved")
            continue

        try:
            t0 = time.time()
            sf = StatsForecast(models=[model], freq=sf_freq, n_jobs=4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = sf.forecast(df=train, h=horizon)
            fit_t = time.time() - t0
        except Exception as e:
            print(f"    [FAIL] {method_name}: {type(e).__name__}: {e}")
            continue

        # fc has columns: unique_id, ds, <ModelName>
        # ModelName is e.g. 'Naive', 'SeasonalNaive', 'AutoETS', 'AutoARIMA'
        model_col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        fc = fc.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        fc_pivot = fc.pivot_table(index="unique_id",
                                   columns=fc.groupby("unique_id").cumcount(),
                                   values=model_col, aggfunc="first").reindex(series_order)
        forecast_arr = fc_pivot.to_numpy()
        if forecast_arr.shape != (n_series, horizon):
            print(f"    !! {method_name}: forecast shape {forecast_arr.shape} != ({n_series},{horizon}); skipping save")
            continue

        # Save in the same format as tree forecasts (cols: 0..K-1, index, series)
        # (We don't have a meaningful 'index' integer here; use 0 for all rows.)
        fc_df = pd.DataFrame(forecast_arr, columns=[str(i) for i in range(horizon)])
        fc_df["index"]  = 0
        fc_df["series"] = series_order
        fc_df.to_csv(forecast_file, index=False)

        ac_df = pd.DataFrame(actual_arr, columns=[str(i) for i in range(horizon)])
        ac_df["index"]  = 0
        ac_df["series"] = series_order
        ac_df.to_csv(actual_file, index=False)
        print(f"    [done] {method_name}: {fit_t:.1f}s -> {forecast_file.name}")


def main() -> None:
    info_table = pd.read_excel(REPO_ROOT / "data" / "repo_data_info.xlsx",
                               "repo_data").set_index("Dataset")
    if "M1 Yearly" not in info_table.index:
        info_table.loc["M1 Yearly"] = {
            "prefix": "m1_yearly", "file_name": "m1_yearly_dataset.tsf",
            "horizon": 6, "predetermined_lag": 2, "integer_conversion": np.nan,
            "run": 1,
        }
    # Use the 1000-series subset, matching the tree retune pipeline
    if "Kaggle Daily" in info_table.index:
        info_table.loc["Kaggle Daily", "file_name"] = "kaggle_web_traffic_1000_dataset.tsf"
    if "Kaggle Daily with Cov" in info_table.index:
        info_table.loc["Kaggle Daily with Cov", "file_name"] = "kaggle_web_traffic_1000_dataset_with_date_covariates.tsf"

    runnable = info_table[info_table["run"] == 1].index.tolist()
    if "M1 Yearly" not in runnable:
        runnable.append("M1 Yearly")

    # Skip "with Cov" datasets: classical baselines only use the target series y,
    # so they would be identical to the non-cov versions. The .tsf with-cov files
    # actually encode covariates as pseudo-series, which is incorrect for univariate
    # baselines. The trees handle covariates via organize_repo_data() exogenous columns.
    runnable = [d for d in runnable if "with Cov" not in d]

    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        runnable = [d for d in runnable if d == only]
        if not runnable:
            print(f"No dataset matching '{only}'"); return
    print(f"Will process {len(runnable)} datasets: {runnable}")

    out_root = REPO_ROOT / "results" / "forecasts_baselines"
    for m in ["naive", "snaive", "ets", "arima"]:
        (out_root / m).mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    for i, ds in enumerate(runnable, 1):
        elapsed = (time.time() - overall_t0) / 60
        print(f"\n=== [{i}/{len(runnable)}] {ds}  (elapsed: {elapsed:.1f}m) ===", flush=True)
        run_for_dataset(ds, info_table.loc[ds], REPO_ROOT / "data", out_root)
    total_min = (time.time() - overall_t0) / 60
    print(f"\n=== Classical baselines done in {total_min:.1f} min ===")


if __name__ == "__main__":
    main()
