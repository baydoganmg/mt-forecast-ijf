"""Controlled synthetic-panel ablation for the Section "A Controlled Synthetic Study".

Options:
  * --scaling {none|mean|std}: defaults to `none` (the principled choice for
    zero-centered Types A, B, D); also supports `mean` to reproduce the mean-scaled behaviour, and `std` as a third option for stationary scaling.
  * --seeds: list of base random seeds for multi-seed sensitivity. The
    script regenerates the full panel under each seed and aggregates
    per-type means with bootstrap 95% CIs across seeds.
  * --include-classical: optionally fits AutoARIMA, AutoETS, naive,
    seasonal-naive on every series via statsforecast, so the Type-C
    "classical wins" hypothesis can be tested directly.
  * Outputs one CSV per seed plus an aggregated CSV that gives
    per-(type, method) mean + 5-seed bootstrap CI.

Hypotheses (registered in this docstring before any rerun) — VERBATIM from
the original ablation script header:
  Type A (smooth seasonal + AR + low noise):  mDT_deriv beats base
  Type B (heterogeneous weekly modes):        mDT_seas beats base
  Type C (strong trend):                      tree methods all hurt; classical wins
  Type D (high-noise pure AR):                silent test (variants ~ base)

Usage:
    python synthetic_panel.py --seeds 2026 2027 2028 2029 2030 \
                                  --scaling none --include-classical

Output:
    mt_trees/results/synthetic_panel/
        per_seed_{seed}.csv      (one row per (method, type, seed))
        aggregate_per_type.csv   (mean ± 95% bootstrap CI across seeds)
        summary.txt              (human-readable narrative)
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("MT_DTYPE", "fp32")

from mttrees.tree import DataMt, CTree_MT  # noqa: E402

from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

# Suppress statsforecast / pandas convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Panel design constants (held fixed across seeds) ----------------------
N_OBS_PER_SERIES = 280
N_SERIES_BY_TYPE = {"A": 80, "B": 60, "C": 40, "D": 20}
HORIZON = 14
LAG = 14
PERIOD = 7
SIGMA_LOW = 0.3
SIGMA_HIGH = 1.0
MAX_DEPTH = 10
MIN_LEAF = 5
N_ESTIMATORS_RF = 100
TEST_HORIZON_HELDOUT = HORIZON

OUT_ROOT = REPO / "results" / "synthetic_panel"


# ---------------------- Panel generators (seed-driven) ---------------------
def gen_type_A(n_series: int, seed_base: int):
    series = {}
    for i in range(n_series):
        rng = np.random.default_rng(seed_base + 1000 + i)
        y = np.zeros(N_OBS_PER_SERIES)
        amp = 1.0 + 0.3 * rng.normal()
        phase = 2 * np.pi * rng.random()
        eps = rng.normal(0, SIGMA_LOW, N_OBS_PER_SERIES)
        y[0] = rng.normal(0, SIGMA_LOW)
        for t in range(1, N_OBS_PER_SERIES):
            y[t] = 0.3 * y[t-1] + amp * np.sin(2 * np.pi * t / PERIOD + phase) + eps[t]
        series[f"A_{i:03d}"] = y
    return series


def gen_type_B(n_series: int, seed_base: int):
    series = {}
    for i in range(n_series):
        rng = np.random.default_rng(seed_base + 2000 + i)
        y = np.zeros(N_OBS_PER_SERIES)
        weekly_mode = rng.normal(0, 1.0, PERIOD)
        weekly_mode -= weekly_mode.mean()
        weekly_mode *= 1.2 / (np.std(weekly_mode) + 1e-9)
        eps = rng.normal(0, SIGMA_LOW, N_OBS_PER_SERIES)
        y[0] = rng.normal(0, SIGMA_LOW)
        for t in range(1, N_OBS_PER_SERIES):
            y[t] = 0.5 * y[t-1] + weekly_mode[t % PERIOD] + eps[t]
        series[f"B_{i:03d}"] = y
    return series


def gen_type_C(n_series: int, seed_base: int):
    series = {}
    for i in range(n_series):
        rng = np.random.default_rng(seed_base + 3000 + i)
        y = np.zeros(N_OBS_PER_SERIES)
        slope = 0.02 + 0.01 * rng.normal()
        amp = 0.3 + 0.1 * rng.normal()
        eps = rng.normal(0, SIGMA_LOW, N_OBS_PER_SERIES)
        y[0] = rng.normal(0, SIGMA_LOW)
        for t in range(1, N_OBS_PER_SERIES):
            y[t] = 0.7 * y[t-1] + slope * t + amp * np.sin(2 * np.pi * t / PERIOD) + eps[t]
        series[f"C_{i:03d}"] = y
    return series


def gen_type_D(n_series: int, seed_base: int):
    series = {}
    for i in range(n_series):
        rng = np.random.default_rng(seed_base + 4000 + i)
        y = np.zeros(N_OBS_PER_SERIES)
        eps = rng.normal(0, SIGMA_HIGH, N_OBS_PER_SERIES)
        y[0] = rng.normal(0, SIGMA_HIGH)
        for t in range(1, N_OBS_PER_SERIES):
            y[t] = 0.6 * y[t-1] + eps[t]
        series[f"D_{i:03d}"] = y
    return series


def build_panel(seed_base: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    all_series = {}
    all_series.update(gen_type_A(N_SERIES_BY_TYPE["A"], seed_base))
    all_series.update(gen_type_B(N_SERIES_BY_TYPE["B"], seed_base))
    all_series.update(gen_type_C(N_SERIES_BY_TYPE["C"], seed_base))
    all_series.update(gen_type_D(N_SERIES_BY_TYPE["D"], seed_base))
    rows = []
    for sid, y in all_series.items():
        for t in range(N_OBS_PER_SERIES):
            rows.append({"index": t, "series": sid, "y": float(y[t]),
                         "season_index": t % PERIOD})
    combined = pd.DataFrame(rows)
    combined["series"] = combined["series"].astype("category")
    return all_series, combined


# ---------------------- Feature construction -------------------------------
def build_lag_target_panel(combined_df: pd.DataFrame, lag: int, horizon: int):
    feat_rows, tgt_rows = [], []
    for sid, grp in combined_df.groupby("series", observed=True):
        y = grp["y"].to_numpy()
        idx = grp["index"].to_numpy()
        seas = grp["season_index"].to_numpy()
        n = len(y)
        for t in range(lag, n - horizon + 1):
            feat = {"index": int(idx[t]), "series": sid}
            for L in range(1, lag + 1):
                feat[f"lag_{L}"] = float(y[t - L])
            feat["sin_season"] = float(np.sin(2 * np.pi * seas[t] / PERIOD))
            feat["cos_season"] = float(np.cos(2 * np.pi * seas[t] / PERIOD))
            feat_rows.append(feat)
            tgt = {"index": int(idx[t]), "series": sid}
            for h in range(horizon):
                tgt[f"ahead_{h}"] = float(y[t + h])
            tgt_rows.append(tgt)
    return pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows)


# ---------------------- Scaling strategies ---------------------------------
def apply_scaling(trf, tef, trt, tet, lag, horizon, scaling: str):
    """Returns (trf2, tef2, trt2, tet2, sm_series).

    `scaling` in {none, mean, std}:
      - none : no scaling (suitable for zero-centered series).
      - mean : v1 behaviour, divide by per-series lag_1 training mean.
      - std  : divide by per-series std of lag_1 training values.

    `sm_series` is a per-series scale dataframe with columns
    (series, lag_1) so it can be handed to DataMt.
    """
    if scaling == "none":
        # sm_series with lag_1=1 so downstream DataMt and skDT inverse-scaling
        # apply unit multipliers.
        sids = sorted(set(trf["series"].cat.categories) if hasattr(trf["series"], "cat")
                       else set(trf["series"]))
        sm_series = pd.DataFrame({"series": sids, "lag_1": 1.0})
        return trf.reset_index(drop=True), tef.reset_index(drop=True), \
               trt.reset_index(drop=True), tet.reset_index(drop=True), sm_series

    if scaling == "mean":
        sm = trf.groupby("series", observed=True)["lag_1"].mean().reset_index()
    elif scaling == "std":
        sm = trf.groupby("series", observed=True)["lag_1"].std().reset_index()
        sm.loc[sm["lag_1"] == 0, "lag_1"] = sm["lag_1"].replace(0, np.nan).mean()
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    sm.loc[sm["lag_1"] == 0, "lag_1"] = sm["lag_1"].mean()
    sm.rename(columns={"lag_1": "scale"}, inplace=True)
    trf2 = trf.merge(sm, on="series").reset_index(drop=True)
    tef2 = tef.merge(sm, on="series").reset_index(drop=True)
    trt2 = trt.merge(sm, on="series").reset_index(drop=True)
    tet2 = tet.merge(sm, on="series").reset_index(drop=True)
    for L in range(1, lag + 1):
        trf2[f"lag_{L}"] = trf2[f"lag_{L}"] / trf2["scale"]
        tef2[f"lag_{L}"] = tef2[f"lag_{L}"] / tef2["scale"]
    for h in range(horizon):
        trt2[f"ahead_{h}"] = trt2[f"ahead_{h}"] / trt2["scale"]
        tet2[f"ahead_{h}"] = tet2[f"ahead_{h}"] / tet2["scale"]
    sm_series = sm.rename(columns={"scale": "lag_1"})
    trf2 = trf2.drop(columns=["scale"])
    tef2 = tef2.drop(columns=["scale"])
    trt2 = trt2.drop(columns=["scale"])
    tet2 = tet2.drop(columns=["scale"])
    return trf2, tef2, trt2, tet2, sm_series


# ---------------------- Snaive scale per series ----------------------------
def per_series_snaive_scale(all_series: dict[str, np.ndarray]) -> pd.Series:
    scales = {}
    for sid, y in all_series.items():
        train_y = y[:-TEST_HORIZON_HELDOUT]
        diffs = np.abs(train_y[PERIOD:] - train_y[:-PERIOD])
        scales[sid] = float(diffs.mean()) if len(diffs) > 0 else np.nan
    return pd.Series(scales)


# ---------------------- Evaluate predictions -------------------------------
def evaluate_pred(pred_orig: np.ndarray, sids: np.ndarray,
                  ac_orig: np.ndarray, scales_s: pd.Series) -> np.ndarray:
    """Returns per-row MASE (n_series, H)."""
    scl = scales_s.loc[sids].to_numpy()
    valid = (scl > 0) & np.isfinite(scl)
    out = np.full(pred_orig.shape, np.nan)
    out[valid] = np.abs(ac_orig[valid] - pred_orig[valid]) / scl[valid, None]
    return out


# ---------------------- mDT + sklearn baselines ----------------------------
def eval_tree_methods(train_data, test_data, sm_series, scales_s, scaling: str):
    results = {}
    for variant, (lam, bet) in [("base",  (0.0, 0.0)),
                                 ("deriv", (0.5, 0.0)),
                                 ("seas",  (0.0, 0.5)),
                                 ("both",  (0.5, 0.5))]:
        tree = CTree_MT(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, mtry=1.0,
                        samp_frac=1.0, prebin=True, n_discrete_lev=32)
        tree.arrangeObjective(train_data, lambda_decay=1.0,
                              objective_weights=[1.0, lam, bet])
        tree.fit(train_data.x, train_data.y)
        pred_raw = tree.predict(test_data.x)
        pred, _ = test_data.process_predictions(pred_raw)
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] > HORIZON:
            pred = pred[:, :HORIZON]
        sids = test_data.index["series"].to_numpy() if "series" in test_data.index.columns else test_data.index.index
        results[f"mDT_{variant}"] = evaluate_pred(pred, sids, test_data.y_org, scales_s)
        del tree
        gc.collect()

    # sklearn DT / RF on the same scaled feature matrix
    X_train = np.asarray(train_data.x if not hasattr(train_data.x, "to_numpy") else train_data.x.to_numpy())
    Y_train = np.asarray(train_data.y if not hasattr(train_data.y, "to_numpy") else train_data.y.to_numpy())
    X_test = np.asarray(test_data.x if not hasattr(test_data.x, "to_numpy") else test_data.x.to_numpy())
    Y_train_lvl = Y_train[:, :HORIZON]

    sids_test = test_data.index["series"].to_numpy() if "series" in test_data.index.columns else test_data.index.index
    if scaling == "none":
        scl_means = np.ones(len(sids_test))
    else:
        scl_means = np.array([sm_series.set_index("series").loc[s, "lag_1"] for s in sids_test])

    skdt = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=0)
    skdt.fit(X_train, Y_train_lvl)
    pred_sk = skdt.predict(X_test) * scl_means[:, None]
    results["skDT"] = evaluate_pred(pred_sk, sids_test, test_data.y_org, scales_s)

    skrf = RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, max_depth=MAX_DEPTH,
                                  min_samples_leaf=MIN_LEAF, max_features=0.5,
                                  random_state=0, n_jobs=4)
    skrf.fit(X_train, Y_train_lvl)
    pred_skrf = skrf.predict(X_test) * scl_means[:, None]
    results["skRF"] = evaluate_pred(pred_skrf, sids_test, test_data.y_org, scales_s)
    return results, sids_test


# ---------------------- Classical baselines via statsforecast --------------
def eval_classical(all_series: dict[str, np.ndarray], scales_s: pd.Series,
                   sids_eval: Sequence[str]) -> dict[str, np.ndarray]:
    """Returns dict mapping classical method name → per-row MASE matrix
    aligned with sids_eval."""
    from statsforecast import StatsForecast  # local import; heavy
    from statsforecast.models import Naive, SeasonalNaive, AutoARIMA, AutoETS

    long_rows = []
    for sid, y in all_series.items():
        train_y = y[:-TEST_HORIZON_HELDOUT]
        for t, v in enumerate(train_y):
            long_rows.append({"unique_id": sid, "ds": int(t), "y": float(v)})
    train_df = pd.DataFrame(long_rows)

    models = {
        "naive":  Naive(),
        "snaive": SeasonalNaive(season_length=PERIOD),
        "ets":    AutoETS(season_length=PERIOD),
        "arima":  AutoARIMA(season_length=PERIOD),
    }
    actuals = {sid: y[-TEST_HORIZON_HELDOUT:] for sid, y in all_series.items()}
    results = {}
    for name, model in models.items():
        sf = StatsForecast(models=[model], freq=1, n_jobs=4)
        sf.fit(df=train_df)
        fc = sf.forecast(h=TEST_HORIZON_HELDOUT)
        # fc has columns: unique_id, ds, ModelName
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        per_sid = {sid: grp[col].to_numpy() for sid, grp in fc.groupby("unique_id")}
        rows_pred, rows_actu = [], []
        for sid in sids_eval:
            rows_pred.append(per_sid.get(sid, np.full(TEST_HORIZON_HELDOUT, np.nan)))
            rows_actu.append(actuals[sid])
        pred_arr = np.vstack(rows_pred)
        actu_arr = np.vstack(rows_actu)
        results[name] = evaluate_pred(pred_arr, np.asarray(sids_eval),
                                       actu_arr, scales_s)
    return results


# ---------------------- Per-type aggregation -------------------------------
def collect_per_type(results: dict[str, np.ndarray],
                     sids: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    types = np.array([s[0] for s in sids])
    for method, mase in results.items():
        for t in ["A", "B", "C", "D"]:
            mask = (types == t)
            if mask.sum() == 0:
                continue
            m = mase[mask]
            rows.append({"seed": seed, "method": method, "type": t,
                         "n_series": int(mask.sum()),
                         "mean_mase":   float(np.nanmean(m)),
                         "median_mase": float(np.nanmedian(m))})
        rows.append({"seed": seed, "method": method, "type": "ALL",
                     "n_series": int(len(mase)),
                     "mean_mase":   float(np.nanmean(mase)),
                     "median_mase": float(np.nanmedian(mase))})
    return pd.DataFrame(rows)


# ---------------------- Across-seed aggregation ----------------------------
def aggregate_across_seeds(per_seed_df: pd.DataFrame,
                            n_boot: int = 2000, alpha: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for (method, t), sub in per_seed_df.groupby(["method", "type"], observed=True):
        vals = sub["mean_mase"].to_numpy()
        if len(vals) == 0 or np.all(np.isnan(vals)):
            continue
        mean_v = float(np.nanmean(vals))
        # Bootstrap CI across seeds — small-n; we still report for transparency
        if len(vals) >= 2:
            boots = [np.nanmean(vals[rng.integers(0, len(vals), len(vals))])
                     for _ in range(n_boot)]
            lo = float(np.nanpercentile(boots, 100 * alpha / 2))
            hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
        else:
            lo, hi = mean_v, mean_v
        rows.append({"method": method, "type": t,
                     "mean": mean_v, "ci_lo": lo, "ci_hi": hi,
                     "n_seeds": len(vals)})
    return pd.DataFrame(rows)


# ---------------------- Single-seed pipeline -------------------------------
def run_one_seed(seed: int, scaling: str, include_classical: bool,
                 out_dir: Path) -> pd.DataFrame:
    print(f"\n[seed={seed}] generating panel ...", flush=True)
    all_series, combined = build_panel(seed)
    print(f"[seed={seed}] panel: {combined['series'].nunique()} series, "
          f"{len(combined)} rows", flush=True)

    lag_g, tgt_g = build_lag_target_panel(combined, LAG, HORIZON)
    grp_f = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grp_f >= 1].reset_index(drop=True)
    tef = lag_g[grp_f < 1].reset_index(drop=True)
    grp_t = tgt_g.groupby("series").cumcount(ascending=False)
    trt = tgt_g[grp_t >= 1].reset_index(drop=True)
    tet = tgt_g[grp_t < 1].reset_index(drop=True)
    trf2, tef2, trt2, tet2, sm_series = apply_scaling(
        trf, tef, trt, tet, LAG, HORIZON, scaling)

    sp = ["sin_season", "cos_season"]
    train_data = DataMt(max_ahead=HORIZON, n_derivative=1, penalty_list=sp,
                        series_means=sm_series, int_convert=False)
    train_data.transform(trf2, trt2)
    test_data = DataMt(max_ahead=HORIZON, n_derivative=1, penalty_list=sp,
                       series_means=sm_series, int_convert=False)
    test_data.transform(tef2, tet2)

    scales_s = per_series_snaive_scale(all_series)
    print(f"[seed={seed}] fitting mDT + sklearn ...", flush=True)
    results, sids_eval = eval_tree_methods(train_data, test_data, sm_series,
                                            scales_s, scaling)
    if include_classical:
        print(f"[seed={seed}] fitting classical baselines (this is the slow part) ...",
              flush=True)
        results.update(eval_classical(all_series, scales_s, sids_eval))

    per_type = collect_per_type(results, sids_eval, seed)
    per_type.to_csv(out_dir / f"per_seed_{seed}.csv", index=False)
    print(f"[seed={seed}] wrote {out_dir / f'per_seed_{seed}.csv'}", flush=True)
    return per_type


# ---------------------- CLI ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--scaling", choices=("none", "mean", "std"), default="none")
    ap.add_argument("--include-classical", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_ROOT / f"scaling_{args.scaling}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {out_dir}", flush=True)
    print(f"seeds: {args.seeds}", flush=True)
    print(f"include classical: {args.include_classical}", flush=True)

    all_per_seed = []
    for s in args.seeds:
        per_type = run_one_seed(s, args.scaling, args.include_classical, out_dir)
        all_per_seed.append(per_type)

    full = pd.concat(all_per_seed, ignore_index=True)
    full.to_csv(out_dir / "per_seed_long.csv", index=False)

    agg = aggregate_across_seeds(full)
    agg.to_csv(out_dir / "aggregate_per_type.csv", index=False)
    print(f"\nwrote {out_dir / 'aggregate_per_type.csv'}", flush=True)
    print(agg.round(3).to_string(index=False))

    # Rel-to-base per type
    pivot = agg.pivot_table(index="method", columns="type", values="mean",
                             aggfunc="first")
    if "mDT_base" in pivot.index:
        rel = (pivot.subtract(pivot.loc["mDT_base"]) / pivot.loc["mDT_base"] * 100).round(2)
        rel.to_csv(out_dir / "rel_to_base_mean.csv")
        print("\n=== rel (%) vs mDT_base per type, averaged across seeds ===")
        print(rel.to_string())

    lines = [f"Synthetic panel v2 — scaling={args.scaling}, "
             f"seeds={args.seeds}, classical={args.include_classical}",
             "Mixture: A=80, B=60, C=40, D=20",
             "Hypotheses (registered in script docstring):",
             "  A: mDT_deriv beats base",
             "  B: mDT_seas beats base",
             "  C: tree methods all hurt; classical wins",
             "  D: variants approximately tie base",
             "",
             agg.round(3).to_string(index=False)]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
