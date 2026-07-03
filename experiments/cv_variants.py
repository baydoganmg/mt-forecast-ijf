"""Cross-validate the four split-regularization variants for each family.

For each (dataset, variant in {base, deriv, seas, both}):
    1. From the dataset's CV log (run fresh if missing) under the
       benchmark result prefix, filter to the variant's slice:
            base:  lam_w == 0  and  bet_w == 0
            deriv: lam_w  > 0  and  bet_w == 0
            seas:  lam_w == 0  and  bet_w  > 0
            both:  lam_w  > 0  and  bet_w  > 0
       Pick best (depth, lambda_decay, objective_weights) by min mean
       test_error across folds.
    2. Refit mDT / mGBT / mRF at those params on full train, predict
       test, compute (mean, median) MASE.
    3. Append a row to results/cv_reg_variants.csv.

Resumable: rows already in the output CSV are skipped on re-run.

Final structure: 27 datasets x 4 variants = 108 rows. Pivot to a wide
12-column form (mDT/mGBT/mRF x 4 variants) for the MCB diagram.
"""
from __future__ import annotations
import argparse, ast, gc, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent  # mt-forecast/
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault("MT_BACKEND", "cpp")
os.environ.setdefault("MT_DTYPE", "fp32")

from mttrees.tree import CTree_MT, DataMt
from mttrees.ensemble import BDT, RF
from mttrees.utils import (organize_repo_data, row_wise_mase, msmape,
                           compute_seasonal_scale)
from experiments.run_benchmark import (build_features_and_targets,
                                       mean_scale_data, _safe_aggregate)

CFG_PATH = REPO / "configs" / "benchmark.yaml"
CFG = yaml.safe_load(open(CFG_PATH))

RESULT_PREFIX = CFG.get("result_prefix", "benchmark")
TUNING_DIR = REPO / CFG["results_path"] / "tuning"
INFO_TABLE = pd.read_excel(CFG["info_table"], "repo_data").set_index("Dataset")

OUT_DIR = Path(os.environ["CV_OUT"]) if os.environ.get("CV_OUT") else (REPO / "results" / "cv")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "cv_variants.csv"

VARIANTS = {
    "base":  lambda lam_w, bet_w: (lam_w == 0) and (bet_w == 0),
    "deriv": lambda lam_w, bet_w: (lam_w  > 0) and (bet_w == 0),
    "seas":  lambda lam_w, bet_w: (lam_w == 0) and (bet_w  > 0),
    "both":  lambda lam_w, bet_w: (lam_w  > 0) and (bet_w  > 0),
}

# 27 production datasets (excludes M5 Daily which was dropped post-fix)
DATASETS = [
    "Bitcoin", "COVID Deaths", "Chaotic logistic", "Electricity Weekly",
    "FREDMD", "Favorita Daily", "Favorita Daily with Cov", "Hospital",
    "Kaggle Daily", "Kaggle Daily with Cov", "M4 Hourly", "M4 Weekly", "M5 Daily",
    "M1 Monthly", "M1 Quarterly", "M1 Yearly", "M3 Monthly", "M3 Quarterly",
    "Mackey-glass", "NN5 Daily", "NN5 Weekly", "Rosmann Daily",
    "Rosmann Daily with Cov", "Tourism Monthly", "Tourism Quarterly",
    "Tourism Yearly", "Traffic Weekly", "Vehicle Trips",
]


def _load_existing():
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV)
    return pd.DataFrame(columns=[
        "dataset", "variant", "depth", "decay", "lam_w", "bet_w",
        "mDT_mase_med", "mGBT_mase_med", "mRF_mase_med",
        "mDT_mase_mean", "mGBT_mase_mean", "mRF_mase_mean",
        "mDT_fit_s", "mGBT_fit_s", "mRF_fit_s",
        "mGBT_best_iter", "wallclock_s",
    ])


def _append_row(row):
    df = _load_existing()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)


def _load_or_run_cv(data_set, train_data, seasonal_scale, cfg):
    """Return CV log DataFrame with columns: depth, lambda_decay, lam_w, bet_w, mean_test_err."""
    cv_log_path = TUNING_DIR / f"{data_set}_{RESULT_PREFIX}_cv_log.csv"

    if not cv_log_path.exists():
        print(f"  [CV] {data_set}: no cached CV log; running fresh CV...", flush=True)
        tree = CTree_MT(
            max_depth=cfg["max_depth"], verbose=0,
            min_samples_leaf=cfg["min_samples_leaf"],
            n_discrete_lev=cfg["n_discrete_lev"], num_threads=1,
            prebin=cfg.get("prebin", False),
            aggregation=cfg.get("aggregation", "score").lower(),
        )
        row_scale = seasonal_scale.loc[train_data.index.series].values.astype(np.float64)
        def cv_eval_fun(target, prediction, _scale=row_scale):
            out = row_wise_mase(target, prediction, _scale)
            return np.where(np.isfinite(out), out, np.nan)
        cv_eval_fun.__name__ = "mase"

        min_depth = cfg["min_depth"]; max_depth = cfg["max_depth"]
        depth_inc = cfg["depth_increment"]
        eval_depth_lev = np.arange(min_depth, max_depth + 1, depth_inc).tolist()
        # Seasonal beta_set is zero for yearly datasets (matches run_benchmark)
        info = INFO_TABLE.loc[data_set]
        # use frequency from train_data? simpler: derive from CV by always
        # passing full beta_set; for yearly the no-beta cells dominate by
        # construction (no seasonal feature exists)
        beta_set = cfg["beta_set"]

        t0 = time.time()
        best_params, cv_log, _, _ = tree.tune_penalties(
            train_data,
            decay_set=cfg["decay_set"], lambda_set=cfg["lambda_set"],
            beta_set=beta_set, eval_depths=eval_depth_lev,
            cv_type=cfg.get("cv_type", "grouped"),
            nfolds=cfg["nfolds"], seed=cfg.get("seed", 10),
            eval_fun=cv_eval_fun, n_jobs_cv=cfg.get("n_jobs_cv", 1),
        )
        print(f"  [CV] {data_set}: done in {time.time()-t0:.1f}s", flush=True)
        cv_log.to_csv(cv_log_path, index=False)
        # also write the "best_params" file so future runs see it cached
        bp_path = TUNING_DIR / f"{data_set}_{RESULT_PREFIX}.json"
        if not bp_path.exists():
            with open(bp_path, "w") as f:
                json.dump(best_params, f)

    log = pd.read_csv(cv_log_path)
    log["w"] = log["objective_weights"].apply(ast.literal_eval)
    log["lam_w"] = log["w"].apply(lambda x: float(x[1]))
    log["bet_w"] = log["w"].apply(lambda x: float(x[2]))
    agg = (log.groupby(["depth", "lambda_decay", "lam_w", "bet_w"])
              ["test_error"].mean().reset_index()
              .rename(columns={"test_error": "mean_test_err"}))
    return agg


def _pick_variant(agg, variant_fn):
    sub = agg[agg.apply(lambda r: variant_fn(r.lam_w, r.bet_w), axis=1)]
    if len(sub) == 0:
        return None
    r = sub.sort_values("mean_test_err").iloc[0]
    return {"depth": int(r.depth), "decay": float(r.lambda_decay),
            "lam_w": float(r.lam_w), "bet_w": float(r.bet_w),
            "cv_err": float(r.mean_test_err)}


def _build_data(data_set):
    info = INFO_TABLE.loc[data_set]
    data_file = REPO / CFG["data_path"] / info["file_name"]
    combined, frequency, seasonality, horizon, ext_features = organize_repo_data(
        str(data_file))
    if not pd.isna(info["horizon"]):
        horizon = int(info["horizon"])
    max_seasonality = max(seasonality)
    if not pd.isna(info.get("predetermined_lag", np.nan)):
        lag = int(info["predetermined_lag"])
    else:
        lag = int(np.ceil(max_seasonality * 1.25))
    max_seasonality = int(max_seasonality)
    max_ahead = int(horizon)

    seasonal_scale = compute_seasonal_scale(
        combined, "y", drop_last=int(horizon),
        max_seasonality=max_seasonality)

    lag_g, tar_g, _ = build_features_and_targets(
        combined, "y", lag, horizon, CFG.get("diff_features", False),
        ["index", "series"], ["season_index"],
        ext_features or [], frequency, max_seasonality)
    del combined; gc.collect()

    th = CFG.get("test_horizon", 1)
    grp = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grp >= th]; tef = lag_g[grp < th]
    grp = tar_g.groupby("series").cumcount(ascending=False)
    trt = tar_g[grp >= th]; tet = tar_g[grp < th]
    mean_scale = data_set != "Carparts"
    trf2, tef2, trt2, tet2, sm = mean_scale_data(
        trf, tef, trt, tet, lag, horizon, mean_scale)

    sp = ["sin_season", "cos_season"] if frequency != "1Y" else None
    integer_conversion = info["integer_conversion"]
    n_diff = CFG.get("n_diff", 1)
    train = DataMt(max_ahead=max_ahead, n_derivative=n_diff,
                   penalty_list=sp, series_means=sm,
                   int_convert=integer_conversion)
    train.transform(trf2, trt2)
    test = DataMt(max_ahead=max_ahead, n_derivative=n_diff,
                  penalty_list=sp, series_means=sm,
                  int_convert=integer_conversion)
    test.transform(tef2, tet2)
    return train, test, seasonal_scale, frequency


def _fit_three_methods(train, test, ss, pick, cfg):
    """Fit mDT/mGBT/mRF at picked (depth, decay, weights). Return per-method MASE + timings."""
    depth = pick["depth"]; decay = pick["decay"]
    weights = [1.0, pick["lam_w"], pick["bet_w"]]
    nthreads = cfg.get("nthreads", 1)
    n_level = cfg["n_discrete_lev"]; min_leaf = cfg["min_samples_leaf"]
    prebin = cfg.get("prebin", False); agg = cfg.get("aggregation", "score").lower()
    seed = cfg.get("seed", 10)

    # ---- mDT ----
    t0 = time.time()
    tree = CTree_MT(max_depth=depth, verbose=0, min_samples_leaf=min_leaf,
                    n_discrete_lev=n_level, num_threads=nthreads,
                    prebin=prebin, aggregation=agg)
    tree.arrangeObjective(train, decay, weights)
    tree.fit(train.x, train.y, random_state=seed)
    dt_pred = tree.predict(test.x)
    dt_pred, _ = test.process_predictions(dt_pred)
    dt_time = time.time() - t0
    dt_mean, dt_med = _safe_aggregate(row_wise_mase(
        test.y_org, dt_pred, ss.loc[test.index.series]))
    del tree; gc.collect()

    # ---- mGBT ----
    bdt_defaults = {
        "random_state": seed,
        "n_estimators_": cfg["bdt"]["n_estimators"],
        "lr": cfg["bdt"]["lr"],
        "bagging_fraction": cfg["bdt"]["bagging_fraction"],
        "feature_fraction": cfg["bdt"]["feature_fraction"],
        "max_depth": cfg["bdt"]["max_depth"],
        "min_samples_leaf": min_leaf,
        "n_discrete_lev": n_level,
        "num_threads": cfg["bdt"].get("num_threads", nthreads),
        "prebin": prebin,
        "aggregation": agg,
    }
    # add ES knobs if present in yaml
    for k in ("early_stopping_rounds", "min_delta", "eval_k", "eval_metric"):
        if k in cfg.get("bdt", {}):
            bdt_defaults[k] = cfg["bdt"][k]
    t0 = time.time()
    bdt = BDT(train, lambda_decay=decay, objective_weights=weights,
              eval_seasonal_scale=ss, **bdt_defaults)
    bdt.fit()
    gbt_pred = bdt.predict(test.x)
    gbt_pred, _ = test.process_predictions(gbt_pred)
    gbt_time = time.time() - t0
    gbt_best_iter = getattr(bdt, "best_iteration_", None)
    gbt_mean, gbt_med = _safe_aggregate(row_wise_mase(
        test.y_org, gbt_pred, ss.loc[test.index.series]))
    del bdt; gc.collect()

    # ---- mRF ----
    rf_defaults = {
        "random_state": seed,
        "n_estimators_": cfg["rf"]["n_estimators"],
        "bagging_fraction": cfg["rf"]["bagging_fraction"],
        "feature_fraction": cfg["rf"]["feature_fraction"],
        "max_depth": cfg["rf"]["max_depth"],
        "min_samples_leaf": min_leaf,
        "n_discrete_lev": n_level,
        "num_threads": cfg["rf"].get("num_threads", nthreads),
        "prebin": prebin,
        "n_jobs": cfg["rf"].get("n_jobs", 1),
        "backend": cfg["rf"].get("backend", "thread"),
        "aggregation": agg,
    }
    t0 = time.time()
    rf = RF(train, lambda_decay=decay, objective_weights=weights,
            **rf_defaults)
    rf.fit()
    rf_pred = rf.predict(test.x)
    rf_pred, _ = test.process_predictions(rf_pred)
    rf_time = time.time() - t0
    rf_mean, rf_med = _safe_aggregate(row_wise_mase(
        test.y_org, rf_pred, ss.loc[test.index.series]))
    del rf; gc.collect()

    return dict(
        mDT_mase_med=dt_med, mDT_mase_mean=dt_mean, mDT_fit_s=dt_time,
        mGBT_mase_med=gbt_med, mGBT_mase_mean=gbt_mean, mGBT_fit_s=gbt_time,
        mGBT_best_iter=gbt_best_iter if gbt_best_iter is not None else -1,
        mRF_mase_med=rf_med, mRF_mase_mean=rf_mean, mRF_fit_s=rf_time,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of datasets to run (default: all 27)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of variants to run (default: all 4)")
    args = ap.parse_args()

    todo_datasets = args.datasets or DATASETS
    todo_variants = args.variants or list(VARIANTS.keys())

    print(f"[cv] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[cv] {len(todo_datasets)} datasets x {len(todo_variants)} variants",
          flush=True)
    print(f"[cv] output: {OUT_CSV}", flush=True)

    overall_t0 = time.time()
    for ds in todo_datasets:
        if ds not in INFO_TABLE.index:
            print(f"  SKIP {ds}: not in info_table", flush=True)
            continue

        existing = _load_existing()
        existing_pairs = set(zip(existing["dataset"], existing["variant"]))
        pending = [v for v in todo_variants if (ds, v) not in existing_pairs]
        if not pending:
            print(f"  SKIP {ds}: all variants already done", flush=True)
            continue

        print(f"\n=== {ds} (variants to run: {pending}) ===", flush=True)
        ds_t0 = time.time()
        try:
            train, test, ss, freq = _build_data(ds)
        except Exception as e:
            print(f"  FAILED to build data: {e}", flush=True)
            continue
        print(f"  built data in {time.time()-ds_t0:.1f}s (train rows={train.x.shape[0]}, "
              f"freq={freq})", flush=True)

        try:
            agg = _load_or_run_cv(ds, train, ss, CFG)
        except Exception as e:
            print(f"  FAILED CV: {e}", flush=True)
            continue

        for vname in pending:
            print(f"\n  --- variant {vname} ---", flush=True)
            pick = _pick_variant(agg, VARIANTS[vname])
            if pick is None:
                print(f"    no cells matching variant {vname}; skip", flush=True)
                continue
            # Yearly datasets have bet_w = 0 always (no seasonal feature). The
            # 'seas' and 'both' variants will be the same as 'base' / 'deriv' for
            # them (no beta to vary). Still report what CV picked.
            print(f"    pick: depth={pick['depth']} decay={pick['decay']} "
                  f"weights=[1, {pick['lam_w']}, {pick['bet_w']}]  "
                  f"CV_err={pick['cv_err']:.4f}", flush=True)
            vt0 = time.time()
            try:
                metrics = _fit_three_methods(train, test, ss, pick, CFG)
            except Exception as e:
                print(f"    FIT FAILED: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            wallclock = time.time() - vt0
            row = {
                "dataset": ds, "variant": vname,
                "depth": pick["depth"], "decay": pick["decay"],
                "lam_w": pick["lam_w"], "bet_w": pick["bet_w"],
                "wallclock_s": round(wallclock, 1),
                **metrics,
            }
            _append_row(row)
            print(f"    [{wallclock:6.1f}s]  "
                  f"mDT med={metrics['mDT_mase_med']:.4f}  "
                  f"mGBT med={metrics['mGBT_mase_med']:.4f} (iter={metrics['mGBT_best_iter']})  "
                  f"mRF med={metrics['mRF_mase_med']:.4f}",
                  flush=True)

        del train, test, ss; gc.collect()
        print(f"  *** {ds} total: {(time.time()-ds_t0)/60:.1f} min ***", flush=True)

    print(f"\n[cv] FINISHED in {(time.time()-overall_t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
