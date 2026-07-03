"""Iterative-vs-direct comparison across all three families.

For each (dataset, family):
  - direct:    fit at CV-best base-variant params (max_ahead=H), predict multi-output
  - iterative: fit at same params (max_ahead=1), roll forward H steps

Both sides use identical data splits and hyperparameters.
CV-best params come from results/cv/cv_variants.csv (base variant).

Datasets: 14 fast well-known benchmarks (all < ~2 min/variant).
Families: mDT, mGBT, mRF.

Output:
  results/iterative_vs_direct/per_horizon_mase.csv
  results/iterative_vs_direct/summary.txt
"""
from __future__ import annotations
import sys, os, gc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("MT_BACKEND", "cpp")
os.environ.setdefault("MT_DTYPE", "fp32")

import numpy as np
import pandas as pd
import yaml

from mttrees.tree import DataMt, CTree_MT, MT_DTYPE
from mttrees.ensemble import BDT, RF
from mttrees.utils import organize_repo_data, row_wise_mase, compute_seasonal_scale
from experiments.run_benchmark import build_features_and_targets, mean_scale_data

CFG = yaml.safe_load(open(REPO / "configs" / "benchmark.yaml"))
INFO_TABLE = pd.read_excel(CFG["info_table"], "repo_data").set_index("Dataset")

# Add yearly datasets missing from info table
_YEARLY_EXTRA = {
    "M1 Yearly": {"prefix": "m1_yearly", "file_name": "m1_yearly_dataset.tsf",
                  "horizon": 6, "predetermined_lag": 2, "integer_conversion": np.nan, "run": 1},
}
for ds, row in _YEARLY_EXTRA.items():
    if ds not in INFO_TABLE.index:
        INFO_TABLE.loc[ds] = row

OUT_DIR = REPO / "results" / "iterative_vs_direct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "M1 Yearly", "M1 Quarterly", "M1 Monthly",
    "M3 Quarterly", "M3 Monthly",
    "NN5 Weekly", "NN5 Daily",
    "Tourism Yearly", "Tourism Quarterly", "Tourism Monthly",
    "Hospital",
    "Electricity Weekly",
    "Traffic Weekly",
    "Mackey-glass",
]
FAMILIES = ["dt", "gbt", "rf"]

# ---------------------------------------------------------------------------
# Load CV-best base-variant params from the cross-validation cache
# ---------------------------------------------------------------------------
OPTB = pd.read_csv(REPO / "results" / "cv" / "cv_variants.csv")
OPTB_BASE = OPTB[OPTB["variant"] == "base"].set_index("dataset")


def get_base_params(ds_name):
    """Return (depth, lambda_decay) from CV-best base variant."""
    if ds_name in OPTB_BASE.index:
        row = OPTB_BASE.loc[ds_name]
        return int(row["depth"]), float(row["decay"])
    # fallback to config defaults if the dataset is not in the CV cache
    return CFG["bdt"]["max_depth"], 0.5


# ---------------------------------------------------------------------------
# Fit helpers — mirror the benchmark _fit_three_methods patterns
# ---------------------------------------------------------------------------
def _cfg_params(cfg):
    return dict(
        nthreads=cfg.get("nthreads", 1),
        n_level=cfg["n_discrete_lev"],
        min_leaf=cfg["min_samples_leaf"],
        prebin=cfg.get("prebin", False),
        agg=cfg.get("aggregation", "score").lower(),
        seed=cfg.get("seed", 10),
    )


def fit_model(family, train_data, depth, decay, cfg, ss=None):
    """Fit mDT/mGBT/mRF. Returns (model, predict_fn) where predict_fn(test_x, test_obj) -> np.ndarray."""
    p = _cfg_params(cfg)
    weights = [1.0, 0.0, 0.0]

    if family == "dt":
        tree = CTree_MT(max_depth=depth, verbose=0,
                        min_samples_leaf=p["min_leaf"],
                        n_discrete_lev=p["n_level"],
                        num_threads=p["nthreads"],
                        prebin=p["prebin"],
                        aggregation=p["agg"])
        tree.arrangeObjective(train_data, decay, weights)
        tree.fit(train_data.x, train_data.y, random_state=p["seed"])
        return tree

    elif family == "gbt":
        bdt_kw = dict(
            random_state=p["seed"],
            n_estimators_=cfg["bdt"]["n_estimators"],
            lr=cfg["bdt"]["lr"],
            bagging_fraction=cfg["bdt"]["bagging_fraction"],
            feature_fraction=cfg["bdt"]["feature_fraction"],
            max_depth=depth,
            min_samples_leaf=p["min_leaf"],
            n_discrete_lev=p["n_level"],
            num_threads=cfg["bdt"].get("num_threads", p["nthreads"]),
            prebin=p["prebin"],
            aggregation=p["agg"],
        )
        for k in ("early_stopping_rounds", "min_delta", "eval_k", "eval_metric"):
            if k in cfg.get("bdt", {}):
                bdt_kw[k] = cfg["bdt"][k]
        m = BDT(train_data, lambda_decay=decay, objective_weights=weights,
                eval_seasonal_scale=ss, **bdt_kw)
        m.fit()
        return m

    elif family == "rf":
        m = RF(train_data, lambda_decay=decay, objective_weights=weights,
               random_state=p["seed"],
               n_estimators_=cfg["rf"]["n_estimators"],
               bagging_fraction=cfg["rf"]["bagging_fraction"],
               feature_fraction=cfg["rf"]["feature_fraction"],
               max_depth=depth,
               min_samples_leaf=p["min_leaf"],
               n_discrete_lev=p["n_level"],
               num_threads=cfg["rf"].get("num_threads", p["nthreads"]),
               prebin=p["prebin"],
               aggregation=p["agg"],
               n_jobs=cfg["rf"].get("n_jobs", 1),
               backend=cfg["rf"].get("backend", "thread"),
               )
        m.fit()
        return m

    raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Iterative rollout (adapted from iterative_vs_direct_mgbt_mrf.py)
# ---------------------------------------------------------------------------
def iterative_predict(model, test_data, sm, lag, horizon, max_seas, freq, use_last=False):
    """Roll forward H steps from one starting row per series.

    use_last=False (default): start from the EARLIEST test row per series.
        Correct for test_k1 (max_ahead=1), where earliest = standard leave-last-H-out start.
    use_last=True: start from the LATEST test row per series (cumcount_back==0).
        Required for test_mt (max_ahead=H), where the latest row is the standard start.
    """
    x = test_data.x
    df_idx = test_data.index.copy()
    df_idx["row_pos"] = np.arange(len(df_idx))
    df_idx["cumcount_back"] = df_idx.groupby("series", observed=True).cumcount(ascending=False)
    if use_last:
        earliest = df_idx[df_idx["cumcount_back"] == 0]
    else:
        earliest = df_idx[df_idx["cumcount_back"] ==
                          df_idx.groupby("series", observed=True)["cumcount_back"].transform("max")]
    earliest = earliest.sort_values(by="series")
    series_order = earliest["series"].to_numpy()
    starting_rows = earliest["row_pos"].to_numpy()
    x_arr = x.iloc[starting_rows].to_numpy().astype(MT_DTYPE).copy()
    col_names = list(x.columns)
    lag_idx = [col_names.index(f"lag_{i+1}") for i in range(lag)]
    sin_idx = col_names.index("sin_season") if "sin_season" in col_names else None
    cos_idx = col_names.index("cos_season") if "cos_season" in col_names else None
    if sin_idx is not None:
        period = float(max_seas) if max_seas > 1 else 1.0
        delta = 2.0 * np.pi / period

    preds_scaled = np.zeros((len(series_order), horizon), dtype=MT_DTYPE)
    for h in range(horizon):
        x_df = pd.DataFrame(x_arr, columns=col_names)
        for c in x_df.columns:
            if x_df[c].dtype != MT_DTYPE:
                x_df[c] = x_df[c].astype(MT_DTYPE)
        pred = model.predict(x_df)
        if hasattr(pred, "to_numpy"):
            pred = pred.to_numpy()
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] > 1:
            pred = pred[:, 0]
        elif pred.size > len(x_df) and pred.size % len(x_df) == 0:
            pred = pred.reshape(len(x_df), -1)[:, 0]
        pred = pred.ravel().astype(MT_DTYPE)
        preds_scaled[:, h] = pred
        for j in range(lag - 1, 0, -1):
            x_arr[:, lag_idx[j]] = x_arr[:, lag_idx[j - 1]]
        x_arr[:, lag_idx[0]] = pred
        if sin_idx is not None and cos_idx is not None:
            sin_cur = x_arr[:, sin_idx].copy()
            cos_cur = x_arr[:, cos_idx].copy()
            x_arr[:, sin_idx] = sin_cur * np.cos(delta) + cos_cur * np.sin(delta)
            x_arr[:, cos_idx] = cos_cur * np.cos(delta) - sin_cur * np.sin(delta)

    if hasattr(sm, "columns") and "mean_series" in sm.columns:
        scale_lookup = dict(sm[["series", "mean_series"]].to_numpy())
        scl = np.array([float(scale_lookup.get(s, 1.0)) for s in series_order])
    else:
        scl = np.ones(len(series_order))
    return series_order, preds_scaled.astype(np.float64) * scl[:, None]


def seasonal_scale(combined, period, horizon):
    scales = {}
    for series, grp in combined.groupby("series", observed=True):
        y = grp["y"].to_numpy().astype(np.float64)
        train = y[:-horizon] if len(y) > horizon else y
        if len(train) > period:
            diffs = np.abs(train[period:] - train[:-period])
            scales[series] = diffs.mean() if len(diffs) > 0 else np.nan
        else:
            scales[series] = np.nan
    return pd.Series(scales)


# ---------------------------------------------------------------------------
# Per-(dataset, family) runner
# ---------------------------------------------------------------------------
def run(ds_name, family):
    info = INFO_TABLE.loc[ds_name]
    data_file = REPO / CFG["data_path"] / info["file_name"]
    combined, freq, seas, _h, ext_features = organize_repo_data(str(data_file))
    horizon = int(info["horizon"]) if not pd.isna(info["horizon"]) else _h
    lag = int(info["predetermined_lag"]) if not pd.isna(info["predetermined_lag"]) else int(np.ceil(max(seas) * 1.25))
    depth, decay = get_base_params(ds_name)
    sp = ["sin_season", "cos_season"] if freq != "1Y" else None

    print(f"  {family.upper():4s} | {ds_name} | H={horizon} L={lag} depth={depth} decay={decay}", flush=True)

    # Seasonal scale (drop last H raw obs from each series, same as the benchmark)
    ss = compute_seasonal_scale(combined, "y", horizon, int(max(seas)))

    # --- Multi-target features (direct fit, max_ahead=H) ---
    lag_g, tar_g, _ = build_features_and_targets(
        combined, "y", lag, horizon, False,
        ["index", "series"], ["season_index"], ext_features, freq, int(max(seas)))
    grp = lag_g.groupby("series").cumcount(ascending=False)
    trf  = lag_g[grp >= horizon]; trt = tar_g[grp >= horizon]
    tef  = lag_g[grp < horizon];  tet = tar_g[grp < horizon]
    trf2, tef2, trt2, tet2, sm = mean_scale_data(trf, tef, trt, tet, lag, horizon, True)

    train_mt = DataMt(max_ahead=horizon, n_derivative=0, penalty_list=sp,
                      series_means=sm, int_convert=info["integer_conversion"])
    train_mt.transform(trf2, trt2)
    test_mt = DataMt(max_ahead=horizon, n_derivative=0, penalty_list=sp,
                     series_means=sm, int_convert=info["integer_conversion"])
    test_mt.transform(tef2, tet2)

    # Last row per series in test_mt = standard leave-last-H-out evaluation start.
    # test_mt has H rows per series; grp=0 (cumcount_back==0) is the latest,
    # with features at t=T-H-1 — same starting point as iterative (test_k1 earliest).
    _cb = test_mt.index.groupby("series").cumcount(ascending=False)
    _last_mask = (_cb == 0).values
    series_order = test_mt.index.loc[_last_mask, "series"].to_numpy()

    # --- Single-target features (iterative fit, max_ahead=1) ---
    lag_g1, tar_g1, _ = build_features_and_targets(
        combined, "y", lag, 1, False,
        ["index", "series"], ["season_index"], ext_features, freq, int(max(seas)))
    grp1 = lag_g1.groupby("series").cumcount(ascending=False)
    trf1 = lag_g1[grp1 >= horizon]; trt1 = tar_g1[grp1 >= horizon]
    tef1 = lag_g1[grp1 < horizon];  tet1 = tar_g1[grp1 < horizon]
    trf12, tef12, trt12, tet12, sm1 = mean_scale_data(trf1, tef1, trt1, tet1, lag, 1, True)

    train_k1 = DataMt(max_ahead=1, n_derivative=0, penalty_list=sp,
                      series_means=sm1, int_convert=info["integer_conversion"])
    train_k1.transform(trf12, trt12)
    test_k1 = DataMt(max_ahead=1, n_derivative=0, penalty_list=sp,
                     series_means=sm1, int_convert=info["integer_conversion"])
    test_k1.transform(tef12, tet12)

    # --- Direct fit and predict ---
    m_direct = fit_model(family, train_mt, depth, decay, CFG, ss=ss)
    direct_pred_raw = m_direct.predict(test_mt.x)
    direct_pred_all, _ = test_mt.process_predictions(direct_pred_raw)

    # Actuals in original scale, filtered to last row per series (standard eval window).
    # test_mt has H rows per series; last row (cumcount_back==0) is t=T-H-1,
    # predicting y[T-H:T] — the standard leave-last-H-out test window.
    actual = test_mt.y_org
    if hasattr(actual, "to_numpy"):
        actual = actual.to_numpy()
    actual = actual[_last_mask]                           # (N_series, H)
    direct_pred = np.asarray(direct_pred_all)[_last_mask] # (N_series, H)

    mase_direct_vec = row_wise_mase(actual, direct_pred, ss.loc[series_order].values)

    # --- Variant 3: multi-target model used iteratively (pred[:,0] at each step) ---
    # use_last=True: start from last test_mt row per series (t=T-H-1, same as iterative start)
    mi_series, mi_pred_unscaled = iterative_predict(
        m_direct, test_mt, sm, lag, horizon, int(max(seas)), freq, use_last=True)
    del m_direct; gc.collect()
    mi_lookup = {s: i for i, s in enumerate(mi_series)}
    mi_pred_aligned = np.array([mi_pred_unscaled[mi_lookup[s]] for s in series_order])
    mase_mi_vec = row_wise_mase(actual, mi_pred_aligned, ss.loc[series_order].values)

    # --- Iterative fit and predict ---
    m_iter = fit_model(family, train_k1, depth, decay, CFG, ss=ss)
    iter_series, iter_pred_unscaled = iterative_predict(
        m_iter, test_k1, sm1, lag, horizon, int(max(seas)), freq)
    del m_iter; gc.collect()

    # Align iterative predictions to same series order as direct
    iter_lookup = {s: i for i, s in enumerate(iter_series)}
    iter_pred_aligned = np.array([iter_pred_unscaled[iter_lookup[s]] for s in series_order])

    mase_iter_vec = row_wise_mase(actual, iter_pred_aligned, ss.loc[series_order].values)

    # Per-horizon MASE (element-wise MAE / scale, then aggregate across series)
    err_direct = np.abs(actual - direct_pred)
    err_iter   = np.abs(actual - iter_pred_aligned)
    err_mi     = np.abs(actual - mi_pred_aligned)
    scl = ss.loc[series_order].values
    mase_d_ph  = err_direct / scl[:, None]   # (N, H)
    mase_i_ph  = err_iter   / scl[:, None]   # (N, H)
    mase_mi_ph = err_mi     / scl[:, None]   # (N, H)

    rows = []
    for h in range(horizon):
        rows.append({"dataset": ds_name, "family": family, "method": "direct", "h": h,
                     "mean_mase":   float(np.nanmean(mase_d_ph[:, h])),
                     "median_mase": float(np.nanmedian(mase_d_ph[:, h]))})
        rows.append({"dataset": ds_name, "family": family, "method": "iterative", "h": h,
                     "mean_mase":   float(np.nanmean(mase_i_ph[:, h])),
                     "median_mase": float(np.nanmedian(mase_i_ph[:, h]))})
        rows.append({"dataset": ds_name, "family": family, "method": "multi_iter", "h": h,
                     "mean_mase":   float(np.nanmean(mase_mi_ph[:, h])),
                     "median_mase": float(np.nanmedian(mase_mi_ph[:, h]))})

    dir_med = float(np.nanmedian(mase_direct_vec))
    itr_med = float(np.nanmedian(mase_iter_vec))
    mi_med  = float(np.nanmedian(mase_mi_vec))
    rel_i  = itr_med / dir_med if dir_med > 0 else float("nan")
    rel_mi = mi_med  / dir_med if dir_med > 0 else float("nan")
    print(f"         direct={dir_med:.4f}  iter={itr_med:.4f}(rel={rel_i:.3f})"
          f"  multi_iter={mi_med:.4f}(rel={rel_mi:.3f})", flush=True)

    del train_mt, test_mt, train_k1, test_k1
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Main loop — resume-safe
# ---------------------------------------------------------------------------
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--datasets", nargs="*", default=None)
_parser.add_argument("--out-suffix", default="")
_args, _ = _parser.parse_known_args()
if _args.datasets:
    DATASETS = _args.datasets
if _args.out_suffix:
    OUT_DIR = OUT_DIR.parent / (OUT_DIR.name + _args.out_suffix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "per_horizon_mase.csv"
done = set()
all_rows = []
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    # Only mark done if multi_iter rows are present (older runs may lack them)
    mi_pairs = set(zip(
        existing[existing["method"] == "multi_iter"]["dataset"],
        existing[existing["method"] == "multi_iter"]["family"],
    ))
    done = mi_pairs
    # Keep only rows for fully-done pairs (drops stale direct/iterative rows for pairs to rerun)
    mask = existing.apply(lambda r: (r["dataset"], r["family"]) in done, axis=1)
    all_rows = existing[mask].to_dict("records")
    n_total = len(set(zip(existing["dataset"], existing["family"])))
    print(f"Resuming — {len(done)} fully done, {n_total - len(done)} pairs to rerun")

for ds in DATASETS:
    if ds not in INFO_TABLE.index:
        print(f"  SKIP {ds} — not in INFO_TABLE"); continue
    for fam in FAMILIES:
        if (ds, fam) in done:
            print(f"  SKIP {fam}/{ds} — already in CSV"); continue
        print(f"\n[{DATASETS.index(ds)*len(FAMILIES) + FAMILIES.index(fam) + 1}"
              f"/{len(DATASETS)*len(FAMILIES)}] {fam.upper()} / {ds}", flush=True)
        try:
            rows = run(ds, fam)
            all_rows.extend(rows)
            done.add((ds, fam))
            pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        except Exception as e:
            import traceback
            print(f"  !! FAILED: {e}")
            traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if all_rows:
    df = pd.DataFrame(all_rows)
    lines = [
        "ITERATIVE-vs-DIRECT  |  all three families  |  post-fix production code",
        "rel = median_MASE(method) / median_MASE(direct)",
        "rel > 1 → direct wins   rel < 1 → method wins",
        "iterative  = K=1 model rolled forward H steps",
        "multi_iter = K=H model, pred[:,0] fed back as lag at each step",
        "=" * 88, ""
    ]
    for fam in FAMILIES:
        lines.append(f"--- Family: {fam.upper()} ---")
        sub_f = df[df["family"] == fam]
        rel_i_all = []
        rel_mi_all = []
        for ds in DATASETS:
            sub = sub_f[sub_f["dataset"] == ds]
            if sub.empty: continue
            piv = sub.pivot_table(index="h", columns="method", values="median_mase", aggfunc="first")
            if "direct" not in piv.columns or "iterative" not in piv.columns: continue
            rel_i = piv["iterative"] / piv["direct"]
            rel_i_all.extend(rel_i.tolist())
            i_wins = int((rel_i < 1.0).sum())
            i_rel  = float(rel_i.mean())
            d_overall = float(piv["direct"].mean())
            i_overall = float(piv["iterative"].mean())
            if "multi_iter" in piv.columns:
                rel_mi = piv["multi_iter"] / piv["direct"]
                rel_mi_all.extend(rel_mi.tolist())
                mi_wins = int((rel_mi < 1.0).sum())
                mi_rel  = float(rel_mi.mean())
                mi_overall = float(piv["multi_iter"].mean())
                mi_str = f"  mi_wins={mi_wins}/{len(piv)}  mi_rel={mi_rel:.3f}  mi={mi_overall:.4f}"
            else:
                mi_str = ""
            lines.append(f"  {ds:<22}  H={len(piv):2d}  "
                         f"iter_wins={i_wins}/{len(piv)}  iter_rel={i_rel:.3f}  "
                         f"direct={d_overall:.4f}  iter={i_overall:.4f}{mi_str}")
        if rel_i_all:
            arr_i  = np.array(rel_i_all)
            arr_mi = np.array(rel_mi_all) if rel_mi_all else None
            mi_overall_str = (f"  mi_rel={arr_mi.mean():.3f}  mi_wins={int((arr_mi<1).sum())}/{len(arr_mi)}"
                              if arr_mi is not None else "")
            lines.append(f"  {'OVERALL':<22}  n={len(arr_i)}  "
                         f"iter_rel={arr_i.mean():.3f}  iter_wins={int((arr_i<1).sum())}/{len(arr_i)}"
                         f"{mi_overall_str}")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    (OUT_DIR / "summary.txt").write_text(text + "\n")
    print(f"\nResults: {OUT_CSV}")
    print(f"Summary: {OUT_DIR / 'summary.txt'}")
