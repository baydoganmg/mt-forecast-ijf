"""Noise sensitivity for R2.5, v2: perturb TRAIN only, evaluate on CLEAN test.

Cleaner experiment than v1:
  - Identify train vs test rows BEFORE perturbing
  - Add N(0, sigma * series_std) noise to TRAIN y only
  - Build features (lag from perturbed train), train model
  - At test, use the ORIGINAL (clean) test actuals to compute MASE
  - Also compute "noisy-train, noisy-test" for reference

Question: as noise grows,
  (a) does CV-best lambda decrease? (self-attenuation)
  (b) does using a high lambda HURT when noise is high?
  (c) what's the regularizer's actual test-MASE benefit at each noise level?
"""
from __future__ import annotations
import sys, os, gc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ.setdefault("MT_DTYPE", "fp32")

import numpy as np
import pandas as pd
import yaml

from mttrees.tree import DataMt, CTree_MT
from mttrees.utils import organize_repo_data
from experiments.run_benchmark import build_features_and_targets, mean_scale_data

CFG = yaml.safe_load(open(REPO / "configs" / "benchmark.yaml"))
INFO_TABLE = pd.read_excel(CFG["info_table"], "repo_data").set_index("Dataset")

OUT_DIR = REPO / "results" / "noise_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DS = "Hospital"
info = INFO_TABLE.loc[DS]
data_file = REPO / CFG["data_path"] / info["file_name"]

combined_orig, freq, seasonality, _h, ext_features = organize_repo_data(str(data_file))
horizon = int(info["horizon"]) if not pd.isna(info["horizon"]) else _h
max_seas = max(seasonality)
lag = int(info["predetermined_lag"]) if not pd.isna(info["predetermined_lag"]) else int(np.ceil(max_seas * 1.25))
period = int(seasonality[0])
H = int(horizon)
print(f"Dataset: {DS}, lag={lag}, horizon={H}, period={period}", flush=True)

SIGMAS = [0.0, 0.1, 0.3, 1.0, 3.0]
LAMBDAS = [0.0, 0.2, 0.5, 1.0, 2.0]
DECAY = 0.5
MAX_DEPTH = 10
N_SEEDS = 3

# Per-series std for noise scaling (compute on ORIGINAL data)
series_stds_orig = combined_orig.groupby("series", observed=True)["y"].std().to_dict()

# Pre-compute ORIGINAL-data snaive scales (these never change across runs)
def compute_scales(combined_df):
    scales = {}
    for series, grp in combined_df.groupby("series", observed=True):
        y = grp["y"].to_numpy().astype(np.float64)
        train = y[:-H] if len(y) > H else y
        if len(train) > period:
            diffs = np.abs(train[period:] - train[:-period])
            scales[series] = diffs.mean() if len(diffs) > 0 else np.nan
        else:
            scales[series] = np.nan
    return pd.Series(scales)

scales_orig = compute_scales(combined_orig)

# Pre-compute CLEAN test y_org (from original data, test = last H per series)
def extract_test_y_org(combined_df):
    """Per-series last H observations stacked into shape (n_series, H)."""
    rows = []
    series_order = []
    for series, grp in combined_df.groupby("series", observed=True):
        y = grp["y"].to_numpy().astype(np.float64)
        if len(y) >= H:
            rows.append(y[-H:])
            series_order.append(series)
    return np.array(rows), series_order

clean_test_y_orig, clean_series_order = extract_test_y_org(combined_orig)
print(f"Clean test set: shape={clean_test_y_orig.shape}", flush=True)


def run_one(sigma, lam, seed):
    """Return dict with both noisy-test and clean-test MASE."""
    np.random.seed(seed)
    cd = combined_orig.copy()
    # Identify train rows per series (all except last H), perturb only those
    if sigma > 0:
        # Track per-row train flag
        cumcount_back = cd.groupby("series", observed=True).cumcount(ascending=False)
        train_mask = (cumcount_back >= H).to_numpy()
        std_arr = pd.Series(cd["series"].astype(str).map(series_stds_orig).values,
                            index=cd.index).fillna(1.0).to_numpy(dtype=float)
        noise = np.random.normal(0, sigma * std_arr)
        # zero noise on test rows
        noise = np.where(train_mask, noise, 0.0)
        cd.loc[:, "y"] = cd["y"].to_numpy() + noise
    lag_g, target_g, _ = build_features_and_targets(
        cd, "y", lag, H, False,
        ["index", "series"], ["season_index"], ext_features, freq, int(max_seas))
    del cd; gc.collect()
    grp = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grp >= 1]; tef = lag_g[grp < 1]
    grp2 = target_g.groupby("series").cumcount(ascending=False)
    trt = target_g[grp2 >= 1]; tet = target_g[grp2 < 1]
    del lag_g, target_g; gc.collect()
    trf2, tef2, trt2, tet2, sm = mean_scale_data(trf, tef, trt, tet, lag, H, True)
    sp = ["sin_season", "cos_season"] if freq != "1Y" else None
    train_data = DataMt(max_ahead=H, n_derivative=1, penalty_list=sp,
                        series_means=sm, int_convert=info["integer_conversion"])
    train_data.transform(trf2, trt2)
    test_data = DataMt(max_ahead=H, n_derivative=1, penalty_list=sp,
                       series_means=sm, int_convert=info["integer_conversion"])
    test_data.transform(tef2, tet2)
    tree = CTree_MT(max_depth=MAX_DEPTH, min_samples_leaf=5, mtry=1.0,
                    samp_frac=1.0, prebin=True, n_discrete_lev=32)
    tree.arrangeObjective(train_data, lambda_decay=DECAY,
                          objective_weights=[1.0, lam, 0.0])
    tree.fit(train_data.x, train_data.y)
    pred_raw = tree.predict(test_data.x)
    pred, _ = test_data.process_predictions(pred_raw)
    pred = np.asarray(pred)
    # Compute MASE against NOISY test (what's in test_data.y_org)
    ac_noisy = test_data.y_org
    err_noisy = np.abs(ac_noisy - pred)
    sids = test_data.index["series"].to_numpy()
    scl_noisy = scales_orig.loc[sids].to_numpy()  # use ORIGINAL scales (stable)
    mase_noisy = (err_noisy / scl_noisy[:, None]).mean(axis=1)
    # Compute MASE against CLEAN test (the recovered-signal metric)
    # Match clean_test_y_orig rows to the order of sids
    series_to_clean_row = {s: i for i, s in enumerate(clean_series_order)}
    aligned = np.array([clean_test_y_orig[series_to_clean_row[s]] if s in series_to_clean_row
                        else np.full(H, np.nan) for s in sids])
    err_clean = np.abs(aligned - pred)
    mase_clean = (err_clean / scl_noisy[:, None]).mean(axis=1)
    return {
        "median_mase_noisy_test": float(np.nanmedian(mase_noisy)),
        "median_mase_clean_test": float(np.nanmedian(mase_clean)),
        "mean_mase_noisy_test":   float(np.nanmean(mase_noisy)),
        "mean_mase_clean_test":   float(np.nanmean(mase_clean)),
    }


rows = []
print(f"\n{'sigma':>7s} {'lambda':>8s} {'seed':>5s} {'med_clean':>10s} {'med_noisy':>10s}")
for sigma in SIGMAS:
    for lam in LAMBDAS:
        for seed in range(N_SEEDS):
            t = pd.Timestamp.now()
            r = run_one(sigma, lam, seed)
            dt = (pd.Timestamp.now() - t).total_seconds()
            print(f"{sigma:>7.2f} {lam:>8.2f} {seed:>5d} "
                  f"{r['median_mase_clean_test']:>10.4f} "
                  f"{r['median_mase_noisy_test']:>10.4f}  ({dt:.1f}s)", flush=True)
            rows.append({"sigma": sigma, "lambda": lam, "seed": seed,
                         "elapsed_s": dt, **r})
            gc.collect()

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "summary.csv", index=False)

# Aggregate: mean over seeds per (sigma, lambda)
def piv(col):
    return df.groupby(["sigma","lambda"])[col].mean().reset_index().pivot_table(
        index="lambda", columns="sigma", values=col)

clean = piv("median_mase_clean_test")
noisy = piv("median_mase_noisy_test")

print("\n=== Median MASE on CLEAN test (signal-recovery metric, mean over seeds) ===")
print(clean.round(4).to_string())
print("\n=== Median MASE on NOISY test (noisy-actuals metric) ===")
print(noisy.round(4).to_string())

# Best lambda per sigma (on clean test)
print("\n=== CV-best lambda per sigma based on CLEAN-test MASE ===")
best_clean = clean.idxmin(axis=0)
for s in clean.columns:
    print(f"  sigma={s}: best lambda = {best_clean[s]}  (clean MASE = {clean.loc[best_clean[s], s]:.4f})")

print("\n=== CV-best lambda per sigma based on NOISY-test MASE ===")
best_noisy = noisy.idxmin(axis=0)
for s in noisy.columns:
    print(f"  sigma={s}: best lambda = {best_noisy[s]}  (noisy MASE = {noisy.loc[best_noisy[s], s]:.4f})")

# Worst-case cost of "wrong" lambda
print("\n=== Cost of using lambda=2.0 when sigma is high (vs the best lambda) ===")
for s in clean.columns:
    best_val_clean = clean.loc[:, s].min()
    high_lam_val_clean = clean.loc[2.0, s]
    pct = (high_lam_val_clean - best_val_clean) / best_val_clean * 100
    print(f"  sigma={s}: best clean MASE = {best_val_clean:.4f}, lambda=2.0 clean MASE = {high_lam_val_clean:.4f} (+{pct:.2f}%)")

# Write summary text
lines = ["Noise-sensitivity v2 for R2.5 (Hospital, mDT_deriv)",
         "TRAIN-only perturbation; evaluate on CLEAN ORIGINAL test",
         "=" * 60, "",
         "Median MASE on CLEAN test (mean over 3 seeds):",
         clean.round(4).to_string(), "",
         "Median MASE on NOISY test (mean over 3 seeds):",
         noisy.round(4).to_string(), "",
         "CV-best lambda per sigma (clean test):",
         "\n".join(f"  sigma={s}: best lambda = {best_clean[s]:.1f}, clean MASE = {clean.loc[best_clean[s], s]:.4f}"
                   for s in clean.columns), "",
         "Cost of lambda=2.0 (vs best) on clean test:",
         "\n".join(f"  sigma={s}: best={clean.loc[:, s].min():.4f}, lambda=2.0={clean.loc[2.0, s]:.4f} (+{(clean.loc[2.0, s]/clean.loc[:, s].min()-1)*100:.2f}%)"
                   for s in clean.columns),
         ""]
(OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
print(f"\nSaved {OUT_DIR / 'summary.csv'} and summary.txt")
