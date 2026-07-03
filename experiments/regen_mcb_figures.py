"""Regenerate the two MASE MCB critical-distance figures.

Tree-method ranks are read from the shipped wide median-MASE table:
  - tree columns now read from notes/src/wide_median_mase.csv
    (built from results/canonical/summary_fixed.csv, mase_med, with
     seas:=base / both:=deriv for the two non-seasonal synthetics)
  - external baselines + SETAR sources UNCHANGED 
  - outputs go to notes/mcb/ only (NOT submission/figures)

2026-07-04: figure labels renamed to the PAPER naming (mDT/mGBT/mRF + base/deriv/
seas/both; DT/DT-D, RF/RF-D, ET/ET-D, CB/CB-D; ETS/ARIMA/naive/sNaive; ST_*/SF_*)
and the dev title tag removed, so the figures match the manuscript text.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata

PAPER = Path("/media/baydogan/files/Research/UvA/Paper")
SRC = PAPER / "notes" / "src"
OUT_DIR = PAPER / "notes" / "mcb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COV_DATASETS = {"Favorita Daily with Cov", "Kaggle Daily with Cov", "Rosmann Daily with Cov"}

# ---------------------------------------------------------------------------
# Assemble the wide table  (tree = the refreshed; ext + setar = unchanged)
# ---------------------------------------------------------------------------
tree = pd.read_csv(SRC / "wide_median_mase.csv", index_col="dataset")

long = pd.read_csv(PAPER / "revision" / "tables" / "per_series_mase_all_methods_long.csv")
ext = long.groupby(["dataset", "method"])["mase"].median().unstack("method")
EXT_COLS = ["naive", "snaive", "arima", "ets",
            "skDT_uniform", "skDT_diff", "skET_uniform", "skET_diff",
            "skRF_uniform", "skRF_diff", "CatBoost_uniform", "CatBoost_diff"]
ext = ext[EXT_COLS]

x = pd.read_excel("/media/baydogan/files/Research/UvA/Experiments/paper_all_results_final_v2.xlsx",
                  sheet_name="MASE", header=[0, 1], index_col=0)
SETAR_MAP = {
    "SETAR-Tree_LT": ("median", "Tree.Lin.Test"),
    "SETAR-Tree_ER": ("median", "Tree.Error.Red"),
    "SETAR-Tree_ER+LT": ("median", "Tree.Lin.Test.Error.Red"),
    "SETAR-Forest_S": ("median", "Forest.Significance"),
    "SETAR-Forest_ER": ("median", "Forest.Error.Red"),
    "SETAR-Forest_S+ER": ("median", "Forest.Significance.Error.Red"),
}
setar = pd.DataFrame({k: pd.to_numeric(x[v], errors="coerce") for k, v in SETAR_MAP.items()})
setar.index.name = "dataset"

wide = tree.join(ext, how="left").join(setar, how="left")
print(f"wide table: {wide.shape[0]} datasets x {wide.shape[1]} methods")


def average_ranks(df):
    ranks = df.apply(lambda row: rankdata(row.values), axis=1, result_type="expand")
    ranks.columns = df.columns
    return ranks.mean(axis=0).sort_values()


def critical_distance(k, n, alpha=0.05):
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
         9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
         15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
         21: 3.569, 22: 3.593, 23: 3.616, 24: 3.637, 25: 3.658, 26: 3.679,
         27: 3.699, 28: 3.719, 29: 3.737, 30: 3.755}
    return q[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def emit(df, title, out_path, fig_w=9.0):
    df = df.dropna(axis=0, how="any")
    k, n = df.shape[1], df.shape[0]
    ranks = average_ranks(df)
    sig = sp.posthoc_nemenyi_friedman(df.values)
    sig.index = sig.columns = df.columns
    cd = critical_distance(k, n)
    _, p_fried = friedmanchisquare(*[df[c].values for c in df.columns])

    fig, ax = plt.subplots(figsize=(fig_w, 0.22 * k + 1.2))
    sp.critical_difference_diagram(
        ranks=ranks, sig_matrix=sig, ax=ax, cd=cd,
        label_fmt_left="{label}  ({rank:.2f})",
        label_fmt_right="({rank:.2f})  {label}",
        text_h_margin=0.01,
    )
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\n  wrote {out_path}  (k={k}, n={n}, p={p_fried:.2e}, CD={cd:.3f})")
    print(f"  datasets used ({n}): {sorted(df.index)}")
    best = ranks.index[0]
    inside = ranks[ranks <= ranks.iloc[0] + cd]
    print(f"  ranks: " + ", ".join(f"{m}={r:.2f}" for m, r in ranks.items()))
    print(f"  CD band of best ({best}): {list(inside.index)}")
    return ranks, cd, p_fried


PAPER_T2_COLS = [
    "naive", "snaive", "arima", "ets",
    "skDT_uniform", "skDT_diff",
    "dt_base", "dt_deriv", "dt_seas", "dt_both",
    "skET_uniform", "skET_diff",
    "skRF_uniform", "skRF_diff",
    "rf_base", "rf_deriv", "rf_seas", "rf_both",
    "CatBoost_uniform", "CatBoost_diff",
    "gbt_base", "gbt_deriv", "gbt_seas", "gbt_both",
]
SETAR_COLS = [
    "dt_base", "dt_deriv", "dt_seas", "dt_both",
    "gbt_base", "gbt_deriv", "gbt_seas", "gbt_both",
    "rf_base", "rf_deriv", "rf_seas", "rf_both",
    "SETAR-Tree_ER", "SETAR-Tree_LT", "SETAR-Tree_ER+LT",
    "SETAR-Forest_ER", "SETAR-Forest_S", "SETAR-Forest_S+ER",
]

# Paper-conform display names (match the manuscript text, tables, and the
# original submission's figure conventions).
PAPER_LABELS = {
    "dt_base": "mDT base", "dt_deriv": "mDT deriv", "dt_seas": "mDT seas", "dt_both": "mDT both",
    "gbt_base": "mGBT base", "gbt_deriv": "mGBT deriv", "gbt_seas": "mGBT seas", "gbt_both": "mGBT both",
    "rf_base": "mRF base", "rf_deriv": "mRF deriv", "rf_seas": "mRF seas", "rf_both": "mRF both",
    "skDT_uniform": "DT", "skDT_diff": "DT-D",
    "skET_uniform": "ET", "skET_diff": "ET-D",
    "skRF_uniform": "RF", "skRF_diff": "RF-D",
    "CatBoost_uniform": "CB", "CatBoost_diff": "CB-D",
    "ets": "ETS", "arima": "ARIMA", "naive": "naive", "snaive": "sNaive",
    "SETAR-Tree_LT": "ST_LT", "SETAR-Tree_ER": "ST_ER", "SETAR-Tree_ER+LT": "ST_ER+LT",
    "SETAR-Forest_S": "SF_S", "SETAR-Forest_ER": "SF_ER", "SETAR-Forest_S+ER": "SF_S+ER",
}

print("=== MCB / Paper Table 2 set (median MASE, identical-input subset) -> fig:maseranktestwosetar ===")
sub24 = wide.loc[[d for d in wide.index if d not in COV_DATASETS], PAPER_T2_COLS]
emit(sub24.rename(columns=PAPER_LABELS), "",
     OUT_DIR / "mcb_paper_t2_median.pdf", fig_w=10.0)

print("\n=== MCB / with SETAR (median MASE, identical-input: no Cov datasets) -> fig:maseranktestwithsetar ===")
sub18 = wide.loc[[d for d in wide.index if d not in COV_DATASETS], SETAR_COLS]
emit(sub18.rename(columns=PAPER_LABELS), "",
     OUT_DIR / "mcb_with_setar_median.pdf", fig_w=11.0)
print("\nDONE. Outputs in notes/mcb/ (submission untouched).")
