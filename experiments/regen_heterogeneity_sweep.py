"""Regenerate Figure 5 (heterogeneity-penalty sweep on Rosmann Daily) in MASE.

The figure reports train/test median MASE on Rosmann Daily across the heterogeneity weight
$\\lambda_\\CH \\in \\{0, 0.2, 0.4, 0.6, 0.8, 1\\}$ for a range of tree depths,
with $r=1$ and $\\lambda_\\CC = 0$. Rosmann Daily is a seasonal benchmark
with a well-defined seasonal-naive baseline, so MASE is the natural metric
to keep the §3.4 parameter illustration consistent with the §4 main results.

This script repeats that experiment and emits a MASE companion plot at
submission/figures/mase_seasonal.pdf.
"""
from __future__ import annotations
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from mttrees.tree import DataMt, CTree_MT
from mttrees.utils import organize_repo_data
from experiments.run_benchmark import build_features_and_targets, mean_scale_data

CFG = yaml.safe_load(open(REPO / "configs" / "benchmark.yaml"))
INFO = pd.read_excel(CFG["info_table"], "repo_data").set_index("Dataset")
DATASET = "Rosmann Daily"
DEPTHS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
DEEP_DEPTH = max(DEPTHS)
BETA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DECAY = 1.0  # r = 1
LAM = 0.0    # \lambda_C fixed to 0


def setup(ds_name):
    info = INFO.loc[ds_name]
    data_file = REPO / CFG["data_path"] / info["file_name"]
    combined, freq, seas, h, ext_features = organize_repo_data(str(data_file))
    horizon = int(info["horizon"]) if not pd.isna(info["horizon"]) else h
    lag = int(info["predetermined_lag"]) if not pd.isna(info["predetermined_lag"]) else int(np.ceil(max(seas) * 1.25))
    max_S = int(max(seas))

    lag_g, tar_g, _ = build_features_and_targets(
        combined, "y", lag, horizon, False,
        ["index", "series"], ["season_index"], ext_features, freq, max_S)
    grouped = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grouped >= horizon]
    tef = lag_g[grouped < horizon]
    grouped = tar_g.groupby("series").cumcount(ascending=False)
    trt = tar_g[grouped >= horizon]
    tet = tar_g[grouped < horizon]
    trf, tef, trt, tet, sm = mean_scale_data(trf, tef, trt, tet, lag, horizon, mean_scale=True)
    sp = ["sin_season", "cos_season"] if freq != "1Y" else None
    train_data = DataMt(max_ahead=int(horizon), n_derivative=1,
                         penalty_list=sp, series_means=sm,
                         int_convert=info["integer_conversion"])
    train_data.transform(trf, trt)
    test_data = DataMt(max_ahead=int(horizon), n_derivative=1,
                        penalty_list=sp, series_means=sm,
                        int_convert=info["integer_conversion"])
    test_data.transform(tef, tet)

    s = combined[["index", "series", "y"]]
    grouped = s.groupby("series").cumcount(ascending=False)
    s_train = s[grouped >= horizon].copy()
    s_train["sd"] = np.abs(s_train["y"] - s_train.groupby("series")["y"].shift(max_S))
    scale = s_train.groupby("series")["sd"].mean()
    return train_data, test_data, scale


def compute_mase(data, predictions, scale):
    series_order = data.index["series"].to_numpy()
    A = data.y_org
    F = predictions
    if F.ndim == 1:
        F = F.reshape(-1, 1); A = A.reshape(-1, 1)
    sc = scale.loc[series_order].to_numpy()
    valid = (sc > 0) & np.isfinite(sc)
    mae = np.mean(np.abs(F[valid] - A[valid]), axis=1)
    return mae / sc[valid]


def main():
    out_csv = REPO / "results" / "fig4_seasonal_mase" / f"{DATASET.lower().replace(' ', '_')}.csv"
    if out_csv.exists():
        print(f"reusing cached fits at {out_csv}", flush=True)
        df = pd.read_csv(out_csv)
    else:
        print(f"Setup: {DATASET}", flush=True)
        train_data, test_data, scale = setup(DATASET)
        rows = []
        for bet in BETA_VALUES:
            tree = CTree_MT(max_depth=DEEP_DEPTH, verbose=0,
                             min_samples_leaf=CFG["min_samples_leaf"],
                             n_discrete_lev=CFG["n_discrete_lev"], num_threads=2,
                             prebin=CFG.get("prebin", True))
            tree.arrangeObjective(train_data, DECAY, [1.0, LAM, bet])
            tree.fit(train_data.x, train_data.y)
            print(f"  beta={bet:.1f}  fit done", flush=True)
            for d in DEPTHS:
                train_pred_raw = tree.predict(train_data.x, depth=d)
                train_pred, _ = train_data.process_predictions(train_pred_raw)
                train_m = float(np.nanmedian(compute_mase(train_data, train_pred, scale)))
                test_pred_raw = tree.predict(test_data.x, depth=d)
                test_pred, _ = test_data.process_predictions(test_pred_raw)
                test_m = float(np.nanmedian(compute_mase(test_data, test_pred, scale)))
                rows.append({"depth": d, "beta": bet, "train_mase": train_m, "test_mase": test_m})
            del tree
            gc.collect()
        df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}", flush=True)

    # Match the R original style: facet by penalty level, x=depth, train/test
    # as two colored lines per facet, with horizontal min-value reference lines.
    fig, axes = plt.subplots(1, len(BETA_VALUES), figsize=(15, 4.0),
                              sharey=True, sharex=True)
    train_min = float(df["train_mase"].min())
    test_min = float(df["test_mase"].min())
    color_train = "#D55E00"  # darker orange / vermillion
    color_test = "#0072B2"   # darker blue
    for ax, bet in zip(axes, BETA_VALUES):
        sub = df[df["beta"] == bet].sort_values("depth")
        ax.plot(sub["depth"], sub["train_mase"], color=color_train,
                marker="o", markersize=7, linewidth=2.6,
                linestyle=(0, (5, 2)),  # long dash, short gap
                label="train")
        ax.plot(sub["depth"], sub["test_mase"], color=color_test,
                marker="s", markersize=7, linewidth=2.6, linestyle="-",
                label="test")
        ax.axhline(train_min, color=color_train, linewidth=1.0, linestyle=":",
                    alpha=0.8)
        ax.axhline(test_min, color=color_test, linewidth=1.0, linestyle=":",
                    alpha=0.8)
        ax.set_title(rf"$\lambda_\mathcal{{H}} = {bet:g}$", fontsize=12)
        ax.set_xlabel("depth")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("Median MASE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=11)
    fig.suptitle("")  # suppress default
    fig.tight_layout()
    out_pdf = REPO.parent / "submission" / "figures" / "mase_seasonal.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
