"""Regenerate Figures 3 (decay-end sweep) and 4 (derivative-penalty sweep)
on the Mackey-Glass synthetic dataset, in median MASE, matching the
visual style of Figure 5 (regen_heterogeneity_sweep.py).

Output:
    submission/figures/mase_decay.pdf
    submission/figures/mase_derivative.pdf

Cache:
    results/fig23_mackey_mase/mackey_decay.csv
    results/fig23_mackey_mase/mackey_derivative.csv

Re-running with cached CSVs only redraws; deleting the CSV forces re-fit.
"""
from __future__ import annotations
import gc
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("MT_DTYPE", "fp32")

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
DATASET = "Mackey-glass"
DEPTHS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
DEEP_DEPTH = max(DEPTHS)
PARAM_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CACHE_DIR = REPO / "results" / "fig23_mackey_mase"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COLOR_TRAIN = "#D55E00"  # vermillion
COLOR_TEST = "#0072B2"   # blue


def setup():
    info = INFO.loc[DATASET]
    data_file = REPO / CFG["data_path"] / info["file_name"]
    combined, freq, seas, h, ext = organize_repo_data(str(data_file))
    horizon = int(info["horizon"]) if not pd.isna(info["horizon"]) else h
    lag = int(info["predetermined_lag"]) if not pd.isna(info["predetermined_lag"]) else int(np.ceil(max(seas) * 1.25))
    max_S = int(max(seas))

    lag_g, tar_g, _ = build_features_and_targets(
        combined, "y", lag, horizon, False,
        ["index", "series"], ["season_index"], ext, freq, max_S)
    grp = lag_g.groupby("series").cumcount(ascending=False)
    trf = lag_g[grp >= horizon]; tef = lag_g[grp < horizon]
    grp = tar_g.groupby("series").cumcount(ascending=False)
    trt = tar_g[grp >= horizon]; tet = tar_g[grp < horizon]
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
    grp = s.groupby("series").cumcount(ascending=False)
    s_train = s[grp >= horizon].copy()
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


def compute_msmape(data, predictions, eps=0.5, last_horizon_only=False):
    A = data.y_org
    F = predictions
    if F.ndim == 1:
        F = F.reshape(-1, 1); A = A.reshape(-1, 1)
    if last_horizon_only:
        A = A[:, -1:]
        F = F[:, -1:]
    denom = np.maximum(np.abs(A) + np.abs(F) + eps, eps) / 2.0
    return 100.0 * np.mean(np.abs(F - A) / denom, axis=1)


def run_sweep(sweep_kind: str, cache_path: Path) -> pd.DataFrame:
    """sweep_kind in {'decay', 'derivative'}."""
    if cache_path.exists():
        print(f"  cached: {cache_path}", flush=True)
        return pd.read_csv(cache_path)
    print(f"  fitting {DATASET} {sweep_kind} sweep", flush=True)
    train_data, test_data, scale = setup()
    rows = []
    for v in PARAM_VALUES:
        if sweep_kind == "decay":
            decay, lam, bet = v, 0.0, 0.0
        elif sweep_kind == "derivative":
            decay, lam, bet = 1.0, v, 0.0
        else:
            raise ValueError(sweep_kind)
        tree = CTree_MT(max_depth=DEEP_DEPTH, verbose=0,
                         min_samples_leaf=CFG["min_samples_leaf"],
                         n_discrete_lev=CFG["n_discrete_lev"], num_threads=2,
                         prebin=CFG.get("prebin", True))
        tree.arrangeObjective(train_data, decay, [1.0, lam, bet])
        tree.fit(train_data.x, train_data.y)
        print(f"    {sweep_kind}={v:.1f} fit done", flush=True)
        for d in DEPTHS:
            tr_pred = train_data.process_predictions(tree.predict(train_data.x, depth=d))[0]
            te_pred = test_data.process_predictions(tree.predict(test_data.x, depth=d))[0]
            rows.append({
                "depth": d, "param": v,
                "train_mase": float(np.nanmedian(compute_mase(train_data, tr_pred, scale))),
                "test_mase": float(np.nanmedian(compute_mase(test_data, te_pred, scale))),
                "train_msmape": float(np.nanmedian(compute_msmape(train_data, tr_pred))),
                "test_msmape": float(np.nanmedian(compute_msmape(test_data, te_pred))),
                "train_msmape_lastH": float(np.nanmedian(compute_msmape(train_data, tr_pred, last_horizon_only=True))),
                "test_msmape_lastH":  float(np.nanmedian(compute_msmape(test_data, te_pred, last_horizon_only=True))),
            })
        del tree
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"  wrote {cache_path}", flush=True)
    return df


def plot_sweep(df: pd.DataFrame, param_label_tex: str, out_pdf: Path, metric: str):
    """metric in {'mase', 'msmape'} — uses train_<metric>/test_<metric> columns."""
    fig, axes = plt.subplots(1, len(PARAM_VALUES), figsize=(15, 4.0),
                              sharey=True, sharex=True)
    train_col, test_col = f"train_{metric}", f"test_{metric}"
    train_min = float(df[train_col].min())
    test_min = float(df[test_col].min())
    for ax, v in zip(axes, PARAM_VALUES):
        sub = df[df["param"] == v].sort_values("depth")
        ax.plot(sub["depth"], sub[train_col], color=COLOR_TRAIN,
                marker="o", markersize=7, linewidth=2.6,
                linestyle=(0, (5, 2)), label="train")
        ax.plot(sub["depth"], sub[test_col], color=COLOR_TEST,
                marker="s", markersize=7, linewidth=2.6, linestyle="-",
                label="test")
        ax.axhline(train_min, color=COLOR_TRAIN, linewidth=1.0, linestyle=":", alpha=0.8)
        ax.axhline(test_min, color=COLOR_TEST, linewidth=1.0, linestyle=":", alpha=0.8)
        ax.set_title(rf"${param_label_tex} = {v:g}$", fontsize=12)
        ax.set_xlabel("depth")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("Median MASE" if metric == "mase" else r"Median msMAPE (\%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_pdf}", flush=True)


def main():
    figdir = REPO.parent / "submission" / "figures"
    # NOTE: outputs use the *_py.pdf suffix so we don't overwrite the
    # R-rendered production files (msmape_horizon.pdf, msmape_derivative.pdf,
    # mase_seasonal_R.pdf) that main.tex includes.
    print("=== Fig 2: decay-end sweep ===")
    df_decay = run_sweep("decay", CACHE_DIR / "mackey_decay.csv")
    plot_sweep(df_decay, r"r", figdir / "mase_decay_py.pdf",            metric="mase")
    plot_sweep(df_decay, r"r", figdir / "msmape_decay_py.pdf",          metric="msmape")
    plot_sweep(df_decay, r"r", figdir / "msmape_decay_lastH_py.pdf",    metric="msmape_lastH")

    print("\n=== Fig 3: derivative-penalty sweep ===")
    df_deriv = run_sweep("derivative", CACHE_DIR / "mackey_derivative.csv")
    plot_sweep(df_deriv, r"\lambda_\mathcal{C}", figdir / "mase_derivative_py.pdf",        metric="mase")
    plot_sweep(df_deriv, r"\lambda_\mathcal{C}", figdir / "msmape_derivative_py.pdf",      metric="msmape")
    plot_sweep(df_deriv, r"\lambda_\mathcal{C}", figdir / "msmape_derivative_lastH_py.pdf", metric="msmape_lastH")


if __name__ == "__main__":
    main()
