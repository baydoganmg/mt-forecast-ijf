"""R2.13: violin plot of per-series MASE distribution across the 28-dataset panel,
one violin per (family, variant). For each (dataset, family, variant) we compute
the per-series MASE on the held-out test horizon using the seasonal-naive scale
from organize_repo_data, then pool ALL series across ALL datasets into one
violin per method. Output CSV alongside so a reviewer can re-aggregate.

Output:
    submission/figures/violin_per_method.pdf  (matplotlib violin plot)
    revision/tables/violin_per_method_long.csv  (long-format: method, dataset, series, mase)
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from mttrees.utils import organize_repo_data

CFG = yaml.safe_load(open(REPO / "configs" / "benchmark.yaml"))
INFO = pd.read_excel(CFG["info_table"], "repo_data").set_index("Dataset")
if "M1 Yearly" not in INFO.index:
    INFO.loc["M1 Yearly"] = {"prefix": "m1_yearly", "file_name": "m1_yearly_dataset.tsf",
                              "horizon": 6, "predetermined_lag": 2,
                              "integer_conversion": np.nan, "run": 1}

FORECASTS = REPO / "results" / "forecasts_benchmark"
OUT_PDF = REPO.parent / "submission" / "figures" / "violin_per_method.pdf"
OUT_CSV = REPO.parent / "revision" / "tables" / "violin_per_method_long.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

VARIANTS = ["dt_base", "dt_deriv", "dt_seas", "dt_both",
            "gbt_base", "gbt_deriv", "gbt_seas", "gbt_both",
            "rf_base",  "rf_deriv",  "rf_seas",  "rf_both"]
FAMILY_OF = {v: v.split("_")[0] for v in VARIANTS}


def seasonal_scale(dataset: str) -> tuple[pd.Series, int]:
    info = INFO.loc[dataset]
    data_file = REPO / CFG["data_path"] / info["file_name"]
    combined, _, seas, h, _ = organize_repo_data(str(data_file))
    horizon = int(info["horizon"]) if not pd.isna(info["horizon"]) else h
    period = int(seas[0]) if seas else 1
    scales = {}
    for series, grp in combined.groupby("series", observed=True):
        y = grp["y"].to_numpy(dtype=float)
        train = y[:-horizon] if len(y) > horizon else y
        if len(train) > period:
            diffs = np.abs(train[period:] - train[:-period])
            scales[series] = float(diffs.mean()) if len(diffs) > 0 else np.nan
        else:
            scales[series] = np.nan
    return pd.Series(scales), horizon


def per_series_mase(forecast_csv: Path, actual_csv: Path,
                    scales: pd.Series, horizon: int,
                    int_convert: bool) -> pd.Series:
    fc = pd.read_csv(forecast_csv)
    ac = pd.read_csv(actual_csv)
    H_cols = [str(h) for h in range(horizon)]
    F = fc[H_cols].to_numpy(dtype=float)
    if int_convert:
        F = np.round(F)
    F = np.clip(F, 0, None)
    A = ac[H_cols].to_numpy(dtype=float)
    sc = scales.reindex(fc["series"].to_numpy()).to_numpy()
    valid = (sc > 0) & np.isfinite(sc)
    mae = np.mean(np.abs(F[valid] - A[valid]), axis=1)
    mase = mae / sc[valid]
    return pd.Series(mase, index=np.asarray(fc["series"])[valid])


def main():
    rows = []
    datasets = sorted({p.stem.split("_benchmark")[0]
                       for p in (FORECASTS / "dt_base").glob("*_forecast.csv")})
    print(f"Found {len(datasets)} datasets")
    for ds in datasets:
        try:
            scales, horizon = seasonal_scale(ds)
        except Exception as e:
            print(f"  skip {ds}: {e}")
            continue
        info = INFO.loc[ds]
        int_convert = bool(info.get("integer_conversion", False)) if not pd.isna(info.get("integer_conversion")) else False
        for v in VARIANTS:
            fc_path = FORECASTS / v / f"{ds}_benchmark_forecast.csv"
            ac_path = FORECASTS / v / f"{ds}_benchmark_actual.csv"
            if not (fc_path.exists() and ac_path.exists()):
                continue
            try:
                m = per_series_mase(fc_path, ac_path, scales, horizon, int_convert)
            except Exception as e:
                print(f"  skip {ds}/{v}: {e}")
                continue
            for s, val in m.items():
                if np.isfinite(val):
                    rows.append({"method": v, "family": FAMILY_OF[v], "dataset": ds,
                                  "series": s, "mase": float(val)})
        print(f"  {ds}: {sum(1 for r in rows if r['dataset']==ds)} rows")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows, {df['method'].nunique()} methods)")

    # Violin plot — one violin per method, grouped/coloured by family
    fig, ax = plt.subplots(figsize=(14, 5.0))
    fam_color = {"dt": "#888888", "gbt": "#D55E00", "rf": "#0072B2"}
    positions = np.arange(len(VARIANTS))
    data = [df[df["method"] == v]["mase"].clip(upper=df["mase"].quantile(0.99)).values
            for v in VARIANTS]
    parts = ax.violinplot(data, positions=positions, widths=0.85, showmedians=True,
                          showextrema=False)
    for pc, v in zip(parts["bodies"], VARIANTS):
        pc.set_facecolor(fam_color[FAMILY_OF[v]])
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels([v.replace("_", "\n") for v in VARIANTS], fontsize=10)
    ax.set_ylabel("Per-series MASE (clipped at 99th percentile)")
    ax.set_title("Distribution of per-series MASE across the 28-dataset panel for each of the 12 multi-target variants")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")
    # Family separators
    for x in [3.5, 7.5]:
        ax.axvline(x, color="gray", linewidth=0.7, alpha=0.4)
    ax.text(1.5, ax.get_ylim()[1] * 0.97, "mDT", ha="center", fontsize=11, color=fam_color["dt"], fontweight="bold")
    ax.text(5.5, ax.get_ylim()[1] * 0.97, "mGBT", ha="center", fontsize=11, color=fam_color["gbt"], fontweight="bold")
    ax.text(9.5, ax.get_ylim()[1] * 0.97, "mRF", ha="center", fontsize=11, color=fam_color["rf"], fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
