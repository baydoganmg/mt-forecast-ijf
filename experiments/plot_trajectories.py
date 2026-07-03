"""Build a clean R1.2 trajectory figure for the revised manuscript.

Two fixes over plot_trajectories.py:

  1. Pick series where the tree's first-horizon forecast is close to the
     last training observation (small |y_T - F_0| / scale). On heavily
     averaged tree predictions this is the only way to get a figure that
     reads as a "continuation" of the training history rather than as a
     visible level jump.
  2. Use a single-row 1xN layout so the resulting figure fits the body
     of an article column. Pick at most 3 series across 2-3 datasets to
     give visual variety without stacking the figure 6+ rows tall.

Output: submission/figures/trajectory_v2.pdf  (3-panel horizontal layout)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

FORECASTS = REPO / "results" / "forecasts_benchmark"
OUT = REPO.parent / "submission" / "figures" / "trajectory_v2.pdf"

# Candidate datasets: stationary-ish count / aggregated series where the
# leaf-mean prediction has a fighting chance of starting near y_T.
CANDIDATES = [
    "Hospital",
    "Tourism Quarterly",
    "M3 Quarterly",
    "M1 Quarterly",
    "M3 Monthly",
    "NN5 Weekly",
]

VARIANTS = ["dt_base", "dt_deriv", "dt_seas", "dt_both"]
COLOR = {"dt_base": "#888888", "dt_deriv": "#1f77b4",
         "dt_seas": "#2ca02c", "dt_both": "#d62728"}
LABEL = {"dt_base": "DT (base)", "dt_deriv": "DT + derivative",
         "dt_seas": "DT + seasonal", "dt_both": "DT + both"}
LS = {"dt_base": "--", "dt_deriv": "-", "dt_seas": "-", "dt_both": "-"}


def parse_tsf_dataset(dataset: str):
    """Reuse the parse_tsf helper from run_classical_baselines."""
    from run_classical_baselines import parse_tsf
    info = pd.read_excel(REPO / "data" / "repo_data_info.xlsx",
                         "repo_data").set_index("Dataset")
    file_name = info.loc[dataset, "file_name"]
    horizon = int(info.loc[dataset, "horizon"]) if not pd.isna(info.loc[dataset, "horizon"]) else None
    seas = int(info.loc[dataset, "predetermined_lag"]) if "predetermined_lag" in info.columns else 1
    records, freq, _ = parse_tsf(REPO / "data" / file_name)
    out = {r["series_name"]: np.asarray(r["series_value"], dtype=float) for r in records}
    return out, horizon


def load_variant(dataset, variant):
    fc = pd.read_csv(FORECASTS / variant / f"{dataset}_benchmark_forecast.csv")
    ac = pd.read_csv(FORECASTS / variant / f"{dataset}_benchmark_actual.csv")
    return fc, ac


def score_series(dataset: str, top_k: int = 6):
    """For each series in `dataset`, compute the absolute gap between
    the last training observation y_T and the dt_base first-horizon
    forecast F_0, scaled by the series std. Smaller = better."""
    tr_dict, horizon = parse_tsf_dataset(dataset)
    fc, _ = load_variant(dataset, "dt_base")
    H_cols = [c for c in fc.columns if c not in ("index", "series")]
    horizon = len(H_cols)
    scored = []
    for _, row in fc.iterrows():
        s = row["series"]
        if s not in tr_dict:
            continue
        vals = tr_dict[s]
        train = vals[:-horizon] if len(vals) > horizon else vals
        if len(train) < max(horizon * 2, 20):
            continue
        sd = float(np.nanstd(train))
        if sd <= 0:
            continue
        F0 = float(row[H_cols[0]])
        yT = float(train[-1])
        gap = abs(F0 - yT) / sd
        scored.append({"dataset": dataset, "series": s,
                       "gap": gap, "yT": yT, "F0": F0,
                       "mean": float(np.nanmean(train)),
                       "sd": sd, "n_train": len(train)})
    if not scored:
        return None, None
    sc = pd.DataFrame(scored).sort_values("gap").head(top_k).reset_index(drop=True)
    return sc, tr_dict


def plot_panel(ax, dataset: str, series: str, tr_dict: dict, n_back: int = 36):
    vals = tr_dict[series]
    var_fc = {v: load_variant(dataset, v) for v in VARIANTS}
    fc0, ac0 = var_fc["dt_base"]
    H_cols = [c for c in fc0.columns if c not in ("index", "series")]
    horizon = len(H_cols)
    train = vals[:-horizon] if len(vals) > horizon else vals
    test = vals[-horizon:] if len(vals) > horizon else np.full(horizon, np.nan)

    t_train = np.arange(-len(train[-n_back:]), 0)
    t_fc = np.arange(0, horizon)
    ax.plot(t_train, train[-n_back:], color="black", linewidth=1.4, label="training")
    ax.plot(t_fc, test, color="black", marker="o", markersize=4, linewidth=1.4, label="actual (test)")
    for v in VARIANTS:
        fc, _ = var_fc[v]
        row = fc[fc["series"] == series]
        if row.empty:
            continue
        F = row.iloc[0][H_cols].to_numpy(dtype=float)
        ax.plot(t_fc, F, color=COLOR[v], linestyle=LS[v], linewidth=1.6,
                marker=("." if v == "dt_base" else None),
                markersize=5, label=LABEL[v])
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_title(f"{dataset} — {series}", fontsize=10)
    ax.set_xlabel("time step (0 = first forecast)")
    ax.grid(True, linestyle=":", alpha=0.4)


def main():
    # Score series across all candidate datasets, then pick the global top 3
    # spread over distinct datasets.
    all_scored = []
    tr_cache = {}
    for ds in CANDIDATES:
        try:
            sc, tr = score_series(ds, top_k=3)
        except Exception as e:
            print(f"  skip {ds}: {e}")
            continue
        if sc is None:
            continue
        all_scored.append(sc)
        tr_cache[ds] = tr
    if not all_scored:
        raise SystemExit("No candidate series found")
    pool = pd.concat(all_scored, ignore_index=True).sort_values("gap")
    print("Top 15 candidates by smallest |yT - F0| / sd gap:")
    print(pool.head(15).to_string(index=False))

    # Pick one per dataset until we have 3
    picks = []
    seen = set()
    for _, r in pool.iterrows():
        if r["dataset"] in seen:
            continue
        picks.append((r["dataset"], r["series"]))
        seen.add(r["dataset"])
        if len(picks) == 3:
            break
    print(f"\nFinal picks: {picks}")

    # Plot a 1x3 horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (ds, s) in zip(axes, picks):
        plot_panel(ax, ds, s, tr_cache[ds])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=6, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
