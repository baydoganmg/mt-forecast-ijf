#!/usr/bin/env python3
"""
Aggregate per-seed CSVs from synthetic_panel.py into a 5-seed summary with
bootstrap 95% CIs on the per-(method, type) relative-to-mDT_base change.

Reads:    mt_trees/results/synthetic_panel/<scaling>/per_seed_<SEED>.csv
Writes:   mt_trees/results/synthetic_panel/<scaling>/aggregated.csv
          mt_trees/results/synthetic_panel/<scaling>/table_snippet.tex
          mt_trees/results/synthetic_panel/<scaling>/summary.txt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(2026)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def aggregate(input_dir: str) -> None:
    # Only the per-seed CSVs whose filename ends in an integer seed; this
    # deliberately excludes per_seed_long.csv (a concatenation of all seeds)
    # to avoid double-counting each seed in the bootstrap.
    per_seed_files = sorted(
        f for f in glob.glob(os.path.join(input_dir, "per_seed_*.csv"))
        if os.path.basename(f).removeprefix("per_seed_").removesuffix(".csv").isdigit()
    )
    if not per_seed_files:
        sys.exit(f"no per_seed_<seed>.csv files in {input_dir}")

    frames = [pd.read_csv(f) for f in per_seed_files]
    df = pd.concat(frames, ignore_index=True)
    seeds = sorted(df["seed"].unique())
    print(f"aggregating {len(seeds)} seeds: {seeds} (from {len(per_seed_files)} files)")

    # For each seed: compute relative change vs mDT_base per (type).
    rows = []
    for seed in seeds:
        sub = df[df["seed"] == seed]
        base_means = (
            sub[sub["method"] == "mDT_base"]
            .set_index("type")["mean_mase"]
            .to_dict()
        )
        for _, r in sub.iterrows():
            base = base_means.get(r["type"])
            rel = (r["mean_mase"] - base) / base if base else float("nan")
            rows.append({
                "seed": seed,
                "method": r["method"],
                "type": r["type"],
                "n_series": r["n_series"],
                "mean_mase": r["mean_mase"],
                "median_mase": r["median_mase"],
                "rel_to_mDT_base": rel,
            })
    long = pd.DataFrame(rows)

    # Aggregate: per (method, type) mean MASE across seeds, mean rel-to-base, 95% bootstrap CI.
    rng = np.random.default_rng(2026)
    agg_rows = []
    for (method, typ), grp in long.groupby(["method", "type"]):
        means = grp["mean_mase"].to_numpy()
        rels = grp["rel_to_mDT_base"].to_numpy()
        rel_lo, rel_hi = bootstrap_ci(rels, rng=rng)
        agg_rows.append({
            "method": method,
            "type": typ,
            "n_seeds": len(grp),
            "mean_mase_avg": means.mean(),
            "mean_mase_std": means.std(ddof=1) if len(means) > 1 else 0.0,
            "rel_mean_pct": 100 * rels.mean(),
            "rel_ci_lo_pct": 100 * rel_lo,
            "rel_ci_hi_pct": 100 * rel_hi,
            "n_series": grp["n_series"].iloc[0],
        })
    agg = pd.DataFrame(agg_rows)
    out_csv = os.path.join(input_dir, "aggregated.csv")
    agg.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"wrote {out_csv}")

    # Compact summary
    summary_lines = []
    summary_lines.append(f"seeds: {seeds}")
    summary_lines.append("")
    method_order = [
        "mDT_base", "mDT_deriv", "mDT_seas", "mDT_both",
        "skDT", "skRF", "naive", "snaive", "ets", "arima",
    ]
    type_order = ["A", "B", "C", "D", "ALL"]
    for typ in type_order:
        summary_lines.append(f"--- Type {typ} ---")
        sub = agg[agg["type"] == typ]
        for m in method_order:
            r = sub[sub["method"] == m]
            if r.empty:
                continue
            row = r.iloc[0]
            summary_lines.append(
                f"  {m:12s}  mean MASE = {row['mean_mase_avg']:.4f}  "
                f"rel = {row['rel_mean_pct']:+.2f}%  "
                f"CI [{row['rel_ci_lo_pct']:+.2f}, {row['rel_ci_hi_pct']:+.2f}]"
            )
        summary_lines.append("")
    out_summary = os.path.join(input_dir, "summary.txt")
    with open(out_summary, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"wrote {out_summary}")
    print()
    print("\n".join(summary_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default="/media/baydogan/files/Research/UvA/Paper/mt_trees/results/synthetic_panel/scaling_none",
    )
    args = ap.parse_args()
    aggregate(args.input_dir)


if __name__ == "__main__":
    main()
