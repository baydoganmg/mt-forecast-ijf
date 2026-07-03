"""Aggregate the synthetic-panel per-seed CSVs into the reported Table
(median-MASE relative-to-mDT_base, 5-seed bootstrap 95% CIs).

This is the aggregation behind Table `table:synth_results` in the manuscript:
for each (method, type) and each generator seed, the relative change is

    rel(seed) = 100 * (median_mase[method] / median_mase[mDT_base] - 1)

computed on the per-(seed, method, type) median MASE written by
experiments/synthetic_panel.py; the point estimate is the mean of rel
across the 5 seeds and the interval is a percentile bootstrap (10,000
resamples) of that mean.

Verified against the shipped evidence pack
(evidence/synthetic_panel_scaling_none/): mDT_both on Type B
-12.1 [-15.9, -9.1], on Type C +45.4 [+26.3, +69.2]; mDT_deriv on Type A
+4.8 [+1.9, +8.5]; ARIMA on Type C -45.8; ETS on Type C -34.9.
(The companion aggregate_synthetic_perseed.py is the archived mean-MASE
aggregator and does not produce the reported table.)

Usage:
    python experiments/aggregate_synthetic.py \
        --input-dir evidence/synthetic_panel_scaling_none
Writes <input-dir>/aggregated_median_rel.csv and prints the table.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

N_BOOT = 10_000
BOOT_SEED = 0


def aggregate(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "per_seed_2*.csv")))
    if not files:
        raise SystemExit(f"no per_seed_2*.csv found under {input_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for t in sorted(df["type"].unique()):
        for m in sorted(df["method"].unique()):
            rel = []
            for s in sorted(df["seed"].unique()):
                sel = df[(df["seed"] == s) & (df["type"] == t)]
                base = sel.loc[sel["method"] == "mDT_base", "median_mase"]
                val = sel.loc[sel["method"] == m, "median_mase"]
                if base.empty or val.empty:
                    continue
                rel.append(100.0 * (val.iloc[0] / base.iloc[0] - 1.0))
            if not rel:
                continue
            rel = np.asarray(rel)
            boots = np.array(
                [
                    rng.choice(rel, size=len(rel), replace=True).mean()
                    for _ in range(N_BOOT)
                ]
            )
            rows.append(
                {
                    "method": m,
                    "type": t,
                    "rel_mean_pct": rel.mean(),
                    "rel_ci_lo_pct": np.percentile(boots, 2.5),
                    "rel_ci_hi_pct": np.percentile(boots, 97.5),
                    "n_seeds": len(rel),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default="evidence/synthetic_panel_scaling_none",
        help="Directory holding per_seed_<SEED>.csv from synthetic_panel.py",
    )
    args = ap.parse_args()

    out = aggregate(args.input_dir)
    out_csv = os.path.join(args.input_dir, "aggregated_median_rel.csv")
    out.to_csv(out_csv, index=False, float_format="%.4f")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(out.round(1).to_string(index=False))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
