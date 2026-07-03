"""Per-horizon slope sign test (response letter, R1.1/R2.6; Limitations paragraph).

Reported claim: the per-dataset OLS slopes of the joint-regularized variant's
relative-to-base MASE against the forecast step are balanced (12 negative and
12 positive among the n=24 seasonal datasets) and a binomial two-sided sign
test does not reject the null of no horizon trend (p=1.0).

Recipe (verbatim from the internal receipt of 2026-07-02):
  Per dataset: merge cv_perh/<ds>_base_perh.csv and <ds>_both_perh.csv
  on h, drop NaN rows (Rosmann Daily and Rosmann Daily with Cov have NaN
  median cells at some horizons), rel_h = median_mase_both / median_mase_base,
  OLS slope of rel_h on h, sign; binomial two-sided test on the sign counts
  (scipy.stats.binomtest).

Dataset basis: n=24. The four S=1 (non-seasonal) datasets are excluded:
  - Mackey-glass and Chaotic logistic have no *_both_perh.csv artifact at all
    (no distinct joint variant was fit), and
  - M1 Yearly and Tourism Yearly have an inert heterogeneity penalty (S=1),
    so their "both" model is numerically the deriv model; including them
    would count deriv-only slopes in a joint-variant test.
Pass --include-s1 to reproduce the n=26 all-available-files variant
(13 negative / 13 positive, p=1.0; conclusion identical).

Inputs:  the per-horizon MASE artifacts written by the benchmark
         pass (see experiments/refit_all.py), one pair of files per
         dataset: <ds>_base_perh.csv and <ds>_both_perh.csv with columns
         (h, mean_mase, median_mase, dataset, variant, ...).
Outputs: <out>/slope_signs.csv  (dataset, n_horizons, slope, sign)
         and the binomial test result on stdout.

Usage:
    python experiments/slope_sign_test.py --perh-dir <dir-with-*_perh.csv> \
        [--out evidence] [--include-s1]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# S=1 datasets (seasonal period 1): heterogeneity penalty inert, "both" is
# numerically "deriv"; excluded from the joint-variant slope test (n=24).
S1_DATASETS = {"M1_Yearly", "Tourism_Yearly", "Mackey-glass", "Chaotic_logistic"}


def dataset_slope(base_csv: str, both_csv: str) -> tuple[float, int]:
    """OLS slope of rel_h = median_mase_both / median_mase_base against h."""
    base = pd.read_csv(base_csv)[["h", "median_mase"]].rename(
        columns={"median_mase": "base"}
    )
    both = pd.read_csv(both_csv)[["h", "median_mase"]].rename(
        columns={"median_mase": "both"}
    )
    merged = base.merge(both, on="h").dropna()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = merged["both"].to_numpy() / merged["base"].to_numpy()
    h = merged["h"].to_numpy(dtype=float)
    # Drop non-finite relative cells (e.g. Rosmann Daily has a zero base
    # median at h=1, so the ratio is undefined there).
    ok = np.isfinite(rel)
    slope = float(np.polyfit(h[ok], rel[ok], 1)[0])
    return slope, int(ok.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--perh-dir",
        required=True,
        help="Directory holding <ds>_base_perh.csv / <ds>_both_perh.csv pairs",
    )
    ap.add_argument("--out", default="evidence", help="Output directory")
    ap.add_argument(
        "--include-s1",
        action="store_true",
        help="Also count the S=1 datasets that have a both-file (n=26 basis)",
    )
    args = ap.parse_args()

    rows = []
    for base_csv in sorted(glob.glob(os.path.join(args.perh_dir, "*_base_perh.csv"))):
        ds = os.path.basename(base_csv)[: -len("_base_perh.csv")]
        both_csv = os.path.join(args.perh_dir, f"{ds}_both_perh.csv")
        if not os.path.exists(both_csv):
            print(f"  [skip] {ds}: no joint-variant per-horizon file")
            continue
        if not args.include_s1 and ds in S1_DATASETS:
            print(f"  [skip] {ds}: S=1 dataset (inert heterogeneity penalty)")
            continue
        slope, n_h = dataset_slope(base_csv, both_csv)
        rows.append(
            {
                "dataset": ds,
                "n_horizons": n_h,
                "slope": slope,
                "sign": "neg" if slope < 0 else "pos",
            }
        )

    df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    n_neg = int((df["slope"] < 0).sum())
    n_pos = int((df["slope"] >= 0).sum())
    test = binomtest(n_neg, n_neg + n_pos, p=0.5, alternative="two-sided")

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "slope_signs.csv")
    df.to_csv(out_csv, index=False, float_format="%.6g")

    print(df.to_string(index=False))
    print()
    print(
        f"n={len(df)} datasets: {n_neg} negative / {n_pos} positive slopes; "
        f"binomial two-sided p={test.pvalue:.4g}"
    )
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
