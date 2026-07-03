#!/usr/bin/env python3
"""Generate the LaTeX data rows for the per-dataset MASE table (Table 5) and the
msMAPE appendix table (Table 7), plus the wide median-MASE source table consumed
by the MCB figure script. The reported statistic is the median (mase_med /
msmape_med). Reads the shipped per-cell summary in evidence/summary.csv.
"""
from pathlib import Path
import pandas as pd

EVID = Path(__file__).resolve().parent.parent / "evidence"
OUT = Path(__file__).resolve().parent

DATASET_ORDER = [
    "Bitcoin", "COVID Deaths", "Chaotic logistic", "Electricity Weekly",
    "FREDMD", "Favorita Daily", "Favorita Daily with Cov", "Hospital",
    "Kaggle Daily", "Kaggle Daily with Cov", "M1 Monthly", "M1 Quarterly",
    "M1 Yearly", "M3 Monthly", "M3 Quarterly", "M4 Hourly", "M4 Weekly",
    "M5 Daily", "Mackey-Glass", "NN5 Daily", "NN5 Weekly", "Rosmann Daily",
    "Rosmann Daily with Cov", "Tourism Monthly", "Tourism Quarterly",
    "Tourism Yearly", "Traffic Weekly", "Vehicle Trips",
]
FAMILIES = ["mDT", "mGBT", "mRF"]
VARIANTS = ["base", "deriv", "seas", "both"]  # column order in the table


def load(csv, metric):
    """metric in {'mase_med','msmape_med'} -> nested dict[ds][fam][variant].

    Datasets without sub-annual seasonality (Chaotic logistic, Mackey-Glass)
    carry only base+deriv; the seasonality regularizer is inert there, so the
    table convention (matching the manuscript) is seas := base and both :=
    deriv. Those are filled in when missing.
    """
    df = pd.read_csv(csv)
    out = {}
    for _, r in df.iterrows():
        out.setdefault(r.dataset, {}).setdefault(r.family, {})[r.variant] = r[metric]
    for ds in out:
        for fam in out[ds]:
            d = out[ds][fam]
            d.setdefault("seas", d["base"])
            d.setdefault("both", d["deriv"])
    return out


def fmt(v):
    return f"{v:.3f}"


def build_rows(data):
    """Return list of (dataset, latex_row_string) using the median statistic."""
    rows = []
    for ds in DATASET_ORDER:
        cells = []
        for fam in FAMILIES:
            vals = {v: data[ds][fam][v] for v in VARIANTS}
            best_within = min(vals, key=lambda v: vals[v])
            reg = {v: vals[v] for v in ("deriv", "seas", "both")}
            best_reg = min(reg, key=lambda v: reg[v])
            pct = (reg[best_reg] - vals["base"]) / vals["base"] * 100.0
            block = [(r"\textbf{" + fmt(vals[v]) + "}") if v == best_within else fmt(vals[v])
                     for v in VARIANTS]
            block.append(("+" if pct >= 0 else "-") + f"{abs(pct):.1f}\\%")
            cells.append(" & ".join(block))
        rows.append((ds, ds + " & " + " & ".join(cells) + r" \\"))
    return rows


def main():
    mase = load(EVID / "summary.csv", "mase_med")
    sm = load(EVID / "summary.csv", "msmape_med")

    with open(OUT / "table_MASE.tex", "w") as f:
        f.write("% Median MASE data rows for table:MASE; bold = within-family best.\n")
        for _, row in build_rows(mase):
            f.write(row + "\n")

    with open(OUT / "table_msMAPE.tex", "w") as f:
        f.write("% Median msMAPE data rows for table:msMAPE; bold = within-family best.\n")
        for _, row in build_rows(sm):
            f.write(row + "\n")

    # wide median-MASE source table (the tree columns of the MCB figures)
    pref = {"mDT": "dt", "mGBT": "gbt", "mRF": "rf"}
    cols = [f"{pref[fam]}_{v}" for fam in FAMILIES for v in VARIANTS]
    wide = []
    for ds in mase:
        rec = {"dataset": ds}
        for fam in FAMILIES:
            for v in VARIANTS:
                rec[f"{pref[fam]}_{v}"] = mase[ds][fam][v]
        wide.append(rec)
    pd.DataFrame(wide)[["dataset"] + cols].to_csv(EVID / "wide_median_mase.csv", index=False)

    print("Wrote table_MASE.tex, table_msMAPE.tex, and evidence/wide_median_mase.csv")


if __name__ == "__main__":
    main()
