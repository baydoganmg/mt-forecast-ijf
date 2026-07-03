"""Fetch the Monash benchmark datasets used in the paper.

Downloads the 28 .tsf datasets cited in `data/repo_data_info.xlsx` from
the Monash Time Series Forecasting Repository (https://forecastingdata.org/)
into `data/`. The two demo datasets bundled with the repo
(`m1_yearly_dataset.tsf`, `mackey_glass_dataset.tsf`) are skipped.

Usage:
    python scripts/fetch_monash_data.py
    python scripts/fetch_monash_data.py --list             # show URLs only
    python scripts/fetch_monash_data.py --dataset hospital # one dataset

Notes
-----
- File sizes range from a few KB (M1 Yearly) to several hundred MB
  (Kaggle Daily, M5). Total download is ~440 MB.
- Files are placed under `data/` next to the demo `.tsf` files. The
  .gitignore excludes them from version control.
- Cite Godahewa et al. (2021), "Monash Time Series Forecasting Archive",
  in addition to the mt-forecast paper when you use these datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# Monash repository points individual datasets to Zenodo or to their own
# release page. The mapping below is a best-effort registry; if a URL has
# moved, please open an issue.  We use the {dataset_label: ([urls], filename)}
# pattern so we can try a primary plus a fallback.
#
# DOIs / Zenodo IDs for the Monash repo: see https://forecastingdata.org/.
# We list the dataset SLUG → expected filename in data/ and the canonical
# download URL.  Add a SHA256 if you want byte-level integrity checks.

DATASETS = {
    # name                                # filename                                            # Zenodo ID
    "M1 Yearly":                          ("m1_yearly_dataset.tsf",                              "4656193"),
    "M1 Quarterly":                       ("m1_quarterly_dataset.tsf",                           "4656154"),
    "M1 Monthly":                         ("m1_monthly_dataset.tsf",                             "4656159"),
    "M3 Yearly":                          ("m3_yearly_dataset.tsf",                              "4656222"),
    "M3 Quarterly":                       ("m3_quarterly_dataset.tsf",                           "4656262"),
    "M3 Monthly":                         ("m3_monthly_dataset.tsf",                             "4656298"),
    "M4 Yearly":                          ("m4_yearly_dataset.tsf",                              "4656379"),
    "M4 Quarterly":                       ("m4_quarterly_dataset.tsf",                           "4656410"),
    "M4 Monthly":                         ("m4_monthly_dataset.tsf",                             "4656480"),
    "M4 Weekly":                          ("m4_weekly_dataset.tsf",                              "4656522"),
    "M4 Daily":                           ("m4_daily_dataset.tsf",                               "4656548"),
    "M4 Hourly":                          ("m4_hourly_dataset.tsf",                              "4656589"),
    "Tourism Yearly":                     ("tourism_yearly_dataset.tsf",                         "4656103"),
    "Tourism Quarterly":                  ("tourism_quarterly_dataset.tsf",                      "4656093"),
    "Tourism Monthly":                    ("tourism_monthly_dataset.tsf",                        "4656096"),
    "NN5 Daily":                          ("nn5_daily_dataset_without_missing_values.tsf",       "4656117"),
    "NN5 Weekly":                         ("nn5_weekly_dataset.tsf",                             "4656125"),
    "Hospital":                           ("hospital_dataset.tsf",                               "4656014"),
    "Bitcoin":                            ("bitcoin_dataset.tsf",                                "5121965"),
    "Electricity Weekly":                 ("electricity_weekly_dataset.tsf",                     "4656141"),
    "FRED-MD":                            ("fred_md_dataset.tsf",                                "4654833"),
    "Traffic Weekly":                     ("traffic_weekly_dataset.tsf",                         "4656135"),
    "Vehicle Trips":                      ("vehicle_trips_dataset.tsf",                          "5122537"),
    "COVID Deaths":                       ("covid_deaths_dataset.tsf",                           "4656009"),
    "M5":                                 ("m5_dataset.tsf",                                     "4656636"),
    "Chaotic Logistic":                   ("chaotic_logistic_dataset.tsf",                       "4656173"),
    "Mackey-Glass":                       ("mackey_glass_dataset.tsf",                           "4656186"),
    # Kaggle / Favorita / Rossmann are SETAR-paper releases on the Monash repo
    # mirror; see Godahewa et al. (2023) "SETAR-Tree".
    "Kaggle Daily":                       ("kaggle_web_traffic_1000_dataset.tsf",                "10518419"),
    "Favorita Daily":                     ("favourita_sales_1000_dataset.tsf",                   "10518419"),
    "Rossmann Daily":                     ("rossmann_dataset_without_missing_values.tsf",        "10518419"),
}

ZENODO_BASE = "https://zenodo.org/records/{zid}/files/{fname}?download=1"

DEMO_BUNDLED = {"m1_yearly_dataset.tsf", "mackey_glass_dataset.tsf"}


def _download(url: str, dest: Path, chunk: int = 1 << 16) -> int:
    """Stream-download `url` to `dest`. Returns bytes written."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_total = 0
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            bytes_total += len(buf)
    tmp.replace(dest)
    return bytes_total


def fetch_all(only: str | None = None, list_only: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    selected = DATASETS
    if only:
        selected = {k: v for k, v in DATASETS.items() if k.lower() == only.lower()}
        if not selected:
            sys.exit(f"unknown dataset: {only!r}")
    print(f"Target directory: {DATA}\n")
    for label, (fname, zid) in selected.items():
        url = ZENODO_BASE.format(zid=zid, fname=fname)
        dest = DATA / fname
        if list_only:
            print(f"  {label:<22}  {url}")
            continue
        if fname in DEMO_BUNDLED and dest.exists():
            print(f"  [skip] {label:<22}  (already bundled with repo)")
            continue
        if dest.exists():
            size_mb = dest.stat().st_size / (1 << 20)
            print(f"  [ok]   {label:<22}  already present ({size_mb:.1f} MB)")
            continue
        print(f"  [get]  {label:<22}  -> {fname}")
        try:
            t0 = time.time()
            n = _download(url, dest)
            elapsed = time.time() - t0
            print(f"         downloaded {n / (1<<20):.1f} MB in {elapsed:.1f}s")
        except urllib.error.HTTPError as e:
            print(f"         HTTP {e.code}: {e.reason}  ({url})")
        except Exception as e:
            print(f"         failed: {type(e).__name__}: {e}")
            if dest.with_suffix(dest.suffix + ".part").exists():
                dest.with_suffix(dest.suffix + ".part").unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--list", action="store_true",
                   help="print dataset URLs without downloading")
    p.add_argument("--dataset", default=None,
                   help="fetch only this dataset (e.g. 'Hospital')")
    args = p.parse_args()
    fetch_all(only=args.dataset, list_only=args.list)


if __name__ == "__main__":
    main()
