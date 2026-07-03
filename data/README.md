# Data

Place the benchmark datasets here before running experiments.

Expected layout:

```
data/
├── repo_data_info.xlsx           # Dataset metadata (name, file, horizon, lag, seasonality, ...)
├── m1_yearly_dataset.tsf         # bundled demo (tiny)
├── mackey_glass_dataset.tsf      # bundled demo (used by the sweep figures)
└── ...                           # all other .tsf files listed in repo_data_info.xlsx
```

All datasets are in the `.tsf` format of the
[Monash Time Series Forecasting Repository](https://forecastingdata.org/)
(Godahewa et al., 2021, "Monash Time Series Forecasting Archive").

To download the full 28-dataset benchmark (~440 MB):

```bash
python scripts/fetch_monash_data.py            # everything
python scripts/fetch_monash_data.py --list     # show URLs only
python scripts/fetch_monash_data.py --dataset hospital
```

The two bundled demo files are skipped by the fetch script. The `data_path`
and `info_table` keys in `configs/benchmark.yaml` point to
this directory.
