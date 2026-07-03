"""Recompute the canonical summary + per_horizon from STORED forecasts using the
Monash-standard aggregation (_safe_aggregate: drop inf/nan per-series, then
mean+median). The canonical run's predictions are correct; only its scalar
aggregation used a plain median that fails to drop degenerate-scale series
(e.g. COVID Deaths). No refitting; reads forecasts/ + actuals/ only.

Outputs: results/summary_fixed.csv, per_horizon_fixed.csv
"""
import os, sys
os.environ.setdefault("MT_BACKEND","cpp"); os.environ.setdefault("MT_DTYPE","fp32")
sys.path.insert(0,"."); import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from mttrees.utils import organize_repo_data, compute_seasonal_scale, row_wise_mase, msmape
from experiments.run_benchmark import _safe_aggregate
from experiments.cv_variants import INFO_TABLE, CFG

OUT = Path("results/canonical")
FC = OUT / "forecasts"

def finite_mean_med(arr):
    a = np.asarray(arr, dtype=float); a = a[np.isfinite(a)]
    if a.size == 0: return float("nan"), float("nan")
    return float(np.mean(a)), float(np.median(a))

# seasonal scale cache per dataset
_ss = {}
def get_ss(ds):
    if ds in _ss: return _ss[ds]
    info = INFO_TABLE.loc[ds]
    combined, freq, seas, h, ext = organize_repo_data(str(Path(CFG["data_path"])/info["file_name"]))
    H = int(info["horizon"]) if not pd.isna(info["horizon"]) else h
    ss = compute_seasonal_scale(combined, "y", H, int(max(seas)))
    _ss[ds] = ss; return ss

old = pd.read_csv(OUT/"summary.csv")
keys = old[["dataset","variant","family"]].drop_duplicates().values.tolist()
srows, prows = [], []
for ds, v, fam in keys:
    f = FC / f"{ds.replace(' ','_')}_{v}_{fam}.csv"
    a = OUT / "actuals" / f"{ds.replace(' ','_')}_{v}.csv"
    if not f.exists() or not a.exists(): continue
    fc = pd.read_csv(f); ac = pd.read_csv(a)
    hc = [c for c in fc.columns if c.startswith("h")]
    pred = fc[hc].to_numpy(); actual = ac[hc].to_numpy()
    series = fc["series"].to_numpy()
    ss = get_ss(ds); scl = ss.loc[series].to_numpy()
    mase = row_wise_mase(actual, pred, scl)
    sm = msmape(actual, pred)
    rmse = np.sqrt(np.nanmean((actual-pred)**2, axis=1))
    mase_mean, mase_med = _safe_aggregate(mase)
    sm_mean, sm_med = _safe_aggregate(sm)
    rmse_mean, rmse_med = _safe_aggregate(rmse)
    orow = old[(old.dataset==ds)&(old.variant==v)&(old.family==fam)].iloc[0]
    srows.append(dict(dataset=ds,variant=v,family=fam,depth=orow.depth,decay=orow.decay,
        lam_w=orow.lam_w,bet_w=orow.bet_w,fit_s=orow.fit_s,best_iter=orow.best_iter,
        mase_mean=mase_mean,mase_med=mase_med,msmape_mean=sm_mean,msmape_med=sm_med,
        rmse_mean=rmse_mean,rmse_med=rmse_med))
    mase_h = np.abs(actual-pred)/scl[:,None]
    sm_h = 200.0*np.abs(actual-pred)/(np.abs(actual)+np.abs(pred)+1e-9)
    for h in range(len(hc)):
        mm,mmd = finite_mean_med(mase_h[:,h]); smm,smmd = finite_mean_med(sm_h[:,h])
        rh = np.sqrt(np.nanmean((actual[:,h]-pred[:,h])**2))
        prows.append(dict(dataset=ds,variant=v,family=fam,h=h+1,mase_mean=mm,mase_med=mmd,
            msmape_mean=smm,msmape_med=smmd,rmse_mean=rh))
pd.DataFrame(srows).to_csv(OUT/"summary_fixed.csv",index=False)
pd.DataFrame(prows).to_csv(OUT/"per_horizon_fixed.csv",index=False)
print(f"recomputed {len(srows)} summary rows -> summary_fixed.csv")
