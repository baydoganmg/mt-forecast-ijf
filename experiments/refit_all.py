"""Canonical regeneration: one run that stores every artifact behind every
number in the paper, on finalized main (afcebe7). Reproducibility source of
truth and the harness for the code release.

For each (dataset, variant in base/deriv/seas/both) x family (mDT/mGBT/mRF),
fit at the cached CV pick and persist:
  results/canonical/forecasts/{ds}_{variant}_{family}.csv   per-series HxN preds
  results/canonical/actuals/{ds}_{variant}.csv              per-series actuals
  results/canonical/summary.csv   row: scalar MASE/msMAPE/RMSE (mean+median),
                                     pick, fit_s, best_iter
  results/canonical/per_horizon.csv  row per (ds,variant,family,h)

Predictions are stored so ANY metric is recomputable later with no rerun.
Resumable: (ds,variant,family) already in summary.csv are skipped.
Reuses the exact cross-validation pipeline (CV picks, data build) for faithfulness.
"""
from __future__ import annotations
import gc, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO); sys.path.insert(0, str(REPO))
os.environ.setdefault("MT_BACKEND", "cpp"); os.environ.setdefault("MT_DTYPE", "fp32")

from mttrees.tree import CTree_MT
from mttrees.ensemble import BDT, RF
from mttrees.utils import row_wise_mase, msmape
from experiments.run_benchmark import _safe_aggregate
from experiments.cv_variants import (
    _build_data, _load_or_run_cv, _pick_variant, VARIANTS, DATASETS, CFG, INFO_TABLE)

OUT = REPO / "results" / "canonical"
(OUT / "forecasts").mkdir(parents=True, exist_ok=True)
(OUT / "actuals").mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "summary.csv"
PERH = OUT / "per_horizon.csv"
SEED = CFG.get("seed", 10)


def _p():
    return dict(n_level=CFG["n_discrete_lev"], min_leaf=CFG["min_samples_leaf"],
                prebin=CFG.get("prebin", False),
                agg=CFG.get("aggregation", "score").lower(),
                nthreads=CFG.get("nthreads", 1))


def fit_capture(family, train, test, pick, ss):
    """Fit one family at pick. mDT uses pick depth; ensembles use config depth.
    Returns (pred NxH orig-scale, fit_seconds, best_iter, depth_used)."""
    p = _p(); decay = pick["decay"]
    depth = pick["depth"] if family == "mDT" else (
        CFG["bdt"]["max_depth"] if family == "mGBT" else CFG["rf"]["max_depth"])
    weights = [1.0, pick["lam_w"], pick["bet_w"]]
    t0 = time.time(); best_iter = -1
    if family == "mDT":
        m = CTree_MT(max_depth=depth, verbose=0, min_samples_leaf=p["min_leaf"],
                     n_discrete_lev=p["n_level"], num_threads=p["nthreads"],
                     prebin=p["prebin"], aggregation=p["agg"])
        m.arrangeObjective(train, decay, weights)
        m.fit(train.x, train.y, random_state=SEED)
        raw = m.predict(test.x)
    elif family == "mGBT":
        kw = dict(random_state=SEED, n_estimators_=CFG["bdt"]["n_estimators"],
                  lr=CFG["bdt"]["lr"], bagging_fraction=CFG["bdt"]["bagging_fraction"],
                  feature_fraction=CFG["bdt"]["feature_fraction"], max_depth=CFG["bdt"]["max_depth"],
                  min_samples_leaf=p["min_leaf"], n_discrete_lev=p["n_level"],
                  num_threads=CFG["bdt"].get("num_threads", p["nthreads"]),
                  prebin=p["prebin"], aggregation=p["agg"])
        for k in ("early_stopping_rounds", "min_delta", "eval_k", "eval_metric"):
            if k in CFG.get("bdt", {}): kw[k] = CFG["bdt"][k]
        m = BDT(train, lambda_decay=decay, objective_weights=weights,
                eval_seasonal_scale=ss, **kw)
        m.fit(); raw = m.predict(test.x)
        best_iter = getattr(m, "best_iteration_", -1) or -1
    else:  # mRF
        m = RF(train, lambda_decay=decay, objective_weights=weights,
               random_state=SEED, n_estimators_=CFG["rf"]["n_estimators"],
               bagging_fraction=CFG["rf"]["bagging_fraction"],
               feature_fraction=CFG["rf"]["feature_fraction"], max_depth=CFG["rf"]["max_depth"],
               min_samples_leaf=p["min_leaf"], n_discrete_lev=p["n_level"],
               num_threads=CFG["rf"].get("num_threads", p["nthreads"]),
               prebin=p["prebin"], aggregation=p["agg"],
               n_jobs=CFG["rf"].get("n_jobs", 1),
               backend=CFG["rf"].get("backend", "thread"))
        m.fit(); raw = m.predict(test.x)
    pred, _ = test.process_predictions(raw)
    fit_s = time.time() - t0
    del m; gc.collect()
    return np.asarray(pred), fit_s, int(best_iter), depth


def metrics(actual, pred, scl):
    """Return scalar dict + per-horizon arrays. actual/pred: (N,H); scl: (N,)."""
    mase = row_wise_mase(actual, pred, scl)            # (N,) per-series mean over H
    mae_h = np.abs(actual - pred)                       # (N,H)
    mase_h = mae_h / scl[:, None]                        # (N,H)
    rmse = np.sqrt(np.nanmean((actual - pred) ** 2, axis=1))  # (N,)
    rmse_h = np.abs(actual - pred)                       # store |err|; sq+sqrt at agg
    sm = msmape(actual, pred)                            # (N,) per-series
    sm_h = 200.0 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-9)  # (N,H) %
    H = actual.shape[1]
    perh = []
    for h in range(H):
        _mh = mase_h[:, h]; _mh = _mh[np.isfinite(_mh)]
        _sh = sm_h[:, h]; _sh = _sh[np.isfinite(_sh)]
        perh.append(dict(h=h + 1,
                         mase_mean=float(np.mean(_mh)) if _mh.size else float("nan"),
                         mase_med=float(np.median(_mh)) if _mh.size else float("nan"),
                         msmape_mean=float(np.mean(_sh)) if _sh.size else float("nan"),
                         msmape_med=float(np.median(_sh)) if _sh.size else float("nan"),
                         rmse_mean=float(np.sqrt(np.nanmean((actual[:, h] - pred[:, h]) ** 2))),
                         ))
    # Monash-standard aggregation: drop inf/nan per-series then mean+median
    # (matches _fit_three_methods / the paper). Plain nanmedian keeps inf rows
    # from degenerate-scale series (e.g. COVID Deaths) and gives wrong numbers.
    mase_mean, mase_med = _safe_aggregate(mase)
    sm_mean, sm_med = _safe_aggregate(sm)
    rmse_mean, rmse_med = _safe_aggregate(rmse)
    scal = dict(mase_mean=mase_mean, mase_med=mase_med,
                msmape_mean=sm_mean, msmape_med=sm_med,
                rmse_mean=rmse_mean, rmse_med=rmse_med)
    return scal, perh


def done_set():
    if not SUMMARY.exists(): return set()
    d = pd.read_csv(SUMMARY)
    return set(zip(d.dataset, d.variant, d.family))


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not Path(path).exists())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    todo = args.datasets or DATASETS
    done = done_set()
    for ds in todo:
        if ds not in INFO_TABLE.index:
            print(f"SKIP {ds}: not in info table", flush=True); continue
        pend_v = [v for v in VARIANTS if any((ds, v, f) not in done
                  for f in ("mDT", "mGBT", "mRF"))]
        if not pend_v:
            print(f"SKIP {ds}: all done", flush=True); continue
        print(f"\n=== {ds} ===", flush=True)
        t0 = time.time()
        train, test, ss, freq = _build_data(ds)
        agg = _load_or_run_cv(ds, train, ss, CFG)
        series = test.index["series"].to_numpy()
        actual = test.y_org
        if hasattr(actual, "to_numpy"): actual = actual.to_numpy()
        actual = np.asarray(actual)
        scl = ss.loc[series].to_numpy()
        H = actual.shape[1]
        hcols = [f"h{j+1}" for j in range(H)]
        for v in pend_v:
            pick = _pick_variant(agg, VARIANTS[v])
            if pick is None:
                print(f"  {v}: no CV cell; skip", flush=True); continue
            actpath = OUT / "actuals" / f"{ds.replace(' ','_')}_{v}.csv"
            if not actpath.exists():
                pd.DataFrame(actual, columns=hcols).assign(series=series).to_csv(actpath, index=False)
            for fam in ("mDT", "mGBT", "mRF"):
                if (ds, v, fam) in done: continue
                pred, fit_s, bi, depth_used = fit_capture(fam, train, test, pick, ss)
                pd.DataFrame(pred, columns=hcols).assign(series=series).to_csv(
                    OUT / "forecasts" / f"{ds.replace(' ','_')}_{v}_{fam}.csv", index=False)
                scal, perh = metrics(actual, pred, scl)
                append(SUMMARY, [dict(dataset=ds, variant=v, family=fam,
                       depth=depth_used, decay=pick["decay"],
                       lam_w=pick["lam_w"], bet_w=pick["bet_w"],
                       fit_s=round(fit_s, 2), best_iter=bi, **scal)])
                append(PERH, [dict(dataset=ds, variant=v, family=fam, **r) for r in perh])
                print(f"  {v} {fam}: MASE med={scal['mase_med']:.4f} "
                      f"msMAPE med={scal['msmape_med']:.3f} ({fit_s:.0f}s, iter={bi})", flush=True)
        del train, test; gc.collect()
        print(f"  *** {ds} {(time.time()-t0)/60:.1f} min ***", flush=True)
    print("\nCANONICAL REGEN FINISHED", flush=True)


if __name__ == "__main__":
    main()
