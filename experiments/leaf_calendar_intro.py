#!/usr/bin/env python3
"""Intro Figure: leaf-by-month composition heatmap, polished for the introduction.
Same generator + fits as leaf_calendar_candidates.py (seed 2026, depth 6)."""
import os
os.environ.setdefault("MT_BACKEND", "cpp"); os.environ.setdefault("MT_DTYPE", "fp32")
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/research/forecasting/mt-forecast"))
from mttrees import DataMt, CTree_MT

rng = np.random.default_rng(2026)
D, YEARS, BUMP = 365, 3, 12.0
N = D * YEARS
doy = np.arange(N) % D + 1; year = np.arange(N) // D
dec = (doy >= 335).astype(float)
temp_season = 15.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
w = np.zeros(N)
for t in range(1, N): w[t] = 0.8 * w[t - 1] + rng.normal(0, 2.2)
mild = ((year == 2) & (dec == 1)).astype(float)
temp = temp_season + w + 9.0 * mild
load = 100.0 - 2.0 * temp + BUMP * dec + rng.normal(0, 1.5, N)
month = ((doy - 1) // 30.44).astype(int).clip(0, 11)
feats = pd.DataFrame({"index": np.arange(N), "series": "T1", "temp": temp,
                      "sin_season": np.sin(2 * np.pi * doy / 365.0),
                      "cos_season": np.cos(2 * np.pi * doy / 365.0)})
tgts = pd.DataFrame({"index": np.arange(N), "series": "T1", "ahead_0": load})
test_mask = (year == 2) & (dec == 1); train_mask = ~test_mask
sm = pd.DataFrame({"series": ["T1"], "mean_series": [1.0]})
def mk(mask):
    d = DataMt(max_ahead=1, n_derivative=None, penalty_list=["sin_season", "cos_season"],
               series_means=sm, int_convert=False)
    d.transform(feats[mask].reset_index(drop=True), tgts[mask].reset_index(drop=True), drop_penalty=False)
    return d
tr = mk(train_mask)
tr_idx = np.where(train_mask)[0]
leaves = {}
for name, bet in [("base", 0.0), ("seas", 0.5)]:
    tree = CTree_MT(max_depth=6, min_samples_leaf=5, mtry=1.0, samp_frac=1.0,
                    prebin=True, n_discrete_lev=32)
    tree.arrangeObjective(tr, lambda_decay=1.0, objective_weights=[1.0, bet])
    tree.fit(tr.x, tr.y)
    leaves[name] = np.asarray(tree.predict(tr.x, leaf_id=True)).reshape(-1).astype(int)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
titles = {"base": "standard greedy splits", "seas": "with the heterogeneity penalty"}
for ax, name in zip(axes, ("base", "seas")):
    lv = leaves[name]; mo = month[tr_idx]
    tab = pd.crosstab(lv, mo, normalize="index")
    for m in range(12):
        if m not in tab.columns: tab[m] = 0.0
    tab = tab[sorted(tab.columns)]
    tab = tab.iloc[tab.values.argmax(axis=1).argsort()]
    im = ax.imshow(tab.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(12)); ax.set_xticklabels(MON, fontsize=7.5, rotation=90)
    ax.set_yticks([]); ax.set_ylabel("leaf nodes", fontsize=9)
    ax.axvline(10.5, color="#d62728", lw=1.0, alpha=0.7)
    ax.set_title(titles[name], fontsize=10)
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("share of the leaf's training days", fontsize=8); cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.expanduser("~/leaf_calendar_intro.pdf"), bbox_inches="tight")
print("written ~/leaf_calendar_intro.pdf")
