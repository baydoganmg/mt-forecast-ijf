#!/usr/bin/env python3
"""Structure-based candidates for the intro example: leaf-calendar visualizations
and statistics from the SAME v3 setup (same generator, same two mDT fits, depth 6)."""
import os
os.environ.setdefault("MT_BACKEND", "cpp"); os.environ.setdefault("MT_DTYPE", "fp32")
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/research/forecasting/mt-forecast"))
from mttrees import DataMt, CTree_MT

rng = np.random.default_rng(2026)
D, YEARS, BUMP = 365, 3, 12.0
N = D * YEARS
doy = np.arange(N) % D + 1
year = np.arange(N) // D
dec = (doy >= 335).astype(float)
temp_season = 15.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
w = np.zeros(N)
for t in range(1, N):
    w[t] = 0.8 * w[t - 1] + rng.normal(0, 2.2)
mild = ((year == 2) & (dec == 1)).astype(float)
temp = temp_season + w + 9.0 * mild
load = 100.0 - 2.0 * temp + BUMP * dec + rng.normal(0, 1.5, N)
month = ((doy - 1) // 30.44).astype(int).clip(0, 11)  # approx month 0..11

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
tr, te = mk(train_mask), mk(test_mask)
tr_idx = np.where(train_mask)[0]; te_idx = np.where(test_mask)[0]

trees, leaves_tr, leaves_te, preds_te = {}, {}, {}, {}
for name, bet in [("base", 0.0), ("seas", 0.5)]:
    tree = CTree_MT(max_depth=6, min_samples_leaf=5, mtry=1.0, samp_frac=1.0,
                    prebin=True, n_discrete_lev=32)
    tree.arrangeObjective(tr, lambda_decay=1.0, objective_weights=[1.0, bet])
    tree.fit(tr.x, tr.y)
    trees[name] = tree
    leaves_tr[name] = np.asarray(tree.predict(tr.x, leaf_id=True)).reshape(-1).astype(int)
    leaves_te[name] = np.asarray(tree.predict(te.x, leaf_id=True)).reshape(-1).astype(int)
    p, _ = te.process_predictions(tree.predict(te.x))
    preds_te[name] = np.asarray(p).reshape(-1)

# introspect tree_info for split-variable inference
ti = trees["base"].tree_info
print("tree_info type:", type(ti), getattr(ti, "shape", None), getattr(ti, "dtype", None))
try:
    arr = np.asarray(ti)
    print("tree_info head:\n", arr[:6])
except Exception as e:
    print("tree_info not arrayable:", e)
print("x columns:", list(tr.x.columns))

stats = {}
for name in ("base", "seas"):
    lv_tr = leaves_tr[name]; mo = month[tr_idx]; dc = dec[tr_idx].astype(bool)
    dfl = pd.DataFrame({"leaf": lv_tr, "month": mo, "dec": dc,
                        "sin": feats["sin_season"].values[tr_idx], "cos": feats["cos_season"].values[tr_idx]})
    g = dfl.groupby("leaf")
    # December purity: best leaf share + concentration of Dec days
    purity = (g["dec"].mean()).max()
    modal_share = dfl[dfl.dec].groupby("leaf").size().max() / dfl.dec.sum()
    months_per_leaf = g["month"].nunique().mean()
    disp = float((g["sin"].var(ddof=0).fillna(0) * g.size() + g["cos"].var(ddof=0).fillna(0) * g.size()).sum() / len(dfl) )
    stats[name] = {"n_leaves": int(dfl.leaf.nunique()),
                   "december_max_leaf_purity": float(purity),
                   "december_modal_leaf_share": float(modal_share),
                   "mean_distinct_months_per_leaf": float(months_per_leaf),
                   "within_leaf_sincos_dispersion": disp,
                   "mean_test_pred": float(preds_te[name].mean())}
truth_mean = float(load[te_idx].mean()); nobump_mean = float((100.0 - 2.0 * temp[te_idx]).mean())
stats["capture"] = {n: (stats[n]["mean_test_pred"] - nobump_mean) / BUMP for n in ("base", "seas")}
stats["truth_mean"] = truth_mean; stats["nobump_mean"] = nobump_mean
print(json.dumps(stats, indent=1))

# anecdote: median-temp mild December day
row = int(np.argmax(np.abs(preds_te["base"] - preds_te["seas"])))
mid = te_idx[row]
anec = {"day": f"Dec {int(doy[mid])-334}, temp {temp[mid]:.1f}C", "truth": float(load[mid])}
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for name in ("base", "seas"):
    lid = leaves_te[name][row]
    mates = month[tr_idx][leaves_tr[name] == lid]
    comp = pd.Series([MON[m] for m in mates]).value_counts().to_dict()
    anec[name] = {"leaf_size": int((leaves_tr[name] == lid).sum()), "composition": comp,
                  "prediction": float(preds_te[name][row])}
print("ANECDOTE:", json.dumps(anec, indent=1))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# ---- Candidate 1: seasonal-circle leaf map ----
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), subplot_kw={"aspect": "equal"})
for ax, name, title in zip(axes, ("base", "seas"),
                           ("$\\lambda_{\\mathcal{H}}=0$ (no penalty)", "$\\lambda_{\\mathcal{H}}=0.5$ (heterogeneity penalty)")):
    lv = leaves_tr[name]
    uniq = pd.unique(lv)
    cmap = cm.get_cmap("tab20", 20)
    color_of = {l: cmap(i % 20) for i, l in enumerate(uniq)}
    ang = 2 * np.pi * doy[tr_idx] / 365.0
    r = 0.78 + 0.09 * year[tr_idx] + rng.normal(0, 0.008, len(tr_idx))
    ax.scatter(r * np.sin(ang), r * np.cos(ang), s=4, c=[color_of[l] for l in lv], linewidths=0)
    for m, lab in enumerate(MON):
        a = 2 * np.pi * (m * 30.44 + 15) / 365.0
        ax.text(1.12 * np.sin(a), 1.12 * np.cos(a), lab, ha="center", va="center", fontsize=7)
    a0, a1 = 2 * np.pi * 335 / 365.0, 2 * np.pi * 365 / 365.0
    th = np.linspace(a0, a1, 40)
    ax.fill(np.concatenate([[0], 1.02 * np.sin(th)]), np.concatenate([[0], 1.02 * np.cos(th)]),
            color="gray", alpha=0.12, zorder=0)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Training days on the seasonal circle, colored by leaf assignment (December sector shaded)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.expanduser("~/december_circle.pdf")); plt.close(fig)

# ---- Candidate 2: leaf x month heatmap ----
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6))
for ax, name, title in zip(axes, ("base", "seas"),
                           ("$\\lambda_{\\mathcal{H}}=0$", "$\\lambda_{\\mathcal{H}}=0.5$")):
    lv = leaves_tr[name]; mo = month[tr_idx]
    tab = pd.crosstab(lv, mo, normalize="index")
    for m in range(12):
        if m not in tab.columns: tab[m] = 0.0
    tab = tab[sorted(tab.columns)]
    tab = tab.iloc[tab.values.argmax(axis=1).argsort()]
    im = ax.imshow(tab.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(12)); ax.set_xticklabels(MON, fontsize=6, rotation=90)
    ax.set_yticks([]); ax.set_ylabel(f"leaves ({tab.shape[0]})", fontsize=8)
    ax.axvline(10.5, color="red", lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.8, label="share of leaf's training days")
fig.suptitle("Leaf composition over calendar months (December column marked)", fontsize=9)
fig.savefig(os.path.expanduser("~/december_heatmap.pdf")); plt.close(fig)
print("figures written: ~/december_circle.pdf ~/december_heatmap.pdf")
