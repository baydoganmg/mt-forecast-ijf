"""Generate the R2.5 noise-sensitivity figure from saved v2 data.

Faceted layout: one panel per training noise level $\\sigma$, x-axis is the
derivative regularizer weight $\\lambda_\\mathcal{C}$, y-axis is the median
MASE on the clean test set. A star marker on each facet flags the CV-best
$\\lambda$ for that noise level. This layout matches the visual style of
the parameter-illustration figures in Section 3.4 and avoids the line
overlap of the original single-panel version.

Output:
    revision/figures/v2/noise_sensitivity.pdf
    submission/figures/noise_sensitivity.pdf  (production copy)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO.parent
OUT_REV = PAPER_ROOT / "revision" / "figures" / "v2" / "noise_sensitivity.pdf"
OUT_SUB = PAPER_ROOT / "submission" / "figures" / "noise_sensitivity.pdf"

df = pd.read_csv(REPO / "results" / "noise_sensitivity" / "summary.csv")
agg = df.groupby(["sigma", "lambda"])["median_mase_clean_test"].mean().reset_index()
wide = agg.pivot_table(index="lambda", columns="sigma", values="median_mase_clean_test")

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

sigmas = sorted(wide.columns)
fig, axes = plt.subplots(1, len(sigmas), figsize=(15, 3.2),
                          sharey=False, sharex=True)
test_color = "#0072B2"   # matches §3.4 test colour
star_color = "#D55E00"   # vermillion accent for the CV-best marker

for ax, sig in zip(axes, sigmas):
    y = wide[sig].values
    x = wide.index.values
    ax.plot(x, y, color=test_color, marker="s", markersize=7,
            linewidth=2.4, linestyle="-")
    best_idx = int(np.argmin(y))
    ax.scatter(x[best_idx], y[best_idx], s=240, marker="*",
               color=star_color, edgecolor="black", linewidth=1.0, zorder=10)
    ax.set_title(rf"$\sigma = {sig:.1f}$", fontsize=12)
    ax.set_xlabel(r"$\lambda_{\mathcal{C}}$")
    ax.grid(True, linestyle=":", alpha=0.4)
    # Symmetric y-margin per facet so small differences are visible
    span = max(y.max() - y.min(), 0.01)
    pad = span * 0.25
    ax.set_ylim(y.min() - pad, y.max() + pad)

axes[0].set_ylabel("Median MASE on clean test")
fig.suptitle(r"Noise sensitivity of the derivative regularizer on the Hospital dataset"
             "\n"
             r"$\star$ marks the CV-best $\lambda_{\mathcal{C}}$ at each training noise level $\sigma$",
             fontsize=12, y=1.04)
fig.tight_layout()
for out in (OUT_REV, OUT_SUB):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")
plt.close(fig)
