"""Two figure types not yet used in the paper.

fig2_binned.png   : median log M_BH vs log M_h in bins of stellar mass, one panel per code,
                    lines with 16-84 per cent bands (the classic 'relation at fixed M*' plot)
fig6_heatmap.png  : forecast as an annotated grid, N x halo-mass precision -> probability
"""

import os, io, contextlib, csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

with contextlib.redirect_stdout(io.StringIO()):
    from headline import tng50, eagle, simba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "paper", "Plots", "v3")
PAL = {"TNG50": "#2ca02c", "EAGLE": "#1f77b4", "SIMBA": "#ff7f0e"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }
)

D = {"TNG50": tng50(), "EAGLE": eagle(), "SIMBA": simba()}

# ------------------------------------------------------------ Fig: M_BH vs M_h at fixed M*
ms_bins = [(9.5, 9.9), (9.9, 10.3), (10.3, 10.7), (10.7, 11.1)]
cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(ms_bins)))
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (code, d) in zip(axes, D.items()):
    base = d["MH"] > 10.5
    for (lo, hi), c in zip(ms_bins, cols):
        m = base & (d["MS"] >= lo) & (d["MS"] < hi)
        mh, bh = d["MH"][m], d["BH"][m]
        edges = np.arange(10.8, 13.4, 0.2)
        x, med, p16, p84 = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (mh >= a) & (mh < b)
            if s.sum() < 25:
                continue
            x.append(0.5 * (a + b))
            q = np.percentile(bh[s], [16, 50, 84])
            p16.append(q[0])
            med.append(q[1])
            p84.append(q[2])
        if len(x) < 2:
            continue
        ax.fill_between(x, p16, p84, color=c, alpha=0.18, lw=0)
        ax.plot(x, med, color=c, lw=2.0, label=f"${lo:.1f}\\leq\\log M_\\star<{hi:.1f}$")
    ax.set_title(code, color=PAL[code], fontweight="bold")
    ax.set_xlabel(r"$\log M_{\rm h}\ [M_\odot]$")
    ax.set_xlim(10.8, 13.3)
axes[0].set_ylabel(r"$\log M_{\rm BH}\ [M_\odot]$")
axes[0].set_ylim(5.6, 9.4)
axes[0].legend(loc="upper left", title=r"stellar-mass bin", title_fontsize=9)
fig.savefig(os.path.join(OUT, "fig2_binned.png"))
fig.savefig(os.path.join(OUT, "fig2_binned.pdf"))
print("-> fig2_binned")

# ------------------------------------------------------------ Fig: forecast heat map
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "forecast.csv"))))
sig = [s for s in sorted(set(float(r["sigma_h"]) for r in rows)) if s > 0]
Ns = sorted(set(int(r["N"]) for r in rows))
get = lambda s, N: float(
    next(r for r in rows if float(r["sigma_h"]) == s and int(r["N"]) == N)["p_EAGLE_above_TNG_3sig"]
)
M = np.array([[get(s, N) for N in Ns] for s in sig])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
cmap = LinearSegmentedColormap.from_list("p", ["#f7fbff", "#9ecae1", "#2E5F8C", "#1A3A5C"])
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
for i in range(len(sig)):
    for j in range(len(Ns)):
        v = M[i, j]
        ax.text(
            j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if v > 0.55 else "#222222"
        )
ax.set_xticks(range(len(Ns)))
ax.set_xticklabels([str(n) for n in Ns])
ax.set_yticks(range(len(sig)))
ax.set_yticklabels([f"{s:.1f}" for s in sig])
ax.set_xlabel("galaxies with dynamical $M_{\\rm BH}$ and a halo mass")
ax.set_ylabel("halo-mass error [dex]")
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(Ns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(sig), 1), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("probability of a $3\\sigma$ decision")
cb.outline.set_visible(False)
fig.savefig(os.path.join(OUT, "fig6_heatmap.png"))
fig.savefig(os.path.join(OUT, "fig6_heatmap.pdf"))
print("-> fig6_heatmap")

# ------------------------------------------------------------ Fig 4: median M_BH on the (M_h, M*) plane, SIMBA variants
from load_simba import load_simba

runs = [
    ("s50nofb", "no feedback"),
    ("s50noagn", "stellar only"),
    ("s50nojet", "+ AGN winds"),
    ("s50nox", "+ jets"),
    ("s50", "+ X-ray (full)"),
]
xe = np.arange(10.8, 13.21, 0.3)
ye = np.arange(9.5, 11.11, 0.2)
fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.7), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (run, lab) in zip(axes, runs):
    d, _ = load_simba(os.path.join(ROOT, "Data", "simba_variants", f"m50n512_{run}_151.hdf5"))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.1) & (d["M200"] > 10.8) & (d["M200"] < 13.2)
    mh, ms, bh = d["M200"][m], d["STELLAR_MASS"][m], d["BH_MASS"][m]
    grid = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            s = (ms >= ye[i]) & (ms < ye[i + 1]) & (mh >= xe[j]) & (mh < xe[j + 1])
            if s.sum() >= 8:
                grid[i, j] = np.median(bh[s])
    # deviation from the median of each stellar-mass row: flat = galaxy sets M_BH, gradient along M_h = halo does
    for i in range(len(ye) - 1):
        row = (ms >= ye[i]) & (ms < ye[i + 1])
        if row.sum() >= 8:
            grid[i, :] -= np.median(bh[row])
    im = ax.pcolormesh(xe, ye, grid, cmap="RdBu_r", vmin=-0.6, vmax=0.6, edgecolors="white", linewidth=0.6)
    ax.set_title(lab, fontweight="bold", color="#222222")
    ax.set_xlabel(r"$\log M_{\rm h}\ [M_\odot]$")
    ax.set_xticks([11, 12, 13])
    ax.tick_params(length=0)
axes[0].set_ylabel(r"$\log M_\star\ [M_\odot]$")
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
cb.set_label(r"$\Delta\log M_{\rm BH}$ at fixed $M_\star$")
cb.outline.set_visible(False)
fig.savefig(os.path.join(OUT, "fig3_plane.png"))
fig.savefig(os.path.join(OUT, "fig3_plane.pdf"))
print("-> fig3_plane")

# ------------------------------------------------------------ Fig 3 (scatter + fit + predictions), Desmond-style
PT = {"TNG50": "#2ca02c", "EAGLE": "#1f77b4", "SIMBA": "#ff7f0e"}
from common import resid

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True, gridspec_kw={"wspace": 0.06})
for k, (ax, (code, d)) in enumerate(zip(axes, D.items())):
    m = (d["MS"] > 9.5) & (d["MS"] < 11.0) & (d["MH"] > 10.5)
    x = resid(d["MH"][m], [d["MS"][m]])
    y = resid(d["BH"][m], [d["MS"][m]])
    sel = np.random.default_rng(3).choice(len(x), min(len(x), 3500), replace=False)
    ax.scatter(x[sel], y[sel], s=5, color="0.45", alpha=0.35, lw=0, rasterized=True)
    b = np.polyfit(x, y, 1)
    xx = np.array([-0.85, 0.85])
    ax.plot(xx, xx * (5.0 / 3.0), ls="--", color="black", lw=1.1,
            label=r"halo binding energy, $b=5/3$")
    ax.plot(xx, xx * 0.0, ls=":", color="black", lw=1.1,
            label=r"galaxy-regulated, $b=0$")
    ax.plot(xx, np.polyval(b, xx), color=PT[code], lw=2.6, label=f"fit  $b={b[0]:+.2f}$")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel(r"$\Delta\log M_{\rm h}$ at fixed $M_\star$")
    ax.set_title(code, fontsize=12)
    ax.legend(loc="upper left", fontsize=9, handlelength=2.2)
axes[0].set_ylabel(r"$\Delta\log M_{\rm BH}$ at fixed $M_\star$")
fig.savefig(os.path.join(OUT, "fig2_scatter.png"))
fig.savefig(os.path.join(OUT, "fig2_scatter.pdf"))
print("-> fig2_scatter")
