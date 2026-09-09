"""Clean figure set in the style of the earlier submission's corner plots:
filled contours at 39.3 / 86.5 / 98.9 per cent with a dark edge, light scatter for the
tails, one colour family per code (TNG50 green, EAGLE blue, SIMBA orange).

fig2_screening.png   2x3  halo at fixed galaxy / galaxy at fixed halo
fig3_variants.png    1x5  SIMBA feedback experiment (halo at fixed galaxy)
fig4_feedback.png    1x4  gas removal (3 codes) + metallicity test
fig6_forecast.png    1x1  forecast
fig_corner.png       corner plot of (M*, M_h, sigma, M_BH) for the three codes overlaid
"""

import os, io, contextlib, pickle, csv
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from scipy.stats import gaussian_kde

with contextlib.redirect_stdout(io.StringIO()):
    from headline import tng50, eagle, simba
from common import pcorr, resid
from load_simba import load_simba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "paper", "Plots", "v3")
rng = np.random.default_rng(11)
PAL = {
    "TNG50": dict(fill=["#00441b", "#238b45", "#a1d99b"], edge="#238b45", hist="#c7e9c0", scatter="#238b45"),
    "EAGLE": dict(fill=["#08306b", "#2171b5", "#9ecae1"], edge="#2171b5", hist="#c6dbef", scatter="#2171b5"),
    "SIMBA": dict(fill=["#7f2704", "#d94801", "#fdae6b"], edge="#d94801", hist="#fdd0a2", scatter="#d94801"),
}
Q = (0.393, 0.865, 0.989)
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


def kde_levels(zz):
    flat = np.sort(zz.ravel())[::-1]
    cum = np.cumsum(flat)
    return [float(flat[min(np.searchsorted(cum, q * cum[-1]), len(flat) - 1)]) for q in Q]


def contour_panel(ax, x, y, code, lims, fit=True, n_kde=4000, grid=70, sym="slope"):
    p = PAL[code]
    i = rng.choice(len(x), min(len(x), n_kde), replace=False)
    kde = gaussian_kde(np.vstack([x[i], y[i]]))
    xx, yy = np.meshgrid(np.linspace(lims[0], lims[1], grid), np.linspace(lims[2], lims[3], grid))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    lv = sorted(set(kde_levels(zz)))
    zmax = zz.max() * 1.01
    if len(lv) == 3 and zmax > lv[2]:
        ax.contourf(xx, yy, zz, levels=[lv[2], zmax], colors=[p["fill"][0]], alpha=0.92, zorder=2)
        ax.contourf(xx, yy, zz, levels=[lv[1], lv[2]], colors=[p["fill"][1]], alpha=0.78, zorder=2)
        ax.contourf(xx, yy, zz, levels=[lv[0], lv[1]], colors=[p["fill"][2]], alpha=0.58, zorder=2)
        ax.contour(xx, yy, zz, levels=lv, colors=p["edge"], linewidths=1.1, zorder=3)
    # tails outside the outer contour
    dens = kde(np.vstack([x, y]))
    out = dens < lv[0]
    ax.scatter(x[out], y[out], s=2.5, color=p["scatter"], alpha=0.45, lw=0, zorder=1, rasterized=True)
    b = np.polyfit(x, y, 1)
    if fit:
        xx1 = np.array([lims[0], lims[1]])
        ax.plot(xx1, np.polyval(b, xx1), color="k", lw=1.6, zorder=4)
        ax.text(
            0.05,
            0.94,
            (f"${sym} = {b[0]:+.2f}$" if sym != "slope" else f"slope {b[0]:+.2f}"),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
        )
    ax.axhline(0, color="0.75", lw=0.6, zorder=0)
    ax.axvline(0, color="0.75", lw=0.6, zorder=0)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    return b[0]


D = {"TNG50": tng50(), "EAGLE": eagle(), "SIMBA": simba()}
SEL = {c: (d["MS"] > 9.5) & (d["MS"] < 11.0) & (d["MH"] > 10.5) for c, d in D.items()}

# ---------------------------------------------------------------- Fig 2
fig, axes = plt.subplots(
    2, 3, figsize=(10.5, 6.6), sharex="row", sharey=True, gridspec_kw={"hspace": 0.3, "wspace": 0.08}
)
for j, code in enumerate(D):
    d, m = D[code], SEL[code]
    ms, mh, bh = d["MS"][m], d["MH"][m], d["BH"][m]
    contour_panel(axes[0, j], resid(mh, [ms]), resid(bh, [ms]), code, (-0.8, 0.8, -1.1, 1.1), sym="b")
    contour_panel(axes[1, j], resid(ms, [mh]), resid(bh, [mh]), code, (-0.8, 0.8, -1.1, 1.1), sym="a")
    axes[0, j].set_title(code, color=PAL[code]["edge"], fontweight="bold")
    axes[0, j].set_xlabel(r"$\Delta \log M_{\rm h}$  (fixed $M_\star$)")
    axes[1, j].set_xlabel(r"$\Delta \log M_\star$  (fixed $M_{\rm h}$)")
axes[0, 0].set_ylabel(r"$\Delta \log M_{\rm BH}$  (fixed $M_\star$)")
axes[1, 0].set_ylabel(r"$\Delta \log M_{\rm BH}$  (fixed $M_{\rm h}$)")
fig.savefig(os.path.join(OUT, "fig2_screening.png"))
fig.savefig(os.path.join(OUT, "fig2_screening.pdf"))
print("-> fig2")

# ---------------------------------------------------------------- Fig 3 (variants)
runs = [
    ("s50nofb", "no feedback"),
    ("s50noagn", "stellar only"),
    ("s50nojet", "+ AGN winds"),
    ("s50nox", "+ jets"),
    ("s50", "+ X-ray (full)"),
]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
for j, (run, lab) in enumerate(runs):
    d, _ = load_simba(os.path.join(ROOT, "Data", "simba_variants", f"m50n512_{run}_151.hdf5"))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0) & (d["M200"] > 10.5)
    ms, mh, bh = d["STELLAR_MASS"][m], d["M200"][m], d["BH_MASS"][m]
    contour_panel(axes[j], resid(mh, [ms]), resid(bh, [ms]), "SIMBA", (-0.8, 0.8, -1.3, 1.3), sym="b")
    axes[j].set_title(lab, fontweight="bold", color=PAL["SIMBA"]["edge"])
    axes[j].set_xlabel(r"$\Delta \log M_{\rm h}$  (fixed $M_\star$)")
axes[0].set_ylabel(r"$\Delta \log M_{\rm BH}$  (fixed $M_\star$)")
fig.savefig(os.path.join(OUT, "fig3_variants.png"))
fig.savefig(os.path.join(OUT, "fig3_variants.pdf"))
print("-> fig3")


# ---------------------------------------------------------------- Fig 4 (gas + metals)
def gas_arrays(ms, mh, bh, mg, sg, hi=11.5):
    m = (ms > 9.0) & (mh < hi) & np.isfinite(mg) & (mg > 6) & np.isfinite(sg)
    Z = [ms[m], mh[m], sg[m]]
    return resid(bh[m], Z), resid(mg[m], Z)


G = {}
g = {
    k: np.asarray(v, float)
    for k, v in pickle.load(open(os.path.join(ROOT, "reanalysis", "data", "TNG50_full.pkl"), "rb")).items()
}
mm = g["BH_SEEDED"] == 1
G["TNG50"] = gas_arrays(
    g["STELLAR_MASS"][mm],
    g["DM_MASS"][mm],
    g["BH_MASS"][mm],
    g["GAS_MASS"][mm],
    np.log10(np.maximum(g["VEL_DISP"][mm], 1e-3)),
)
e = pd.read_csv(os.path.join(ROOT, "Data", "eagle_v3", "RefL0100N1504_z0.csv"))
ok = (e["BlackHoleMass"] > 0) & (e["Group_M_Crit200"] > 0)
G["EAGLE"] = gas_arrays(
    np.log10(e["Mass_Star"][ok].to_numpy(float)),
    np.log10(e["Group_M_Crit200"][ok].to_numpy(float)),
    np.log10(e["BlackHoleMass"][ok].to_numpy(float)),
    np.log10(np.maximum(e["Mass_Gas"][ok].to_numpy(float), 1)),
    np.log10(np.maximum(e["VelDisp"][ok].to_numpy(float), 1e-3)),
)
d, _ = load_simba(os.path.join(ROOT, "Data", "simba_m100n1024_151.hdf5"))
G["SIMBA"] = gas_arrays(d["STELLAR_MASS"], d["M200"], d["BH_MASS"], d["GAS_MASS"], d["LOG_SIGMA"])
P = os.path.join(ROOT, "Data")
tng = {
    k: np.asarray(v, float) for k, v in pickle.load(open(os.path.join(P, "tng50_final.pkl"), "rb")).items()
}
S = {
    "TNG50": {"STELLAR_MASS": tng["GAS_MASS"], "GAS_METALLICITY": tng["GAS_METALLICITY"], "SFR": tng["SFR"]},
    "EAGLE": {
        k: np.asarray(v, float)
        for k, v in pickle.load(open(os.path.join(P, "eagle_final.pkl"), "rb")).items()
    },
    "SIMBA": {
        k: np.asarray(v, float)
        for k, v in pickle.load(open(os.path.join(P, "simba_final.pkl"), "rb")).items()
    },
}


def prep(dd):
    ms, z, sfr = dd["STELLAR_MASS"], dd["GAS_METALLICITY"], dd["SFR"]
    ok = (sfr > 0) & (z > 0) & np.isfinite(ms)
    return ms[ok], np.log10(z[ok]), np.log10(sfr[ok]) - ms[ok]


fig, axes = plt.subplots(
    1, 4, figsize=(15, 3.7), gridspec_kw={"wspace": 0.25, "width_ratios": [1, 1, 1, 1.15]}
)
for j, code in enumerate(["TNG50", "EAGLE", "SIMBA"]):
    x, y = G[code]
    contour_panel(axes[j], x, y, code, (-1.0, 1.0, -1.0, 1.0))
    axes[j].set_title(code, color=PAL[code]["edge"], fontweight="bold")
    axes[j].set_xlabel(r"$\Delta \log M_{\rm BH}$  (fixed $M_\star, M_{\rm h}, \sigma$)")
    if j == 0:
        axes[j].set_ylabel(r"$\Delta \log M_{\rm gas}$")
ax = axes[3]
BINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]
for code, dd in S.items():
    ms, z, ss = prep(dd)
    xs, vs, es = [], [], []
    for lo, hi in BINS:
        m = (ms > lo) & (ms < hi)
        n = int(m.sum())
        if n < 120:
            continue
        v = pcorr(z[m], ss[m], [ms[m]])[0]
        bs = [
            pcorr(z[m][i], ss[m][i], [ms[m][i]])[0]
            for i in (rng.choice(n, n, replace=True) for _ in range(100))
        ]
        xs.append(0.5 * (lo + hi))
        vs.append(v)
        es.append(np.std(bs))
    ax.errorbar(xs, vs, es, fmt="o-", color=PAL[code]["edge"], ms=6, lw=1.6, capsize=2.5, label=code)
ax.axhline(0, color="0.3", lw=0.8)
ax.axvspan(10.5, 11.5, color="0.5", alpha=0.12, lw=0)
ax.set_xlabel(r"$\log M_\star\ [M_\odot]$")
ax.set_ylabel(r"$r(Z_{\rm gas},\,{\rm sSFR}\,|\,M_\star)$")
ax.set_ylim(-0.95, 0.75)
ax.legend(loc="upper left")
ax.set_title("metallicity vs star formation", fontweight="bold")
fig.savefig(os.path.join(OUT, "fig4_feedback.png"))
fig.savefig(os.path.join(OUT, "fig4_feedback.pdf"))
print("-> fig4")

# ---------------------------------------------------------------- Fig 6 (forecast, one panel)
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "forecast.csv"))))
sig = sorted(set(float(r["sigma_h"]) for r in rows))
Ns = sorted(set(int(r["N"]) for r in rows))
get = lambda s, N, k: float(next(r for r in rows if float(r["sigma_h"]) == s and int(r["N"]) == N)[k])
fig, ax = plt.subplots(figsize=(5.2, 4.0))
shade = {0.1: "#1A3A5C", 0.2: "#2E5F8C", 0.3: "#5B9BD5", 0.4: "#9CC3E4"}
for s in sig:
    if s == 0.0:
        continue
    ax.plot(
        Ns,
        [get(s, N, "p_EAGLE_above_TNG_3sig") for N in Ns],
        "o-",
        color=shade[s],
        ms=6,
        lw=1.8,
        label=f"{s:.1f} dex",
    )
ax.axhline(0.95, color="0.4", lw=0.9, ls="--")
ax.text(Ns[0] * 0.95, 0.955, "95%", fontsize=8.5, color="0.35", va="bottom")
ax.set_xscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0, 1.04)
ax.set_xlabel("galaxies with dynamical $M_{\\rm BH}$ and a halo mass")
ax.set_ylabel("probability of a 3$\\sigma$ decision")
ax.legend(loc="lower right", title="halo-mass error", title_fontsize=9)
fig.savefig(os.path.join(OUT, "fig6_forecast.png"))
fig.savefig(os.path.join(OUT, "fig6_forecast.pdf"))
print("-> fig6")

# ---------------------------------------------------------------- corner plots: one per code, in a row (old-paper style)
vars_ = [
    ("MS", r"$\log M_\star$"),
    ("MH", r"$\log M_{\rm h}$"),
    ("SIG", r"$\log \sigma_\star$"),
    ("BH", r"$\log M_{\rm BH}$"),
]
n = len(vars_)
lims = {"MS": (9.5, 11.0), "MH": (10.8, 13.2), "SIG": (1.7, 2.5), "BH": (5.5, 9.3)}
fig = plt.figure(figsize=(15, 5.2))
outer = fig.add_gridspec(1, 3, wspace=0.12)
for c_i, code in enumerate(D):
    d, m, p = D[code], SEL[code], PAL[code]
    inner = outer[c_i].subgridspec(n, n, hspace=0.06, wspace=0.06)
    for i, (vi, li) in enumerate(vars_):
        for j, (vj, lj) in enumerate(vars_):
            if j > i:
                continue
            ax = fig.add_subplot(inner[i, j])
            if i == j:
                ax.hist(
                    d[vi][m],
                    bins=35,
                    range=lims[vi],
                    density=True,
                    histtype="stepfilled",
                    facecolor=p["hist"],
                    edgecolor=p["edge"],
                    lw=1.2,
                )
                ax.set_yticks([])
            else:
                x, y = d[vj][m], d[vi][m]
                k = rng.choice(len(x), min(len(x), 3000), replace=False)
                kde = gaussian_kde(np.vstack([x[k], y[k]]))
                xx, yy = np.meshgrid(np.linspace(*lims[vj], 60), np.linspace(*lims[vi], 60))
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                lv = sorted(set(kde_levels(zz)))
                zmax = zz.max() * 1.01
                if len(lv) == 3:
                    ax.contourf(xx, yy, zz, levels=[lv[2], zmax], colors=[p["fill"][0]], alpha=0.92, zorder=2)
                    ax.contourf(
                        xx, yy, zz, levels=[lv[1], lv[2]], colors=[p["fill"][1]], alpha=0.78, zorder=2
                    )
                    ax.contourf(
                        xx, yy, zz, levels=[lv[0], lv[1]], colors=[p["fill"][2]], alpha=0.58, zorder=2
                    )
                    ax.contour(xx, yy, zz, levels=lv, colors=p["edge"], linewidths=0.9, zorder=3)
                dens = kde(np.vstack([x, y]))
                out = dens < lv[0]
                ax.scatter(
                    x[out], y[out], s=1.5, color=p["scatter"], alpha=0.4, lw=0, zorder=1, rasterized=True
                )
                ax.set_ylim(lims[vi])
            ax.set_xlim(lims[vj])
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.tick_params(labelsize=8)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(lj, fontsize=10)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(li, fontsize=10)
    fig.text(
        outer[c_i].get_position(fig).x0 + outer[c_i].get_position(fig).width / 2,
        0.93,
        code,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
fig.savefig(os.path.join(OUT, "fig_corner.png"))
fig.savefig(os.path.join(OUT, "fig_corner.pdf"))
print("-> corner")
