"""Money plot: residual-residual. Does the halo still know about M_BH once
stellar mass is accounted for?  A slope means YES (halo-coupled accretion)."""

import pickle, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
LO, HI = 9.5, 11.0


def res(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
COL = {"TNG50": "#C1272D", "EAGLE": "#0B6E99", "SIMBA": "#2E7D32"}
for j, t in enumerate(["TNG50", "EAGLE", "SIMBA"]):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    d = {k: v[m] for k, v in d.items()}
    bh, ms, mh = d["BH_MASS"], d["STELLAR_MASS"], d["DM_MASS"]

    # top: M_BH residual vs halo residual, both at fixed M*
    x, y = res(mh, [ms]), res(bh, [ms])
    ax = axes[0, j]
    ax.scatter(x, y, s=3, alpha=0.18, color=COL[t], lw=0)
    s = np.polyfit(x, y, 1)[0]
    r = np.corrcoef(x, y)[0, 1]
    xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 50)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="k", lw=2)
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.set_title(f"{t}", fontsize=13, weight="bold", color=COL[t])
    ax.set_xlabel(r"$\Delta \log M_{\rm halo}$  (at fixed $M_\star$)")
    if j == 0:
        ax.set_ylabel(r"$\Delta \log M_{\rm BH}$  (at fixed $M_\star$)")
    ax.text(
        0.04,
        0.93,
        f"slope = {s:.2f}\nr = {r:+.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(fc="white", ec="0.7", alpha=0.9),
    )
    ax.set_xlim(np.percentile(x, 0.5), np.percentile(x, 99.5))
    ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))

    # bottom: the mirror test - M_BH residual vs stellar residual at fixed halo
    x2, y2 = res(ms, [mh]), res(bh, [mh])
    ax = axes[1, j]
    ax.scatter(x2, y2, s=3, alpha=0.18, color=COL[t], lw=0)
    s2 = np.polyfit(x2, y2, 1)[0]
    r2 = np.corrcoef(x2, y2)[0, 1]
    xs = np.linspace(np.percentile(x2, 1), np.percentile(x2, 99), 50)
    ax.plot(xs, np.polyval(np.polyfit(x2, y2, 1), xs), color="k", lw=2)
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel(r"$\Delta \log M_\star$  (at fixed $M_{\rm halo}$)")
    if j == 0:
        ax.set_ylabel(r"$\Delta \log M_{\rm BH}$  (at fixed $M_{\rm halo}$)")
    ax.text(
        0.04,
        0.93,
        f"slope = {s2:.2f}\nr = {r2:+.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(fc="white", ec="0.7", alpha=0.9),
    )
    ax.set_xlim(np.percentile(x2, 0.5), np.percentile(x2, 99.5))
    ax.set_ylim(np.percentile(y2, 0.5), np.percentile(y2, 99.5))

fig.suptitle(
    "What does the black hole actually know about?   " "EAGLE: the halo.   TNG50 / SIMBA: the stars.",
    fontsize=14,
    weight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/Users/vidigoat/astrophysics/reanalysis/results/bh_screening.png", dpi=170)
print("saved bh_screening.png")
