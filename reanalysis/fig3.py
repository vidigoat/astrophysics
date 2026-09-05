import pickle, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
rng = np.random.default_rng(55)


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    return {k: v[m] for k, v in d.items()}


def sd(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.std(y - A @ b))


def sl(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b[1]


COL = {"TNG50": "#C1272D", "EAGLE": "#0B6E99", "SIMBA": "#2E7D32"}
fig, ax = plt.subplots(figsize=(9.6, 5.6))
xs = []
for i, t in enumerate(["TNG50", "EAGLE", "SIMBA"]):
    d = load(t)
    mh, ms, bh = d["DM_MASS"], d["STELLAR_MASS"], d["BH_MASS"]
    V = np.var(mh)
    b_ = sl(mh, ms)
    c_ = sl(mh, bh)
    s1 = sd(ms, [mh])
    s2 = sd(bh, [mh])
    a_f = c_ * b_ * V / (b_ * b_ * V + s1 * s1)
    vg = s2 * s2 + c_ * c_ * V - (c_ * b_ * V) ** 2 / (b_ * b_ * V + s1 * s1)
    Rf = s2 / np.sqrt(a_f**2 * s1 * s1 + max(vg, 1e-12))
    am = sl(ms, bh)
    Rm = sd(bh, [mh]) / np.sqrt(am**2 * sd(ms, [mh]) ** 2 + sd(bh, [ms]) ** 2)
    n = len(mh)
    bs = []
    for _ in range(500):
        j = rng.choice(n, n, replace=True)
        a2 = sl(ms[j], bh[j])
        bs.append(sd(bh[j], [mh[j]]) / np.sqrt(a2**2 * sd(ms[j], [mh[j]]) ** 2 + sd(bh[j], [ms[j]]) ** 2))
    e = np.std(bs)
    ax.errorbar(i, Rm, yerr=5 * e, fmt="o", ms=14, color=COL[t], capsize=6, lw=2.5, zorder=5, label=None)
    ax.plot(i, Rf, marker="_", ms=34, mew=3, color="0.35", zorder=4)
    ax.annotate(
        f"{Rm:.3f}",
        (i, Rm),
        xytext=(16, -4),
        textcoords="offset points",
        fontsize=11,
        color=COL[t],
        weight="bold",
    )
    ax.annotate(
        "fork\nprediction",
        (i, Rf),
        xytext=(-64, -16),
        textcoords="offset points",
        fontsize=9,
        color="0.35",
        ha="center",
    )
    xs.append(t)
ax.axhline(1.0, color="k", lw=2, ls="--", zorder=2)
ax.text(2.42, 1.005, "CHAIN\n$M_h\\to M_\\star\\to M_{BH}$", fontsize=10, va="bottom", ha="right")
ax.set_xticks(range(3))
ax.set_xticklabels(xs, fontsize=13, weight="bold")
for lbl, c in zip(ax.get_xticklabels(), [COL[t] for t in xs]):
    lbl.set_color(c)
ax.set_ylabel(r"$R=\sigma(M_{BH}|M_h)_{\rm obs}\ /\ \sigma(M_{BH}|M_h)_{\rm chain}$", fontsize=12)
ax.set_ylim(0.52, 1.12)
ax.set_xlim(-0.5, 2.5)
ax.grid(alpha=0.2, axis="y")
ax.set_title(
    "A parameter-free test of black hole growth topology\n"
    "error bars are $5\\sigma$; the fork prediction has no free parameters",
    fontsize=13,
    weight="bold",
)
fig.tight_layout()
fig.savefig("/Users/vidigoat/astrophysics/reanalysis/results/topology.png", dpi=170)
print("saved")
