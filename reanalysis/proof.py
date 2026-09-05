"""PROOF PASS. Each block answers a specific referee objection."""

import pickle, numpy as np

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
rng = np.random.default_rng(31)


def load(t, lo=None, hi=None, key="STELLAR_MASS"):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    if lo is not None:
        m = (d[key] > lo) & (d[key] < hi)
        d = {k: v[m] for k, v in d.items()}
    return d


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


print("#" * 78)
print("# E1.  JOINT REGRESSION.  log M_BH = a*log M* + b*log M_halo + c")
print("#      The cleanest statement: is the halo coefficient b consistent with ZERO?")
print("#      Bootstrap 2000x. Common window 9.5<logM*<11.")
print("#" * 78)
print(f'{"":8s}{"a (stellar)":>20s}{"b (halo)":>20s}{"b/sigma_b":>12s}{"R^2":>8s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t, 9.5, 11.0)
    y, X = d["BH_MASS"], np.column_stack([np.ones(len(d["BH_MASS"])), d["STELLAR_MASS"], d["DM_MASS"]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.var(y - X @ b) / np.var(y)
    n = len(y)
    bs = []
    for _ in range(2000):
        i = rng.choice(n, n, replace=True)
        bb, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
        bs.append(bb)
    bs = np.array(bs)
    sa, sb = bs[:, 1].std(), bs[:, 2].std()
    print(f"{t:8s}{b[1]:>13.3f} +-{sa:5.3f}{b[2]:>13.3f} +-{sb:5.3f}{b[2]/sb:>12.1f}{r2:>8.3f}")

print()
print("#" * 78)
print("# E2.  APERTURE OBJECTION.  EAGLE M* is a 30 kpc aperture mass, TNG/SIMBA are")
print("#      total. Could that ALONE explain EAGLE? Find the M* degradation needed")
print("#      to push TNG50 down to EAGLE's value of r(M_BH,M*|M_halo)=0.182.")
print("#" * 78)
eagle_val = pc(
    *[load("EAGLE", 9.5, 11.0)[k] for k in ["BH_MASS", "STELLAR_MASS"]], [load("EAGLE", 9.5, 11.0)["DM_MASS"]]
)
print(f"EAGLE observed r(M_BH,M*|M_halo) = {eagle_val:.3f}")
d = load("TNG50", 9.5, 11.0)
for nz in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
    ms = d["STELLAR_MASS"] + rng.normal(0, nz, len(d["STELLAR_MASS"])) if nz > 0 else d["STELLAR_MASS"]
    v = pc(d["BH_MASS"], ms, [d["DM_MASS"]])
    flag = " <-- reaches EAGLE" if v <= eagle_val else ""
    print(f"   TNG50 with {nz:4.2f} dex noise on log M*:  r = {v:6.3f}{flag}")
print("   For scale: a 30 kpc aperture loses <0.05 dex of stellar mass at these masses.")

print()
print("#" * 78)
print("# E3.  MECHANISM TEST.  TNG's feedback mode switches on the BLACK HOLE MASS")
print("#      itself: chi = min(0.002 (M_BH/1e8)^2, 0.1)  (Weinberger+2017).")
print("#      Kinetic mode engages near M_BH ~ 1e8. PREDICTION: if the mechanism is")
print("#      right, TNG50 should stay halo-blind on BOTH sides of that threshold,")
print("#      because the switch never references the halo.")
print("#" * 78)
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    for lab, m in [
        ("M_BH < 1e7 ", d["BH_MASS"] < 7.0),
        ("1e7-1e8   ", (d["BH_MASS"] >= 7.0) & (d["BH_MASS"] < 8.0)),
        ("M_BH > 1e8", d["BH_MASS"] >= 8.0),
    ]:
        n = int(m.sum())
        if n < 120:
            print(f"  {t:7s} {lab}  N={n:6d}   (too few)")
            continue
        r_h = pc(d["BH_MASS"][m], d["DM_MASS"][m], [d["STELLAR_MASS"][m]])
        r_s = pc(d["BH_MASS"][m], d["STELLAR_MASS"][m], [d["DM_MASS"][m]])
        print(f"  {t:7s} {lab}  N={n:6d}   r(bh,Mh|M*)={r_h:+6.3f}   r(bh,M*|Mh)={r_s:+6.3f}")
    print()

print("#" * 78)
print("# E4.  BINDING-ENERGY PREDICTION.  If EAGLE self-regulates by unbinding halo")
print("#      gas:  eps_f eps_r M_BH c^2 = E_bind ~ f_g M_h * (G M_h / R_vir),")
print("#      R_vir ~ M_h^(1/3)  =>  M_BH ~ M_h^(5/3) = M_h^1.67")
print("#      Booth & Schaye 2010 measured 1.55 in OWLS. What does EAGLE give here?")
print("#" * 78)
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    mh, bh = d["DM_MASS"], d["BH_MASS"]
    lo, hi = np.percentile(mh, [2, 98])
    m = (mh > lo) & (mh < hi)
    A = np.column_stack([np.ones(m.sum()), mh[m]])
    b, *_ = np.linalg.lstsq(A, bh[m], rcond=None)
    n = int(m.sum())
    bs = []
    for _ in range(1000):
        i = rng.choice(n, n, replace=True)
        bb, *_ = np.linalg.lstsq(A[i], bh[m][i], rcond=None)
        bs.append(bb[1])
    print(
        f"  {t:7s} M_BH ~ M_halo^({b[1]:.3f} +- {np.std(bs):.3f})   over log M_halo {lo:.1f}-{hi:.1f}, N={n}"
    )

print()
print("#" * 78)
print("# E5.  IS IT CONDITIONAL INDEPENDENCE, NOT JUST WEAK CORRELATION?")
print("#      Fisher z-test of r(M_BH, M*|M_halo)=0 and r(M_BH,M_halo|M*)=0.")
print("#" * 78)
from math import atanh, sqrt

print(f'{"":8s}{"z[M* | halo]":>18s}{"z[halo | M*]":>18s}{"verdict":>28s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t, 9.5, 11.0)
    n = len(d["BH_MASS"])
    rs = pc(d["BH_MASS"], d["STELLAR_MASS"], [d["DM_MASS"]])
    rh = pc(d["BH_MASS"], d["DM_MASS"], [d["STELLAR_MASS"]])
    zs, zh = atanh(rs) * sqrt(n - 4), atanh(rh) * sqrt(n - 4)
    if abs(zh) < 3 and abs(zs) > 5:
        v = "M_BH _||_ halo  GIVEN M*"
    elif abs(zs) < 8 and abs(zh) > 8 and abs(zh) > abs(zs):
        v = "halo dominates"
    else:
        v = "both non-zero"
    print(f"{t:8s}{zs:>18.1f}{zh:>18.1f}{v:>28s}")
