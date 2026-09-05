"""Which property predicts M_BH most tightly, in each code, versus the OBSERVED ranking.
Observed (Kormendy & Ho 2013; Kormendy & Bender 2011):
   M_BH-sigma      intrinsic scatter ~0.29 dex   <- tightest or tied
   M_BH-M_bulge    intrinsic scatter ~0.29-0.33 dex
   M_BH-M_halo     NO correlation independent of the bulge
So observations rank: sigma ~ bulge  >>  halo (halo adds nothing once bulge is fixed)."""

import pickle, numpy as np

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
rng = np.random.default_rng(66)


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    return {k: v[m] for k, v in d.items()}


def sd(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.std(y - A @ b))


PRED = [("sigma", "LOG_SIGMA"), ("M*", "STELLAR_MASS"), ("M_halo", "DM_MASS")]
print("Scatter in log M_BH at fixed each predictor (dex). Lower = tighter.")
print(f'{"":8s}' + "".join(f"{p:>12s}" for p, _ in PRED) + f'{"tightest":>12s}{"matches obs?":>16s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    bh = d["BH_MASS"]
    vals = [sd(bh, [d[k]]) for _, k in PRED]
    best = PRED[int(np.argmin(vals))][0]
    ok = "YES" if best in ("sigma", "M*") else "NO  <-- halo wins"
    print(f"{t:8s}" + "".join(f"{v:>12.3f}" for v in vals) + f"{best:>12s}{ok:>16s}")
print()
print("Does the halo add ANYTHING once the galaxy is fixed?  (Kormendy & Bender 2011: no)")
print(f'{"":8s}{"sd(MBH|M*,sig)":>18s}{"sd(MBH|M*,sig,Mh)":>20s}{"improvement":>14s}{"verdict":>22s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    bh = d["BH_MASS"]
    a = sd(bh, [d["STELLAR_MASS"], d["LOG_SIGMA"]])
    b = sd(bh, [d["STELLAR_MASS"], d["LOG_SIGMA"], d["DM_MASS"]])
    imp = 100 * (a - b) / a
    v = "agrees with obs" if imp < 3 else "HALO MATTERS - conflicts"
    print(f"{t:8s}{a:>18.3f}{b:>20.3f}{imp:>13.1f}%{v:>22s}")
print()
print("Mirror: does the GALAXY add anything once the halo is fixed?")
print(f'{"":8s}{"sd(MBH|Mh)":>14s}{"sd(MBH|Mh,M*,sig)":>20s}{"improvement":>14s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    bh = d["BH_MASS"]
    a = sd(bh, [d["DM_MASS"]])
    b = sd(bh, [d["DM_MASS"], d["STELLAR_MASS"], d["LOG_SIGMA"]])
    print(f"{t:8s}{a:>14.3f}{b:>20.3f}{100*(a-b)/a:>13.1f}%")
