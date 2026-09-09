"""The conditional fit through cosmic time (Sect. 7 of the paper; Table 4).

For each epoch the same fit as Sect. 5 is made on the complete population of centrals in
the same window: log M_BH = a log M* + b log M200 + c.  Also reported are

    T   = d log M_BH / d log M200 with no other variable held fixed,
    s   = d log M* / d log M200 in the window,
    phi = b / T, the share of the black-hole--halo link that is direct,

which satisfy the least-squares identity T = a s + b exactly.

Needs the EAGLE snapshot pulls of pull_eagle_snaps.py in Data/eagle_v3/ and the SIMBA
Caesar catalogues in Data/simba_variants/ (see README).  Writes results/time_evolution.csv.
"""
import os, sys, csv
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ols, pcorr, window
from load_simba import load_simba
from headline import tng50

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(ROOT, "Data", "eagle_v3")
VAR = os.path.join(ROOT, "Data", "simba_variants")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)

EAGLE_SNAP = {28: 0.0, 23: 0.5, 19: 1.0, 15: 2.0, 12: 3.0}
SIMBA_SNAP = {151: 0.0, 125: 0.5, 105: 1.0, 78: 2.0, 62: 3.0}


def eagle_epoch(run, snap):
    f = os.path.join(RAW, f"{run}_z0.csv") if snap == 28 else os.path.join(RAW, f"{run}_snap{snap}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d[(d.BlackHoleMass > 0) & (d.Group_M_Crit200 > 10 ** 10.5)
          & (d.Mass_Star > 10 ** 9.5) & (d.Mass_Star < 1e11)]
    return (np.log10(d.BlackHoleMass.values), np.log10(d.Mass_Star.values),
            np.log10(d.Group_M_Crit200.values))


def simba_epoch(box, snap):
    if box == "m50n512":
        f = os.path.join(VAR, f"m50n512_s50_{snap:03d}.hdf5")
    elif snap == 151:
        f = os.path.join(ROOT, "Data", "simba_m100n1024_151.hdf5")
    else:
        f = os.path.join(VAR, f"m100n1024_{snap:03d}.hdf5")
    if not os.path.exists(f):
        return None
    d, _ = load_simba(f)
    w = window(d) & (d["M200"] > 10.5)
    return d["BH_MASS"][w], d["STELLAR_MASS"][w], d["M200"][w]


def row(code, z, bh, ms, mh, boot=500):
    (a, b), (ea, eb) = ols(bh, ms, mh, boot=boot, seed=1)
    T = np.polyfit(mh, bh, 1)[0]
    s = np.polyfit(mh, ms, 1)[0]
    r = pcorr(bh, mh, [ms])[0]
    print(f"{code:20s} z={z:<4.1f} N={len(bh):6d}  a={a:+.2f}+-{ea:.2f}  b={b:+.2f}+-{eb:.2f}  "
          f"T={T:.2f}  s={s:.2f}  phi={b/T:+.2f}  r={r:+.2f}  [a*s+b={a*s+b:.2f}]")
    return dict(code=code, z=z, N=len(bh), a=a, e_a=ea, b=b, e_b=eb, T=T, s=s, phi=b / T, r_halo=r)


if __name__ == "__main__":
    rows = []
    for run, lab in [("RefL0100N1504", "EAGLE 100 Mpc"), ("RefL0050N0752", "EAGLE 50 Mpc"),
                     ("AGNdT9L0050N0752", "EAGLE 50 dT9")]:
        for snap, z in sorted(EAGLE_SNAP.items(), key=lambda kv: -kv[1]):
            d = eagle_epoch(run, snap)
            if d: rows.append(row(lab, z, *d))
        print()
    for box, lab in [("m50n512", "SIMBA 50 Mpc/h"), ("m100n1024", "SIMBA 100 Mpc/h")]:
        for snap, z in sorted(SIMBA_SNAP.items(), key=lambda kv: -kv[1]):
            d = simba_epoch(box, snap)
            if d: rows.append(row(lab, z, *d))
        print()
    d = tng50()
    m = (d["MS"] > 9.5) & (d["MS"] < 11.0) & (d["MH"] > 10.5)
    rows.append(row("TNG50", 0.0, d["BH"][m], d["MS"][m], d["MH"][m]))
    with open(os.path.join(OUT, "time_evolution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n-> {os.path.join(OUT,'time_evolution.csv')}")
