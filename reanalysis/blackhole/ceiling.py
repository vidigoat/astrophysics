"""The ceiling on black-hole mass at fixed halo mass (Sect. 8 of the paper).

Three measurements:

  (a) the distribution of log M_BH in a fixed halo-mass bin at each epoch, and how its
      90th percentile moves relative to z = 0 (Fig. 8);
  (b) the same envelope fitted across the full range of halo masses, so that its
      zero-point drift can be compared with the halo binding energy, which grows as
      E(z)^(2/3) at fixed mass;
  (c) the fraction of EAGLE galaxies that have reached the ceiling, followed along their
      main progenitor branches and binned by the halo mass they have at each epoch.

Writes results/ceiling.csv.
"""
import os, sys, csv
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_simba import load_simba
from common import window

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(ROOT, "Data", "eagle_v3")
VAR = os.path.join(ROOT, "Data", "simba_variants")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)

OM, OL = 0.307, 0.693
E = lambda z: np.sqrt(OM * (1 + z) ** 3 + OL)          # H(z)/H0
EAGLE_SNAP = {28: 0.0, 23: 0.5, 19: 1.0, 15: 2.0, 12: 3.0}
SIMBA_SNAP = {151: 0.0, 125: 0.5, 105: 1.0, 78: 2.0, 62: 3.0}
BIN = (12.0, 12.5)                                      # the bin shown in Fig. 8


def eagle_epoch(snap):
    f = os.path.join(RAW, "RefL0100N1504_z0.csv") if snap == 28 else os.path.join(RAW, f"RefL0100N1504_snap{snap}.csv")
    d = pd.read_csv(f)
    d = d[(d.BlackHoleMass > 0) & (d.Group_M_Crit200 > 10 ** 11.0) & (d.Mass_Star > 1e9)]
    return np.log10(d.BlackHoleMass.values), np.log10(d.Group_M_Crit200.values)


def simba_epoch(snap):
    d, _ = load_simba(os.path.join(VAR, f"m50n512_s50_{snap:03d}.hdf5"))
    m = (d["BH_MASS"] > 0) & (d["STELLAR_MASS"] > 9.0)
    return d["BH_MASS"][m], d["M200"][m]


def envelope_zeropoint(bh, mh, q=90, lo=11.5, hi=14.0, step=0.25, ref=12.5, nmin=20):
    """Fit the q-th percentile of log M_BH across halo-mass bins and return its value at
    log M200 = ref, so that epochs can be compared at a common halo mass."""
    xs, ys = [], []
    for a in np.arange(lo, hi, step):
        k = (mh >= a) & (mh < a + step)
        if k.sum() >= nmin:
            xs.append(a + step / 2); ys.append(np.percentile(bh[k], q))
    if len(xs) < 3:
        return np.nan, np.nan
    p = np.polyfit(xs, ys, 1)
    return np.polyval(p, ref), p[0]


if __name__ == "__main__":
    rows = []
    print(f"(a) percentiles of log M_BH for {BIN[0]} < log M200 < {BIN[1]}, relative to z = 0\n")
    for code, loader, SNAP in [("EAGLE", eagle_epoch, EAGLE_SNAP), ("SIMBA", simba_epoch, SIMBA_SNAP)]:
        ref = None
        for snap, z in sorted(SNAP.items(), key=lambda kv: kv[1]):
            bh, mh = loader(snap); k = (mh >= BIN[0]) & (mh < BIN[1])
            p10, p50, p90 = np.percentile(bh[k], [10, 50, 90])
            if ref is None: ref = (p10, p50, p90)
            print(f"  {code:6s} z={z:<4.1f} N={k.sum():5d}   d(90th)={p90-ref[2]:+.2f}   "
                  f"d(median)={p50-ref[1]:+.2f}   d(10th)={p10-ref[0]:+.2f}   "
                  f"binding-energy prediction {2/3*np.log10(E(z)):+.2f}")
            rows.append(dict(part="bin_percentiles", code=code, z=z, N=int(k.sum()),
                             d_p90=p90-ref[2], d_p50=p50-ref[1], d_p10=p10-ref[0],
                             predicted=2/3*np.log10(E(z))))
        print()
    print("(b) 90th-percentile envelope fitted over all halo masses, evaluated at log M200 = 12.5\n")
    for code, loader, SNAP in [("EAGLE", eagle_epoch, EAGLE_SNAP), ("SIMBA", simba_epoch, SIMBA_SNAP)]:
        ref = None
        for snap, z in sorted(SNAP.items(), key=lambda kv: kv[1]):
            bh, mh = loader(snap); v, slope = envelope_zeropoint(bh, mh)
            if ref is None: ref = v
            print(f"  {code:6s} z={z:<4.1f}  envelope slope={slope:.2f}  zero-point drift={v-ref:+.2f}   "
                  f"predicted {2/3*np.log10(E(z)):+.2f}")
            rows.append(dict(part="envelope_fit", code=code, z=z, N=0, d_p90=v-ref,
                             d_p50=np.nan, d_p10=np.nan, predicted=2/3*np.log10(E(z))))
        print()
    print("(c) fraction of EAGLE galaxies within 0.3 dex of the z=0 envelope scaled by E(z)^(2/3),\n"
          "    binned by the halo mass they have at that epoch\n")
    t = pd.read_csv(os.path.join(RAW, "RefL0100N1504_temporal.csv"))
    t = t[(t.BlackHoleMass > 0) & (t.Group_M_Crit200 > 0) & (t.Mass_Star > 0)]
    z0 = t[t.SnapNum == 28]
    z0 = z0[(z0.Mass_Star > 10 ** 9.5) & (z0.Group_M_Crit200 > 10 ** 11.5)]
    xs, ys = [], []
    mh0, bh0 = np.log10(z0.Group_M_Crit200.values), np.log10(z0.BlackHoleMass.values)
    for a in np.arange(11.5, 14.0, 0.25):
        k = (mh0 >= a) & (mh0 < a + 0.25)
        if k.sum() >= 20: xs.append(a + 0.125); ys.append(np.percentile(bh0[k], 90))
    p = np.polyfit(xs, ys, 1)
    print(f"    z=0 envelope: log M_BH = {p[0]:.2f} log M200 + {p[1]:.2f}\n")
    ZT = {28: 0.0, 19: 1.0, 15: 2.0, 12: 3.0}
    t["r"] = np.log10(t.BlackHoleMass) - (np.polyval(p, np.log10(t.Group_M_Crit200))
                                          + 2 / 3 * np.log10(E(t.SnapNum.map(ZT))))
    # follow only galaxies that sit in the paper's window at z = 0
    ms0 = t[t.SnapNum == 28].set_index("RootID")["Mass_Star"]
    keep = set(ms0[(ms0 > 10 ** 9.5) & (ms0 < 1e11)].index)
    t = t[t.RootID.isin(keep)]
    print(f"    following {len(keep)} main branches whose z=0 stellar mass is in the window\n")
    for lo, hi in [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0), (13.0, 13.5)]:
        line = f"    log M200(z) in [{lo},{hi}): "
        for snap in (12, 15, 19, 28):
            s_ = t[(t.SnapNum == snap) & (np.log10(t.Group_M_Crit200) >= lo) & (np.log10(t.Group_M_Crit200) < hi)]
            if len(s_) >= 15:
                f_at = float(np.mean(s_["r"] > -0.3))
                line += f" z={ZT[snap]:.0f}: {f_at:.2f} (N={len(s_)})  "
                rows.append(dict(part="arrival", code="EAGLE", z=ZT[snap], N=len(s_), d_p90=f_at,
                                 d_p50=lo, d_p10=hi, predicted=np.nan))
            else:
                line += f" z={ZT[snap]:.0f}:   --        "
        print(line)
    with open(os.path.join(OUT, "ceiling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n-> {os.path.join(OUT,'ceiling.csv')}")
