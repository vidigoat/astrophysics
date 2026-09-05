"""Headline three-code table with consistent definitions.

  black-hole mass : subgrid mass in every code (TNG SubhaloBHMass, EAGLE BlackHoleMass,
                    SIMBA Caesar masses.bh)
  halo mass       : TNG50 subhalo DM mass (no FOF M200 in the catalogue we hold),
                    EAGLE Group_M_Crit200, SIMBA m200c
  sample          : central, seeded, 9.5 < log M* < 11.0; split by log sSFR = -11
Writes results/headline.csv
"""

import os, sys, csv, pickle
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ols, pcorr, resid, scatter_reduction
from load_simba import load_simba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def tng50():
    g = {
        k: np.asarray(v, float)
        for k, v in pickle.load(
            open(os.path.join(ROOT, "reanalysis", "data", "TNG50_full.pkl"), "rb")
        ).items()
    }
    m = g["BH_SEEDED"] == 1
    ssfr = np.where(np.isfinite(g["SSFR"]), g["SSFR"], -99.0)
    return dict(
        MS=g["STELLAR_MASS"][m],
        MH=g["DM_MASS"][m],
        BH=g["BH_MASS"][m],
        SSFR=ssfr[m],
        SIG=np.log10(np.maximum(g["VEL_DISP"][m], 1e-3)),
    )


def eagle():
    e = pd.read_csv(os.path.join(ROOT, "Data", "eagle_v3", "RefL0100N1504_z0.csv"))
    bh = e["BlackHoleMass"].to_numpy(float)
    m200 = e["Group_M_Crit200"].to_numpy(float)
    ok = (bh > 0) & (m200 > 0)
    with np.errstate(divide="ignore"):
        ssfr = np.log10(e["SFR"].to_numpy(float)) - np.log10(e["Mass_Star"].to_numpy(float))
    ssfr = np.where(np.isfinite(ssfr), ssfr, -99.0)
    return dict(
        MS=np.log10(e["Mass_Star"].to_numpy(float))[ok],
        MH=np.log10(m200)[ok],
        BH=np.log10(bh)[ok],
        SSFR=ssfr[ok],
        SIG=np.log10(np.maximum(e["VelDisp"].to_numpy(float), 1e-3))[ok],
    )


def simba():
    d, _ = load_simba(os.path.join(ROOT, "Data", "simba_m100n1024_151.hdf5"))
    ssfr = np.where(np.isfinite(d["SSFR"]), d["SSFR"], -99.0)
    return dict(MS=d["STELLAR_MASS"], MH=d["M200"], BH=d["BH_MASS"], SSFR=ssfr, SIG=d["LOG_SIGMA"])


rows = []
for code, d in [("TNG50", tng50()), ("EAGLE", eagle()), ("SIMBA", simba())]:
    w = (d["MS"] > 9.5) & (d["MS"] < 11.0) & (d["MH"] > 10.5) & np.isfinite(d["SIG"])
    for tag, m in [("all", w), ("sf", w & (d["SSFR"] > -11)), ("q", w & (d["SSFR"] <= -11))]:
        y, ms, mh, sg = d["BH"][m], d["MS"][m], d["MH"][m], d["SIG"][m]
        (a, b), (ea, eb) = ols(y, ms, mh, boot=1000, seed=1)
        rh, erh = pcorr(y, mh, [ms], boot=500, seed=2)
        rs, ers = pcorr(y, ms, [mh], boot=500, seed=3)
        rows.append(
            dict(
                code=code,
                sample=tag,
                N=int(m.sum()),
                a_star=a,
                e_star=ea,
                b_halo=b,
                e_halo=eb,
                r_halo_given_star=rh,
                e_r_halo=erh,
                r_star_given_halo=rs,
                e_r_star=ers,
                raw_r_halo=float(np.corrcoef(y, mh)[0, 1]),
                raw_r_star=float(np.corrcoef(y, ms)[0, 1]),
                sd_given_star=float(np.std(resid(y, [ms]))),
                sd_given_halo=float(np.std(resid(y, [mh]))),
                halo_adds_pct=scatter_reduction(y, [ms, sg], [mh]),
                galaxy_adds_pct=scatter_reduction(y, [mh], [ms, sg]),
                f_quenched=float((d["SSFR"][m] <= -11).mean()),
            )
        )
        r = rows[-1]
        print(
            f"{code:6s} {tag:3s} N={r['N']:5d}  M*^{a:+.2f}±{ea:.2f} Mh^{b:+.2f}±{eb:.2f}  r(BH,Mh|M*)={rh:+.3f}±{erh:.3f}  "
            f"r(BH,M*|Mh)={rs:+.3f}  halo adds {r['halo_adds_pct']:4.1f}%  galaxy adds {r['galaxy_adds_pct']:4.1f}%"
        )
with open(os.path.join(HERE, "results", "headline.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
