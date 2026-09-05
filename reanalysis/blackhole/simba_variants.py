"""Which mass sets the black hole, as a function of which feedback modules are on.

SIMBA m50n512 public variants (Dave et al. 2019, Sect. 2.7):
  s50      full physics
  s50nox   no X-ray feedback            (winds + jets)
  s50nojet no jet feedback              (winds only; X-ray requires jets, so off too)
  s50noagn no AGN feedback at all       (stellar feedback only)
  s50nofb  no feedback at all           (no stellar winds, no AGN)

For each run, in the common window 9.5 < log M* < 11, seeded centrals:
  log M_BH = a log M* + b log M200c + c
plus the partial correlation r(M_BH, M200 | M*), the mirror r(M_BH, M* | M200),
and the same split into star-forming and quenched (log sSFR = -11).

Writes results/simba_variants.csv
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from load_simba import load_simba
from common import ols, pcorr, resid, window

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAR = os.path.join(ROOT, "Data", "simba_variants")
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

RUNS = [
    ("s50nofb", "none"),
    ("s50noagn", "stellar"),
    ("s50nojet", "stellar+wind"),
    ("s50nox", "stellar+wind+jet"),
    ("s50", "stellar+wind+jet+xray"),
]

rows = []
for run, modules in RUNS:
    d, meta = load_simba(os.path.join(VAR, f"m50n512_{run}_151.hdf5"))
    w = window(d) & (d["M200"] > 10.5)
    ssfr = np.where(np.isfinite(d["SSFR"]), d["SSFR"], -99.0)
    for tag, m in [("all", w), ("sf", w & (ssfr > -11)), ("q", w & (ssfr <= -11))]:
        if m.sum() < 40:
            continue
        y, ms, mh = d["BH_MASS"][m], d["STELLAR_MASS"][m], d["M200"][m]
        (a, b), (ea, eb) = ols(y, ms, mh, boot=1000, seed=1)
        r_h, e_h = pcorr(y, mh, [ms], boot=500, seed=2)
        r_s, e_s = pcorr(y, ms, [mh], boot=500, seed=3)
        rows.append(
            dict(
                run=run,
                modules=modules,
                sample=tag,
                N=int(m.sum()),
                a_star=a,
                e_star=ea,
                b_halo=b,
                e_halo=eb,
                r_halo_given_star=r_h,
                e_r_halo=e_h,
                r_star_given_halo=r_s,
                e_r_star=e_s,
                sd_given_star=float(np.std(resid(y, [ms]))),
                sd_given_halo=float(np.std(resid(y, [mh]))),
                f_quenched=float((ssfr[m] <= -11).mean()),
                mean_logMstar=float(ms.mean()),
                mean_logM200=float(mh.mean()),
                mean_logMBH=float(y.mean()),
            )
        )
        print(
            f"{run:9s} {tag:3s} N={m.sum():5d}  M*^{a:+.2f}±{ea:.2f} M200^{b:+.2f}±{eb:.2f}  "
            f"r(BH,Mh|M*)={r_h:+.3f}±{e_h:.3f}  r(BH,M*|Mh)={r_s:+.3f}±{e_s:.3f}"
        )

import csv

with open(os.path.join(OUT, "simba_variants.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print("->", os.path.join(OUT, "simba_variants.csv"))
