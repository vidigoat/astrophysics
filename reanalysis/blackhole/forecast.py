"""Observational forecast: what sample would decide between the halo-regulated
(EAGLE) and galaxy-regulated (TNG50/SIMBA) pictures?

For each code take seeded centrals with log M* > 10 (where dynamical black-hole
masses exist), add Gaussian errors: 0.10 dex on log M*, 0.30 dex on log M_BH,
sigma_h on log M_h.  Draw N galaxies, measure the halo exponent b in
log M_BH = a log M* + b log M_h + c, repeat 2000 times.  Report the median and
16-84 range of b per code, and the probability that a single draw from EAGLE
gives b larger than a single draw from TNG50 (the "separation").
Writes results/forecast.csv
"""

import os, sys, csv, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ols

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
rng = np.random.default_rng(2026)


def load(code):
    g = {
        k: np.asarray(v, float)
        for k, v in pickle.load(
            open(os.path.join(ROOT, "reanalysis", "data", f"{code}_full.pkl"), "rb")
        ).items()
    }
    m = (g["BH_SEEDED"] == 1) & (g["STELLAR_MASS"] > 10.0) & (g["STELLAR_MASS"] < 11.5)
    return g["STELLAR_MASS"][m], g["DM_MASS"][m], g["BH_MASS"][m]


D = {c: load(c) for c in ["TNG50", "EAGLE", "SIMBA"]}
for c, (ms, mh, bh) in D.items():
    print(c, "N available", len(ms))

rows = []
for sig_h in [0.0, 0.1, 0.2, 0.3, 0.4]:
    for N in [30, 60, 100, 200, 400]:
        draws = {}
        for c, (ms, mh, bh) in D.items():
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(ms), N)
                x1 = ms[i] + rng.normal(0, 0.10, N)
                x2 = mh[i] + rng.normal(0, sig_h, N)
                y = bh[i] + rng.normal(0, 0.30, N)
                bs.append(ols(y, x1, x2)[0][1])
            draws[c] = np.array(bs)
        sep = float((draws["EAGLE"] > draws["TNG50"]).mean())
        # 3-sigma style criterion: fraction of EAGLE draws above the 99.7th pct of TNG draws
        thr = np.percentile(draws["TNG50"], 99.7)
        p3 = float((draws["EAGLE"] > thr).mean())
        r = dict(sigma_h=sig_h, N=N, sep_EAGLE_gt_TNG=sep, p_EAGLE_above_TNG_3sig=p3)
        for c in D:
            r[f"b_{c}_med"] = float(np.median(draws[c]))
            r[f"b_{c}_16"] = float(np.percentile(draws[c], 16))
            r[f"b_{c}_84"] = float(np.percentile(draws[c], 84))
        rows.append(r)
        print(
            f"sigma_h={sig_h:.1f} N={N:3d}  b: TNG {r['b_TNG50_med']:+.2f} [{r['b_TNG50_16']:+.2f},{r['b_TNG50_84']:+.2f}]  "
            f"EAGLE {r['b_EAGLE_med']:+.2f} [{r['b_EAGLE_16']:+.2f},{r['b_EAGLE_84']:+.2f}]  SIMBA {r['b_SIMBA_med']:+.2f}   "
            f"P(EAGLE>TNG)={sep:.3f}  P(EAGLE beyond TNG 3sig)={p3:.2f}"
        )
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "forecast.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
