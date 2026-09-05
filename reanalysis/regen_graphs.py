"""Regenerate the consensus PAGs on CORRECTED data.
Fixes applied vs the submitted version:
  - TNG50 particle-type indices corrected (0=gas, 1=DM, 4=stars)
  - algebraic identities removed (BARYONIC_MASS = M*+Mgas, COLOUR = u-r)
  - censored M_BH removed (seeded black holes only)
  - definitionally incomparable variables dropped for cross-code work
"""

import os, io, contextlib, pickle
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
OUT = "/Users/vidigoat/astrophysics/reanalysis/results/"
os.makedirs(OUT, exist_ok=True)
V = [
    "STELLAR_MASS",
    "DM_MASS",
    "BH_MASS",
    "GAS_MASS",
    "LOG_SIGMA",
    "STAR_METALLICITY",
    "GAS_METALLICITY",
    "SSFR",
]
NRUNS = 50
PEN = 5
TRUNC = 7
ALPHA = 0.01


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    return {k: v[m] for k, v in d.items() if k in V}


def run(df, seed):
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    try:
        s.set_seed(seed)
    except Exception:
        pass
    s.use_basis_function_lrt(truncation_limit=TRUNC, alpha=ALPHA)
    s.use_basis_function_bic(truncation_limit=TRUNC, penalty_discount=PEN)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = []
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E.append((p[1], p[2], p[3]))
    return E


for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    df = pd.DataFrame(np.column_stack([d[c] for c in V]), columns=V)
    pres, marks = {}, {}
    for i in range(NRUNS):
        for a, m, b in run(df, 2000 + i):
            k = tuple(sorted([a, b]))
            pres[k] = pres.get(k, 0) + 1
            e = (
                f"{a} {m} {b}"
                if [a, b] == list(k)
                else f'{b} {m[::-1].translate(str.maketrans("<>","><"))} {a}'
            )
            marks.setdefault(k, {})
            marks[k][e] = marks[k].get(e, 0) + 1
    rows = []
    for k, c in sorted(pres.items(), key=lambda kv: -kv[1]):
        if c / NRUNS < 0.5:
            continue
        best, bc = max(marks[k].items(), key=lambda kv: kv[1])
        rows.append({"edge": best, "presence": round(c / NRUNS, 3), "orient_frac": round(bc / c, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT + f"consensus_corrected_{t}.csv", index=False)
    print(f"=== {t}  N={len(df)}  {len(out)} edges (50 runs, penalty {PEN}) ===")
    print(out.to_string(index=False))
    print()
