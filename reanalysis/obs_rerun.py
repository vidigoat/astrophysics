"""Referee concern 3: trivial/definitional edges in the observational graphs.

The referee named "luminosity -> stellar mass <- mass-to-light ratio" as a
definitional edge. It is: regressing log M* on log(M/L) and log L_r returns
coefficients 1.003 and 0.996 with a residual scatter of 0.025 dex (R^2=0.998),
i.e. an identity, because the NSA stellar mass IS the product of the other two.
The same applies to BARYONIC_MASS in the ALFALFA-NSA set, which is built from
ELPETRO_MASS and logMH.

A search handed a quantity together with its own arguments recovers the
arithmetic and reports it as structure. This rerun removes the derived
quantities and keeps the measured ones.
"""

import pickle, io, contextlib, numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

DROP = {"nsa": ["ELPETRO_MTOL"], "alfalfa": ["ELPETRO_MTOL", "BARYONIC_MASS"]}
FILES = {"nsa": "../Data/nsa_final_10props.pkl", "alfalfa": "../Data/alfalfa_nsa_final_13props.pkl"}
PARAMS = {"nsa": (14, 50), "alfalfa": (7, 35)}


def run(tag, drop):
    d = dict(pickle.load(open(FILES[tag], "rb")))
    V = [k for k in sorted(d) if k not in drop]
    df = pd.DataFrame({k: np.asarray(d[k], float) for k in V})
    df = df[np.all(np.isfinite(df.values), axis=1)]
    tr, pen = PARAMS[tag]
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    s.use_basis_function_lrt(truncation_limit=tr, alpha=0.01)
    s.use_basis_function_bic(truncation_limit=tr, penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = []
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E.append((p[1], p[2], p[3]))
    return V, len(df), E


for tag in ["nsa", "alfalfa"]:
    for label, drop in [("WITH derived quantities", []), ("WITHOUT derived quantities", DROP[tag])]:
        V, n, E = run(tag, drop)
        nor = sum(1 for e in E if ">" in e[1] or "<" in e[1])
        print(f"\n=== {tag.upper()} : {label} ===")
        print(f"    {len(V)} variables, N={n}, {len(E)} edges, {nor} oriented")
        for a, m, b in sorted(E):
            flag = ""
            if {"ELPETRO_MASS"} & {a, b} and {"ELPETRO_MTOL", "ELPETRO_ABSMAG_R", "BARYONIC_MASS"} & {a, b}:
                flag = "   <-- DEFINITIONAL"
            print(f"      {a:20s} {m:4s} {b}{flag}")
