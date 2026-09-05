"""Referee concern 2: astrophysical inductive bias.

Enforce the single domain prior the referee asked for -- dark-matter halo mass
cannot be a CHILD of a baryonic property -- as tiered background knowledge, and
compare the recovered graphs with and without it.

The prior is deliberately minimal. Tier 0 contains DM_MASS alone; tier 1
contains every baryonic and derived quantity. This forbids edges INTO the halo
without asserting that the halo causes anything, so it constrains the search
without presupposing the answer to the question the paper asks.
"""

import pickle, io, contextlib, numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D = "data/"
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
BARYONIC = [v for v in V if v != "DM_MASS"]
PEN = 5
TRUNC = 7


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    return {k: np.asarray(d[k], float) for k in V}


def run(d, idx, knowledge):
    df = pd.DataFrame({k: d[k][idx] for k in V})
    df = (df - df.mean()) / df.std()
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    if knowledge:
        s.add_to_tier(0, "DM_MASS")
        for v in BARYONIC:
            s.add_to_tier(1, v)
    s.use_basis_function_lrt(truncation_limit=TRUNC, alpha=0.01)
    s.use_basis_function_bic(truncation_limit=TRUNC, penalty_discount=PEN)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = []
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E.append((p[1], p[2], p[3]))
    return E


data = {t: load(t) for t in ["TNG50", "EAGLE", "SIMBA"]}
N = min(len(d["STELLAR_MASS"]) for d in data.values())
rng = np.random.default_rng(4242)
idx = {t: rng.permutation(len(d["STELLAR_MASS"]))[:N] for t, d in data.items()}
print(f"matched N = {N} per code, penalty {PEN}\n")

for kn in (False, True):
    tag = "WITH background knowledge" if kn else "WITHOUT background knowledge"
    print("=" * 66)
    print(tag)
    print("=" * 66)
    for t in data:
        E = run(data[t], idx[t], kn)
        nor = sum(1 for e in E if ">" in e[1] or "<" in e[1])
        print(f"\n{t}: {len(E)} edges, {nor} with a determined arrowhead")
        for a, m, b in sorted(E):
            star = " <-- halo/BH" if {"DM_MASS", "BH_MASS"} <= {a, b} else ""
            print(f"    {a:18s} {m:4s} {b}{star}")
