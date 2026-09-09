"""Regenerate the two tables that still hold pre-correction numbers."""

import pickle, io, contextlib, itertools
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D = "reanalysis/data/"
rng = np.random.default_rng(777)
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


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    return {k: v[m] for k, v in d.items() if k in V}


def graph(d, idx, pen, nr):
    df = pd.DataFrame(
        np.column_stack([(d[c][idx] - d[c][idx].mean()) / d[c][idx].std() for c in V]), columns=V
    )
    pres, ori = {}, {}
    for i in range(nr):
        s = ts.TetradSearch(df)
        s.set_verbose(False)
        try:
            s.set_seed(5000 + i)
        except Exception:
            pass
        s.use_basis_function_lrt(truncation_limit=7, alpha=0.01)
        s.use_basis_function_bic(truncation_limit=7, penalty_discount=pen)
        with contextlib.redirect_stdout(io.StringIO()):
            s.run_fcit()
        for line in str(s.get_java()).split("\n"):
            p = line.strip().split()
            if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
                a, m_, b = p[1], p[2], p[3]
                k = tuple(sorted([a, b]))
                pres[k] = pres.get(k, 0) + 1
                head = a if m_.endswith(">") else (b if m_.startswith("<") else None)
                ori[(k, head)] = ori.get((k, head), 0) + 1
    out = {}
    for k, c in pres.items():
        if c / nr < 0.5:
            continue
        cands = {h: n for (kk, h), n in ori.items() if kk == k}
        best = max(cands, key=cands.get) if cands else None
        out[k] = best if (best is not None and cands[best] / c >= 0.6) else None
    return out


data = {t: load(t) for t in ["TNG50", "EAGLE", "SIMBA"]}
N = min(len(d["STELLAR_MASS"]) for d in data.values())

print("=" * 72)
print("TABLE: penalty dependence (corrected data, matched N=%d)" % N)
print("=" * 72)
print(f'{"p":>4} {"TNG50/EAGLE/SIMBA":>20} {"union":>7} {"in all 3":>9} {"same dir":>9}')
for pen in [3, 5, 8]:
    G = {t: graph(d, rng.choice(len(d["STELLAR_MASS"]), N, replace=False), pen, 20) for t, d in data.items()}
    u = set().union(*[set(g) for g in G.values()])
    all3 = [e for e in u if all(e in G[t] for t in G)]
    same = sum(1 for e in all3 if len({G[t][e] for t in G}) == 1 and None not in {G[t][e] for t in G})
    print(
        f'{pen:>4} {"/".join(str(len(G[t])) for t in ["TNG50","EAGLE","SIMBA"]):>20} '
        f"{len(u):>7} {len(all3):>9} {same:>9}"
    )

print()
print("=" * 72)
print("TABLE: within- vs cross-code agreement (corrected data)")
print("=" * 72)


def jac(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def orient(a, b):
    sh = [k for k in set(a) & set(b) if a[k] is not None and b[k] is not None]
    if not sh:
        return float("nan"), 0
    return sum(1 for k in sh if a[k] == b[k]) / len(sh), len(sh)


halves = {}
for t, d in data.items():
    n = len(d["STELLAR_MASS"])
    p = rng.permutation(n)
    h = n // 2
    halves[t] = (graph(d, p[:h], 5, 20), graph(d, p[h : 2 * h], 5, 20))
print(f'{"comparison":<22}{"skeleton (Jaccard)":>20}{"orientation agreement":>24}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    o, ns = orient(*halves[t])
    print(f'{"Within "+t:<22}{jac(*halves[t]):>20.2f}{f"{o:.2f} ({ns})":>24}')
for a, b in itertools.combinations(["TNG50", "EAGLE", "SIMBA"], 2):
    o, ns = orient(halves[a][0], halves[b][0])
    print(f'{a+" vs "+b:<22}{jac(halves[a][0],halves[b][0]):>20.2f}{f"{o:.2f} ({ns})":>24}')
