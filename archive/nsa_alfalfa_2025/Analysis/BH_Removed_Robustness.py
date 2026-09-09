import os, pickle
from collections import Counter
import numpy as np, pandas as pd
import pytetrad.tools.TetradSearch as ts
import os as _os

REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = REPO + "/Data"
ALPHA, PRES, ORI, NRUN, T, P, N = 0.01, 0.50, 0.60, 40, 7, 8, 10000
CODES = ["TNG50", "EAGLE", "SIMBA"]
BN = {
    "DM_MASS": "halo_mass",
    "STELLAR_MASS": "stellar_mass",
    "GAS_MASS": "gas_mass",
    "BARYONIC_MASS": "baryon_mass",
    "HALFMASS_RAD": "size",
    "VEL_DISP": "veldisp",
    "STAR_METALLICITY": "Z_star",
    "GAS_METALLICITY": "Z_gas",
    "PHOTOMETRIC_R": "r_mag",
    "PHOTOMETRIC_U": "u_mag",
    "COLOUR": "colour",
    "SFR": "sfr",
}  # BH_MASS dropped
M = ("<->", "-->", "<--", "o->", "<-o", "o-o")
FL = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}
dr = lambda m: "fwd" if m in ("-->", "o->") else "rev" if m in ("<--", "<-o") else "und"


def zc(x):
    x = np.asarray(x, float)
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1)


def load(c):
    d = pickle.load(open(f"{DATA}/{c.lower()}_final.pkl", "rb"))
    o = {}
    for k, nm in BN.items():
        v = np.asarray(d[k], float)
        if nm in ("size", "sfr"):
            v = np.log10(np.maximum(v, 1e-3))
        o[nm] = v
    m = np.ones(len(o["stellar_mass"]), bool)
    for v in o.values():
        m &= np.isfinite(v)
    o = {k: zc(v[m]) for k, v in o.items()}
    return o, len(o["stellar_mass"])


def parse(g):
    e = []
    inb = False
    for raw in g.split("\n"):
        s = raw.strip()
        if s.startswith("Graph Edges"):
            inb = True
            continue
        if not inb:
            continue
        if s.startswith("Graph "):
            break
        for mk in M:
            if mk in s:
                b = s.split(".", 1)[1].strip() if "." in s.split(mk)[0] else s
                l, r = b.split(mk, 1)
                a, bb = l.strip(), r.strip()
                e.append(((a, bb), mk) if a <= bb else ((bb, a), FL[mk]))
                break
    return e


def cons(df):
    pr, mk = Counter(), {}
    for _ in range(NRUN):
        S = ts.TetradSearch(df)
        S.set_verbose(False)
        S.use_basis_function_lrt(truncation_limit=T, alpha=ALPHA)
        S.use_basis_function_bic(truncation_limit=T, penalty_discount=P)
        S.run_fcit()
        for k, c in parse(str(S.get_java())):
            pr[k] += 1
            mk.setdefault(k, Counter())[c] += 1
    g = {}
    for k, c in pr.items():
        if c / NRUN < PRES:
            continue
        top, tn = mk[k].most_common(1)[0]
        g[k] = top if tn / c >= ORI else "o-o"
    return g


std = {c: load(c) for c in CODES}
g = {
    c: cons(
        pd.DataFrame(
            {k: v[np.random.RandomState(42).choice(n, N, replace=False)] for k, v in std[c][0].items()}
        )
    )
    for c in CODES
    for n in [std[c][1]]
}
allp = set().union(*[set(x) for x in g.values()])
inv = sum(
    1
    for pr in allp
    if all(pr in g[c] for c in CODES)
    and len({dr(g[c][pr]) for c in CODES}) == 1
    and dr(g[CODES[0]][pr]) != "und"
)
p3 = sum(1 for pr in allp if all(pr in g[c] for c in CODES))
print(
    f"BH-REMOVED: per-code edges {[len(g[c]) for c in CODES]} union={len(allp)} present-in-3={p3} invariant={inv}"
)
print("DONE")
