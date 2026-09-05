"""
Audit critical #2: the decisive within(0.72)-vs-cross(0.43) skeleton comparison
mixed N=5,000 (within, disjoint halves) with N=10,000 (cross). Re-do it at MATCHED
N=5,000: for each code draw two disjoint 5,000 halves; within = agree(half1,half2);
cross = agree(codeA-half1, codeB-half1). All quantities at identical N=5,000.
Reports skeleton (Jaccard) and orientation (edges both orient) with bootstrap CIs.
Output: Results/null_matchedN.csv + printed summary.
"""

import os, pickle
from collections import Counter
import numpy as np, pandas as pd
import pytetrad.tools.TetradSearch as ts

import os as _os

REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = REPO + "/Data"
RES = REPO + "/Results"
ALPHA, PRES, ORI, NRUN, T, P, NH = 0.01, 0.50, 0.60, 40, 7, 8, 5000
CODES = ["TNG50", "EAGLE", "SIMBA"]
BN = {
    "DM_MASS": "halo_mass",
    "STELLAR_MASS": "stellar_mass",
    "GAS_MASS": "gas_mass",
    "BH_MASS": "bh_mass",
    "BARYONIC_MASS": "baryon_mass",
    "HALFMASS_RAD": "size",
    "VEL_DISP": "veldisp",
    "STAR_METALLICITY": "Z_star",
    "GAS_METALLICITY": "Z_gas",
    "PHOTOMETRIC_R": "r_mag",
    "PHOTOMETRIC_U": "u_mag",
    "COLOUR": "colour",
    "SFR": "sfr",
}
_M = ("<->", "-->", "<--", "o->", "<-o", "o-o")
_FL = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}
_dir = lambda m: "fwd" if m in ("-->", "o->") else "rev" if m in ("<--", "<-o") else "und"


def zc(x):
    x = np.asarray(x, float)
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)


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


def frame(o, idx):
    return pd.DataFrame({k: v[idx] for k, v in o.items()})


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
        for mk in _M:
            if mk in s:
                b = s.split(".", 1)[1].strip() if "." in s.split(mk)[0] else s
                l, r = b.split(mk, 1)
                a, bb = l.strip(), r.strip()
                e.append(((a, bb), mk) if a <= bb else ((bb, a), _FL[mk]))
                break
    return e


def consensus(df):
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


def skel_jac(g1, g2):
    s1, s2 = set(g1), set(g2)
    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0


def orient_vec(g1, g2):
    o1 = {k: _dir(v) for k, v in g1.items() if _dir(v) != "und"}
    o2 = {k: _dir(v) for k, v in g2.items() if _dir(v) != "und"}
    both = sorted(set(o1) & set(o2))
    return np.array([1 if o1[k] == o2[k] else 0 for k in both])


def ci(v, B=4000):
    if len(v) == 0:
        return (float("nan"),) * 3
    rs = np.random.RandomState(7)
    f = v.mean()
    return f, *np.percentile([v[rs.randint(0, len(v), len(v))].mean() for _ in range(B)], [2.5, 97.5])


def main():
    std = {c: load(c) for c in CODES}
    half = {}
    for c in CODES:
        o, n = std[c]
        idx = np.random.RandomState(1).choice(n, 2 * NH, replace=False)
        half[(c, 1)] = consensus(frame(o, idx[:NH]))
        half[(c, 2)] = consensus(frame(o, idx[NH : 2 * NH]))
        print(f"  {c} halves done", flush=True)
    rows = []
    for c in CODES:
        j = skel_jac(half[(c, 1)], half[(c, 2)])
        v = orient_vec(half[(c, 1)], half[(c, 2)])
        f, lo, hi = ci(v)
        rows.append(
            dict(
                kind="within",
                comp=c,
                skel_jac=round(j, 3),
                n_or=len(v),
                orient=round(f, 3),
                lo=round(lo, 3),
                hi=round(hi, 3),
            )
        )
    for i in range(3):
        for k in range(i + 1, 3):
            a, b = CODES[i], CODES[k]
            j = skel_jac(half[(a, 1)], half[(b, 1)])
            v = orient_vec(half[(a, 1)], half[(b, 1)])
            f, lo, hi = ci(v)
            rows.append(
                dict(
                    kind="cross",
                    comp=f"{a}-{b}",
                    skel_jac=round(j, 3),
                    n_or=len(v),
                    orient=round(f, 3),
                    lo=round(lo, 3),
                    hi=round(hi, 3),
                )
            )
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/null_matchedN.csv", index=False)
    print("\n== MATCHED N=5,000 (all within and cross) ==")
    print(d.to_string(index=False))
    wj = d[d.kind == "within"].skel_jac.mean()
    cj = d[d.kind == "cross"].skel_jac.mean()
    wo = d[d.kind == "within"].orient.mean()
    co = d[d.kind == "cross"].orient.mean()
    print(f"\nSkeleton Jaccard  within {wj:.2f}  vs  cross {cj:.2f}")
    print(f"Orientation agree within {wo:.2f}  vs  cross {co:.2f}   (chance 0.50)")
    print("DONE")


if __name__ == "__main__":
    main()
