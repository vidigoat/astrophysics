"""
Within-code null test for the cross-code causal-structure comparison.

The headline result (only a minority of causal directions are recovered
identically by all three simulation codes) is only meaningful if the cross-code
disagreement exceeds the disagreement between two independent draws of the SAME
code. This script draws two independent N=10,000 subsamples of each simulation,
runs the same FCIT consensus on each, and reports the within-code
direction-agreement fraction (over edges present in both draws), to be compared
with the cross-code agreement. It also reports the pairwise cross-code agreement
for the same denominator definition.

Output: Results/within_code_null.csv  + printed summary.
"""
import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
RESULTS = os.path.join(REPO, "Results")
ALPHA, PRES, ORI, NRUN, T, P, N = 0.01, 0.50, 0.60, 50, 7, 8, 10000
BNAME = {"DM_MASS": "halo_mass", "STELLAR_MASS": "stellar_mass", "GAS_MASS": "gas_mass",
         "BH_MASS": "bh_mass", "BARYONIC_MASS": "baryon_mass", "HALFMASS_RAD": "size",
         "VEL_DISP": "veldisp", "STAR_METALLICITY": "Z_star", "GAS_METALLICITY": "Z_gas",
         "PHOTOMETRIC_R": "r_mag", "PHOTOMETRIC_U": "u_mag", "COLOUR": "colour", "SFR": "sfr"}
_MARKS = ("<->", "-->", "<--", "o->", "<-o", "o-o")
_FLIP = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}


def _dir(m):
    return "fwd" if m in ("-->", "o->") else "rev" if m in ("<--", "<-o") else "und"


def zscore(x):
    x = np.asarray(x, float); s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)


def build(code, seed):
    d = pickle.load(open(os.path.join(DATA, f"{code.lower()}_final.pkl"), "rb"))
    out = {}
    for k, nm in BNAME.items():
        v = np.asarray(d[k], float)
        if nm in ("size", "sfr"):
            v = np.log10(np.maximum(v, 1e-3))
        out[nm] = v
    m = np.ones(len(out["stellar_mass"]), bool)
    for v in out.values():
        m &= np.isfinite(v)
    out = {k: zscore(v[m]) for k, v in out.items()}
    n = len(out["stellar_mass"])
    idx = np.random.RandomState(seed).choice(n, min(n, N), replace=False)
    return pd.DataFrame({k: v[idx] for k, v in out.items()})


def parse(g):
    edges, inb = [], False
    for raw in g.split("\n"):
        s = raw.strip()
        if s.startswith("Graph Edges"):
            inb = True; continue
        if not inb:
            continue
        if s.startswith("Graph "):
            break
        for m in _MARKS:
            if m in s:
                body = s.split(".", 1)[1].strip() if "." in s.split(m)[0] else s
                l, r = body.split(m, 1)
                a, b = l.strip(), r.strip()
                edges.append(((a, b), m) if a <= b else ((b, a), _FLIP[m]))
                break
    return edges


def consensus(df):
    pres, marks = Counter(), {}
    for _ in range(NRUN):
        s = ts.TetradSearch(df); s.set_verbose(False)
        s.use_basis_function_lrt(truncation_limit=T, alpha=ALPHA)
        s.use_basis_function_bic(truncation_limit=T, penalty_discount=P)
        s.run_fcit()
        for key, cm in parse(str(s.get_java())):
            pres[key] += 1
            marks.setdefault(key, Counter())[cm] += 1
    g = {}
    for key, c in pres.items():
        if c / NRUN < PRES:
            continue
        top, tn = marks[key].most_common(1)[0]
        g[key] = top if tn / c >= ORI else "o-o"
    return g


def agree(g1, g2):
    common = set(g1) & set(g2)
    same = sum(1 for k in common if _dir(g1[k]) == _dir(g2[k]))
    return len(common), same


def main():
    os.makedirs(RESULTS, exist_ok=True)
    codes = ["TNG50", "EAGLE", "SIMBA"]
    graphs = {(c, sd): consensus(build(c, sd)) for c in codes for sd in (1, 2)}
    rows = []
    print("\n=== WITHIN-CODE (two independent 10k draws of the SAME code) ===")
    for c in codes:
        n, s = agree(graphs[(c, 1)], graphs[(c, 2)])
        rows.append({"comparison": f"within_{c}", "common_edges": n,
                     "same_direction": s, "frac": round(s / n, 3) if n else 0})
        print(f"  {c}: {s}/{n} = {s/n:.2f} of common edges same direction")
    print("\n=== CROSS-CODE (draw 1 of each), pairwise ===")
    for i in range(3):
        for j in range(i + 1, 3):
            n, s = agree(graphs[(codes[i], 1)], graphs[(codes[j], 1)])
            rows.append({"comparison": f"cross_{codes[i]}_{codes[j]}", "common_edges": n,
                         "same_direction": s, "frac": round(s / n, 3) if n else 0})
            print(f"  {codes[i]}-{codes[j]}: {s}/{n} = {s/n:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "within_code_null.csv"), index=False)
    wc = np.mean([r["frac"] for r in rows if r["comparison"].startswith("within")])
    cc = np.mean([r["frac"] for r in rows if r["comparison"].startswith("cross")])
    print(f"\nMEAN within-code agreement: {wc:.2f}   MEAN cross-code agreement: {cc:.2f}")


if __name__ == "__main__":
    main()
