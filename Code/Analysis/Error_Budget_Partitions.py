"""
Audit critical #1/#2 fix: put an ERROR BUDGET on the flagship within-vs-cross
skeleton-Jaccard contrast (and the orientation fractions). Repeat the disjoint
5,000-galaxy split over B independent random partitions per code/pair, so we get
a DISTRIBUTION, not a single realization. Report median + 95% interval for within
and cross skeleton Jaccard and orientation agreement, and a permutation test that
within-code Jaccard > cross-code Jaccard.
Output: Results/error_budget.csv + printed summary.
"""
import os, pickle
from collections import Counter
import numpy as np, pandas as pd
import pytetrad.tools.TetradSearch as ts

import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); DATA = REPO + "/Data"; RES = REPO + "/Results"
ALPHA, PRES, ORI, NRUN, T, P, NH = 0.01, 0.50, 0.60, 6, 7, 8, 5000
B = 40  # random partitions per code/pair
CODES = ["TNG50", "EAGLE", "SIMBA"]
BN = {"DM_MASS": "halo_mass", "STELLAR_MASS": "stellar_mass", "GAS_MASS": "gas_mass", "BH_MASS": "bh_mass",
      "BARYONIC_MASS": "baryon_mass", "HALFMASS_RAD": "size", "VEL_DISP": "veldisp", "STAR_METALLICITY": "Z_star",
      "GAS_METALLICITY": "Z_gas", "PHOTOMETRIC_R": "r_mag", "PHOTOMETRIC_U": "u_mag", "COLOUR": "colour", "SFR": "sfr"}
_M = ("<->", "-->", "<--", "o->", "<-o", "o-o"); _FL = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}
_dir = lambda m: "fwd" if m in ("-->", "o->") else "rev" if m in ("<--", "<-o") else "und"


def zc(x): x = np.asarray(x, float); s = np.std(x); return (x - np.mean(x)) / (s if s > 0 else 1.0)
def load(c):
    d = pickle.load(open(f"{DATA}/{c.lower()}_final.pkl", "rb")); o = {}
    for k, nm in BN.items():
        v = np.asarray(d[k], float)
        if nm in ("size", "sfr"): v = np.log10(np.maximum(v, 1e-3))
        o[nm] = v
    m = np.ones(len(o["stellar_mass"]), bool)
    for v in o.values(): m &= np.isfinite(v)
    o = {k: zc(v[m]) for k, v in o.items()}; return o, len(o["stellar_mass"])
def frame(o, idx): return pd.DataFrame({k: v[idx] for k, v in o.items()})
def parse(g):
    e = []; inb = False
    for raw in g.split("\n"):
        s = raw.strip()
        if s.startswith("Graph Edges"): inb = True; continue
        if not inb: continue
        if s.startswith("Graph "): break
        for mk in _M:
            if mk in s:
                b = s.split(".", 1)[1].strip() if "." in s.split(mk)[0] else s
                l, r = b.split(mk, 1); a, bb = l.strip(), r.strip()
                e.append(((a, bb), mk) if a <= bb else ((bb, a), _FL[mk])); break
    return e
def consensus(df):
    pr, mk = Counter(), {}
    for _ in range(NRUN):
        S = ts.TetradSearch(df); S.set_verbose(False)
        S.use_basis_function_lrt(truncation_limit=T, alpha=ALPHA)
        S.use_basis_function_bic(truncation_limit=T, penalty_discount=P)
        S.run_fcit()
        for k, c in parse(str(S.get_java())): pr[k] += 1; mk.setdefault(k, Counter())[c] += 1
    g = {}
    for k, c in pr.items():
        if c / NRUN < PRES: continue
        top, tn = mk[k].most_common(1)[0]; g[k] = top if tn / c >= ORI else "o-o"
    return g
def skel_jac(g1, g2): s1, s2 = set(g1), set(g2); return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
def orient_frac(g1, g2):
    o1 = {k: _dir(v) for k, v in g1.items() if _dir(v) != "und"}
    o2 = {k: _dir(v) for k, v in g2.items() if _dir(v) != "und"}
    both = set(o1) & set(o2)
    return (sum(1 for k in both if o1[k] == o2[k]) / len(both)) if both else np.nan


def main():
    std = {c: load(c) for c in CODES}
    within_j = {c: [] for c in CODES}; within_o = {c: [] for c in CODES}
    cross_j, cross_o = [], []
    for b in range(B):
        halves = {}
        for c in CODES:
            o, n = std[c]; idx = np.random.RandomState(1000 + b).choice(n, 2 * NH, replace=False)
            halves[(c, 1)] = consensus(frame(o, idx[:NH])); halves[(c, 2)] = consensus(frame(o, idx[NH:2 * NH]))
            within_j[c].append(skel_jac(halves[(c, 1)], halves[(c, 2)]))
            within_o[c].append(orient_frac(halves[(c, 1)], halves[(c, 2)]))
        for i in range(3):
            for k in range(i + 1, 3):
                cross_j.append(skel_jac(halves[(CODES[i], 1)], halves[(CODES[k], 1)]))
                cross_o.append(orient_frac(halves[(CODES[i], 1)], halves[(CODES[k], 1)]))
        if (b + 1) % 10 == 0: print(f"  {b+1}/{B} partitions", flush=True)

    w_j = np.concatenate([within_j[c] for c in CODES]); c_j = np.array(cross_j)
    w_o = np.array([x for c in CODES for x in within_o[c] if not np.isnan(x)])
    c_o = np.array([x for x in cross_o if not np.isnan(x)])
    def med_ci(a): a = np.asarray(a); return np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

    rows = []
    for c in CODES:
        m, lo, hi = med_ci(within_j[c]); rows.append(dict(quant="within_skel", grp=c, median=round(m, 3), lo=round(lo, 3), hi=round(hi, 3)))
    m, lo, hi = med_ci(c_j); rows.append(dict(quant="cross_skel", grp="all", median=round(m, 3), lo=round(lo, 3), hi=round(hi, 3)))
    m, lo, hi = med_ci(w_o); rows.append(dict(quant="within_orient", grp="all", median=round(m, 3), lo=round(lo, 3), hi=round(hi, 3)))
    m, lo, hi = med_ci(c_o); rows.append(dict(quant="cross_orient", grp="all", median=round(m, 3), lo=round(lo, 3), hi=round(hi, 3)))
    pd.DataFrame(rows).to_csv(f"{RES}/error_budget.csv", index=False)

    # permutation test: is within-code Jaccard > cross-code Jaccard?
    obs = np.mean(w_j) - np.mean(c_j)
    pool = np.concatenate([w_j, c_j]); nW = len(w_j); rs = np.random.RandomState(3)
    perm = [np.mean((p := rs.permutation(pool))[:nW]) - np.mean(p[nW:]) for _ in range(10000)]
    pval = (np.sum(np.array(perm) >= obs) + 1) / (len(perm) + 1)

    print("\n" + "=" * 64)
    print(f"ERROR BUDGET over B={B} random partitions (t=7,p=8)")
    print(f"  SKELETON within (pooled): median {np.median(w_j):.2f}  95% [{np.percentile(w_j,2.5):.2f}-{np.percentile(w_j,97.5):.2f}]")
    print(f"  SKELETON cross  (pooled): median {np.median(c_j):.2f}  95% [{np.percentile(c_j,2.5):.2f}-{np.percentile(c_j,97.5):.2f}]")
    print(f"  permutation test within>cross skeleton: dObs={obs:.3f}  p={pval:.4f}")
    print(f"  ORIENTATION within: median {np.median(w_o):.2f}  95% [{np.percentile(w_o,2.5):.2f}-{np.percentile(w_o,97.5):.2f}]")
    print(f"  ORIENTATION cross : median {np.median(c_o):.2f}  95% [{np.percentile(c_o,2.5):.2f}-{np.percentile(c_o,97.5):.2f}]  (chance 0.50)")
    print("=" * 64)
    print("DONE")


if __name__ == "__main__":
    main()
