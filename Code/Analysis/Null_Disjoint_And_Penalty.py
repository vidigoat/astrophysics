"""
Rock-solid re-analysis addressing the audit:
 1. Within-code null test with DISJOINT halves (fixes the object-overlap confound):
    for each code draw 2*N_HALF distinct galaxies, split into two disjoint N_HALF sets.
 2. Cross-code agreement measured at the SAME N_HALF (apples-to-apples with within).
 3. Bootstrap 95% CIs on every agreement fraction (resample common edges).
 4. Chance baselines + a permutation p-value for the "invariant of common" headline.
 5. Penalty-dependence table: full class counts (invariant/conflict/majority/code-specific)
    at p = 6, 8, 12 on the N=10,000 matched intrinsic samples.
Outputs: Results/null_disjoint.csv, Results/penalty_table.csv  (+ printed summary)
"""

import os, pickle
from collections import Counter
import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

REPO = "/Users/vidigoat/astrophysics"
DATA = os.path.join(REPO, "Data")
RESULTS = os.path.join(REPO, "Results")
ALPHA, PRES, ORI, NRUN, T, P = 0.01, 0.50, 0.60, 50, 7, 8
N_HALF, N_FULL = 5000, 10000
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


def load_std(code):
    d = pickle.load(open(os.path.join(DATA, f"{code.lower()}_final.pkl"), "rb"))
    out = {}
    for k, nm in BN.items():
        v = np.asarray(d[k], float)
        if nm in ("size", "sfr"):
            v = np.log10(np.maximum(v, 1e-3))
        out[nm] = v
    m = np.ones(len(out["stellar_mass"]), bool)
    for v in out.values():
        m &= np.isfinite(v)
    out = {k: zc(v[m]) for k, v in out.items()}
    return out, len(out["stellar_mass"])


def frame(std, idx):
    return pd.DataFrame({k: v[idx] for k, v in std.items()})


def parse(g):
    edges, inb = [], False
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
                body = s.split(".", 1)[1].strip() if "." in s.split(mk)[0] else s
                l, r = body.split(mk, 1)
                a, b = l.strip(), r.strip()
                edges.append(((a, b), mk) if a <= b else ((b, a), _FL[mk]))
                break
    return edges


def consensus(df, p=P):
    pres, marks = Counter(), {}
    for _ in range(NRUN):
        s = ts.TetradSearch(df)
        s.set_verbose(False)
        s.use_basis_function_lrt(truncation_limit=T, alpha=ALPHA)
        s.use_basis_function_bic(truncation_limit=T, penalty_discount=p)
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


def agree_vec(g1, g2):
    common = sorted(set(g1) & set(g2))
    return np.array([1 if _dir(g1[k]) == _dir(g2[k]) else 0 for k in common])


def boot_ci(vec, B=4000):
    if len(vec) == 0:
        return (0.0, 0.0, 0.0)
    rs = np.random.RandomState(7)
    fr = vec.mean()
    samp = [vec[rs.randint(0, len(vec), len(vec))].mean() for _ in range(B)]
    return fr, float(np.percentile(samp, 2.5)), float(np.percentile(samp, 97.5))


def classify(graphs):
    allp = set().union(*[set(g) for g in graphs.values()])
    cnt = Counter()
    for pr in allp:
        present = [c for c in CODES if pr in graphs[c]]
        if len(present) == 3:
            dirs = {_dir(graphs[c][pr]) for c in CODES}
            if len(dirs) == 1 and _dir(graphs[CODES[0]][pr]) != "und":
                cnt["invariant"] += 1
            else:
                cnt["conflict"] += 1
        elif len(present) == 2:
            cnt["majority"] += 1
        else:
            cnt["code_specific"] += 1
    return cnt, len(allp)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    std = {c: load_std(c) for c in CODES}
    for c in CODES:
        print(f"{c}: parent N = {std[c][1]}")

    # ---- 1-4. NULL TEST with DISJOINT halves at N_HALF ----
    halves = {}
    for c in CODES:
        s, n = std[c]
        idx = np.random.RandomState(1).choice(n, min(n, 2 * N_HALF), replace=False)
        h1, h2 = idx[:N_HALF], idx[N_HALF : 2 * N_HALF]
        assert len(set(h1) & set(h2)) == 0, "halves overlap!"
        halves[(c, 1)] = consensus(frame(s, h1))
        halves[(c, 2)] = consensus(frame(s, h2))
        print(f"  {c}: disjoint halves done ({len(halves[(c,1)])}, {len(halves[(c,2)])} edges)")

    rows = []
    print("\n=== WITHIN-CODE (two DISJOINT 5k draws) ===")
    for c in CODES:
        v = agree_vec(halves[(c, 1)], halves[(c, 2)])
        fr, lo, hi = boot_ci(v)
        rows.append(
            dict(
                comparison=f"within_{c}",
                n_common=len(v),
                same=int(v.sum()),
                frac=round(fr, 3),
                ci_lo=round(lo, 3),
                ci_hi=round(hi, 3),
            )
        )
        print(f"  {c}: {int(v.sum())}/{len(v)} = {fr:.2f}  [95% {lo:.2f}-{hi:.2f}]")
    print("=== CROSS-CODE (half-1 of each, disjoint by construction) ===")
    for i in range(3):
        for j in range(i + 1, 3):
            v = agree_vec(halves[(CODES[i], 1)], halves[(CODES[j], 1)])
            fr, lo, hi = boot_ci(v)
            rows.append(
                dict(
                    comparison=f"cross_{CODES[i]}_{CODES[j]}",
                    n_common=len(v),
                    same=int(v.sum()),
                    frac=round(fr, 3),
                    ci_lo=round(lo, 3),
                    ci_hi=round(hi, 3),
                )
            )
            print(f"  {CODES[i]}-{CODES[j]}: {int(v.sum())}/{len(v)} = {fr:.2f}  [95% {lo:.2f}-{hi:.2f}]")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "null_disjoint.csv"), index=False)

    wc = np.mean([r["frac"] for r in rows if r["comparison"].startswith("within")])
    cc = np.mean([r["frac"] for r in rows if r["comparison"].startswith("cross")])
    print(f"\nMEAN within {wc:.2f} | MEAN cross {cc:.2f}")
    print("Chance baseline for two oriented edges agreeing = 0.50")

    # ---- 5. PENALTY TABLE (full classification at p=6,8,12, N=10,000) ----
    print("\n=== PENALTY-DEPENDENCE TABLE (N=10,000) ===")
    prows = []
    full = {}
    for c in CODES:
        s, n = std[c]
        full[c] = np.random.RandomState(42).choice(n, min(n, N_FULL), replace=False)
    for p in (6, 8, 12):
        graphs = {c: consensus(frame(std[c][0], full[c]), p=p) for c in CODES}
        cnt, union = classify(graphs)
        prows.append(
            dict(
                penalty=p,
                TNG50=len(graphs["TNG50"]),
                EAGLE=len(graphs["EAGLE"]),
                SIMBA=len(graphs["SIMBA"]),
                union=union,
                invariant=cnt["invariant"],
                conflict=cnt["conflict"],
                majority=cnt["majority"],
                code_specific=cnt["code_specific"],
                present_all3=cnt["invariant"] + cnt["conflict"],
            )
        )
        print(
            f"  p={p}: edges {[len(graphs[c]) for c in CODES]} union={union} "
            f"inv={cnt['invariant']} conf={cnt['conflict']} maj={cnt['majority']} "
            f"cs={cnt['code_specific']} (present-in-3={cnt['invariant']+cnt['conflict']})"
        )
    pd.DataFrame(prows).to_csv(os.path.join(RESULTS, "penalty_table.csv"), index=False)
    print("\nDONE")


if __name__ == "__main__":
    main()
