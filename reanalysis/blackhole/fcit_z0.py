"""z=0 FCIT graphs for the three codes on the eight comparable variables, consistent
definitions (subgrid BH mass everywhere, M200c for EAGLE/SIMBA), seeded star-forming
centrals (sSFR must be finite), matched N, halo-tier-0 prior.  Also the penalty scan
and the disjoint-halves null.  Writes results/fcit_z0.txt and fig_pags.png.
"""

import os, io, contextlib, pickle, sys
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts
from load_simba import load_simba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
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
TRUNC = 7


def tng():
    g = {
        k: np.asarray(v, float)
        for k, v in pickle.load(
            open(os.path.join(ROOT, "reanalysis", "data", "TNG50_full.pkl"), "rb")
        ).items()
    }
    g["LOG_SIGMA"] = np.log10(np.maximum(g["VEL_DISP"], 1e-3))
    m = (g["BH_SEEDED"] == 1) & np.isfinite(g["SSFR"])
    return {k: g[k][m] for k in V}


def eag():
    e = pd.read_csv(os.path.join(ROOT, "Data", "eagle_v3", "RefL0100N1504_z0.csv"))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = {
            "STELLAR_MASS": np.log10(e["Mass_Star"].to_numpy(float)),
            "DM_MASS": np.log10(e["Group_M_Crit200"].to_numpy(float)),
            "BH_MASS": np.log10(e["BlackHoleMass"].to_numpy(float)),
            "GAS_MASS": np.log10(e["Mass_Gas"].to_numpy(float)),
            "LOG_SIGMA": np.log10(e["VelDisp"].to_numpy(float)),
            "STAR_METALLICITY": np.log10(e["Stars_Metallicity"].to_numpy(float)),
            "GAS_METALLICITY": np.log10(e["SF_Metallicity"].to_numpy(float)),
            "SSFR": np.log10(e["SFR"].to_numpy(float)) - np.log10(e["Mass_Star"].to_numpy(float)),
        }
    m = np.ones(len(e), bool)
    for k in V:
        m &= np.isfinite(d[k])
    m &= d["STELLAR_MASS"] > 8.5
    return {k: d[k][m] for k in V}


def sim():
    d, _ = load_simba(os.path.join(ROOT, "Data", "simba_m100n1024_151.hdf5"))
    d = dict(d)
    d["DM_MASS"] = d["M200"]
    m = np.ones(len(d["STELLAR_MASS"]), bool)
    for k in V:
        m &= np.isfinite(d[k])
    return {k: d[k][m] for k in V}


def run(d, idx, pen, knowledge=True):
    df = pd.DataFrame({k: d[k][idx] for k in V})
    df = (df - df.mean()) / df.std()
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    if knowledge:
        s.add_to_tier(0, "DM_MASS")
        for v in V:
            if v != "DM_MASS":
                s.add_to_tier(1, v)
    s.use_basis_function_lrt(truncation_limit=TRUNC, alpha=0.01)
    s.use_basis_function_bic(truncation_limit=TRUNC, penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = []
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E.append((p[1], p[2], p[3]))
    return E


def canon(a, m, b):
    """(sorted pair, direction) with direction = which node gets an arrowhead, or None."""
    heads = set()
    if m.endswith(">"):
        heads.add(b)
    if m.startswith("<"):
        heads.add(a)
    return tuple(sorted([a, b])), frozenset(heads)


data = {"TNG50": tng(), "EAGLE": eag(), "SIMBA": sim()}
N = min(len(d["STELLAR_MASS"]) for d in data.values())
rng = np.random.default_rng(4242)
idx = {t: rng.permutation(len(d["STELLAR_MASS"]))[:N] for t, d in data.items()}
out = [f"matched N = {N} per code (seeded, star-forming centrals; subgrid BH; M200c for EAGLE/SIMBA)\n"]
graphs = {}
for pen in [3, 5, 8, 12]:
    G = {t: run(data[t], idx[t], pen) for t in data}
    if pen == 5:
        graphs = G
    sk = {t: {canon(*e)[0] for e in G[t]} for t in G}
    union = set.union(*sk.values())
    all3 = set.intersection(*sk.values())
    dirs = {t: {canon(*e)[0]: canon(*e)[1] for e in G[t]} for t in G}
    inv = [p for p in all3 if len({dirs[t][p] for t in G}) == 1 and len(dirs["TNG50"][p]) > 0]
    out.append(
        f"penalty {pen}: edges {[len(G[t]) for t in G]}  union {len(union)}  in all three {len(all3)}  invariant (same arrowhead in all three) {len(inv)}"
    )
    if pen == 5:
        for t in G:
            nor = sum(1 for e in G[t] if ">" in e[1] or "<" in e[1])
            out.append(f"\n{t}: {len(G[t])} edges, {nor} with a determined arrowhead")
            for a, m, b in sorted(G[t]):
                out.append(
                    f"    {a:18s} {m:4s} {b}" + ("   <-- BH-halo" if {a, b} == {"DM_MASS", "BH_MASS"} else "")
                )
# disjoint-halves null at penalty 5
half = N // 2
within, cross = [], []
for part in range(20):
    draws = {}
    for t, d in data.items():
        p = rng.permutation(len(d["STELLAR_MASS"]))
        draws[t] = (
            {canon(*e)[0] for e in run(d, p[:half], 5)},
            {canon(*e)[0] for e in run(d, p[half : 2 * half], 5)},
        )
    J = lambda A, B: len(A & B) / max(1, len(A | B))
    within += [J(*draws[t]) for t in draws]
    cross += [J(draws[a][0], draws[b][1]) for a in draws for b in draws if a < b]
out.append(
    f"\nDisjoint-halves null ({half} galaxies per draw, 20 partitions, penalty 5): within-code skeleton Jaccard median {np.median(within):.2f} "
    f"(16-84: {np.percentile(within,16):.2f}-{np.percentile(within,84):.2f}); cross-code median {np.median(cross):.2f} ({np.percentile(cross,16):.2f}-{np.percentile(cross,84):.2f})"
)
open(os.path.join(HERE, "results", "fcit_z0.txt"), "w").write("\n".join(out))
print("\n".join(out))
