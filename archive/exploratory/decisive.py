"""Decisive test. Two remaining confounds:
  (A) mass-range mismatch  -> restrict all codes to a COMMON stellar mass window
  (B) SIMBA's SFR is 100-Myr averaged, hence smoother -> inject noise into SIMBA
      sSFR to degrade it to the noisiest code's main-sequence scatter.
If the three-way dissociation survives both, it is physics, not bookkeeping."""

import pickle, io, contextlib
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
rng = np.random.default_rng(11)


def graph_at(df, pen, trunc=7, alpha=0.01):
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    s.use_basis_function_lrt(truncation_limit=trunc, alpha=alpha)
    s.use_basis_function_bic(truncation_limit=trunc, penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = []
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E.append((p[1], p[2], p[3]))
    return E


def has(E, a, b):
    for x, m, y in E:
        if {x, y} == {a, b}:
            return True
    return False


raw = {t: pickle.load(open(D + f"{t}_clean.pkl", "rb")) for t in ["TNG50", "EAGLE", "SIMBA"]}
LO, HI = 9.5, 11.0
sub = {}
for t, d in raw.items():
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    sub[t] = {k: v[m] for k, v in d.items()}
    print(
        f"{t}: {m.sum()} galaxies in {LO} < log M* < {HI}   "
        f'(med M*={np.median(sub[t]["STELLAR_MASS"]):.2f}, med Mbh={np.median(sub[t]["BH_MASS"]):.2f})'
    )
N = min(len(s["STELLAR_MASS"]) for s in sub.values())
print(f"\nmatched N = {N}\n")


# noise needed to bring SIMBA sSFR down to the noisiest code
def ms_scatter(d):
    x, y = d["STELLAR_MASS"], d["SSFR"]
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.std(y - A @ b))


sc = {t: ms_scatter(sub[t]) for t in sub}
print("main-sequence scatter (dex):", {t: round(v, 3) for t, v in sc.items()})
worst = max(sc.values())
add = np.sqrt(max(worst**2 - sc["SIMBA"] ** 2, 0))
print(f"injecting {add:.3f} dex of noise into SIMBA sSFR to match the noisiest code\n")

PAIRS = [
    ("BH_MASS", "SSFR"),
    ("BH_MASS", "DM_MASS"),
    ("BH_MASS", "STELLAR_MASS"),
    ("BH_MASS", "GAS_MASS"),
    ("BH_MASS", "VEL_DISP"),
]
NB = 30
for pen in [3, 5]:
    print("=" * 84)
    print(f"COMMON MASS WINDOW {LO}-{HI}, matched N={N}, {NB} bootstraps, penalty={pen}")
    print(f"SIMBA-noisy = SIMBA with sSFR degraded to matched noise")
    print("=" * 84)
    cnt = {}
    for t in ["TNG50", "EAGLE", "SIMBA", "SIMBA-noisy"]:
        base = sub["SIMBA"] if t == "SIMBA-noisy" else sub[t]
        cols = list(base.keys())
        n = len(base["STELLAR_MASS"])
        cnt[t] = {p: 0 for p in PAIRS}
        for _ in range(NB):
            idx = rng.choice(n, N, replace=False)
            cur = {c: base[c][idx].copy() for c in cols}
            if t == "SIMBA-noisy":
                cur["SSFR"] = cur["SSFR"] + rng.normal(0, add, N)
            df = pd.DataFrame(np.column_stack([cur[c] for c in cols]), columns=cols)
            E = graph_at(df, pen)
            for p in PAIRS:
                if has(E, *p):
                    cnt[t][p] += 1
    print(f'{"link to M_BH":16s}' + "".join(f"{t:>14s}" for t in ["TNG50", "EAGLE", "SIMBA", "SIMBA-noisy"]))
    for p in PAIRS:
        lbl = (
            p[1]
            .replace("STELLAR_MASS", "M*")
            .replace("DM_MASS", "M_halo")
            .replace("GAS_MASS", "M_gas")
            .replace("VEL_DISP", "sigma")
        )
        print(
            f"{lbl:16s}"
            + "".join(f"{100*cnt[t][p]/NB:>13.0f}%" for t in ["TNG50", "EAGLE", "SIMBA", "SIMBA-noisy"])
        )
    print()
