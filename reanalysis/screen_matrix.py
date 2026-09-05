"""Full screening matrix for M_BH.
For each candidate driver X, report r(M_BH,X) and r(M_BH,X | all other drivers).
The driver whose partial SURVIVES is the fundamental one; those that collapse are
projections of it."""

import pickle, numpy as np

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
LO, HI = 9.5, 11.0
DRV = ["STELLAR_MASS", "LOG_SIGMA", "DM_MASS", "GAS_MASS", "HALFMASS_RAD"]
NICE = {
    "STELLAR_MASS": "M*",
    "LOG_SIGMA": "sigma",
    "DM_MASS": "M_halo",
    "GAS_MASS": "M_gas",
    "HALFMASS_RAD": "R_half",
}


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z)) if Z else np.ones((len(v), 1))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    d = {k: v[m] for k, v in d.items()}
    bh = d["BH_MASS"]
    print(f"\n===== {t}   N={len(bh)}   ({LO} < log M* < {HI}) =====")
    print(f'{"driver":10s}{"r(Mbh,X)":>12s}{"r(Mbh,X | rest)":>18s}{"verdict":>14s}')
    for x in DRV:
        rest = [d[k] for k in DRV if k != x]
        a, b_ = pc(bh, d[x], []), pc(bh, d[x], rest)
        keep = abs(b_) / max(abs(a), 1e-9)
        v = "FUNDAMENTAL" if abs(b_) > 0.25 else ("screened" if keep < 0.4 else "weak")
        print(f"{NICE[x]:10s}{a:>12.3f}{b_:>18.3f}{v:>14s}")

print()
print("=" * 76)
print("M-sigma vs M-M* : which survives when the OTHER ONE ALONE is controlled?")
print("=" * 76)
print(f'{"":8s}{"r(Mbh,M*|sig)":>16s}{"r(Mbh,sig|M*)":>16s}{"fundamental":>16s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    d = {k: v[m] for k, v in d.items()}
    a = pc(d["BH_MASS"], d["STELLAR_MASS"], [d["LOG_SIGMA"]])
    b = pc(d["BH_MASS"], d["LOG_SIGMA"], [d["STELLAR_MASS"]])
    print(f'{t:8s}{a:>16.3f}{b:>16.3f}{("M* (bulge-like)" if abs(a)>abs(b) else "sigma"):>16s}')

print()
print("=" * 76)
print("Is EAGLE halo->BH link DIRECT, or mediated by the gas reservoir?")
print("=" * 76)
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    d = {k: v[m] for k, v in d.items()}
    r1 = pc(d["BH_MASS"], d["DM_MASS"], [d["STELLAR_MASS"]])
    r2 = pc(d["BH_MASS"], d["DM_MASS"], [d["STELLAR_MASS"], d["GAS_MASS"]])
    print(f"{t:8s} r(Mbh,Mhalo|M*) = {r1:6.3f}   adding M_gas as control -> {r2:6.3f}")
