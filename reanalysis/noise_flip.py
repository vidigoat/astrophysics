"""Fragility test. EAGLE's M* is a 30 kpc aperture mass; TNG50/SIMBA use total mass.
If measurement noise in M* alone can flip a code from 'stellar-coupled' to
'halo-coupled', the result is an artefact. Inject noise and find the flip point."""

import pickle, numpy as np

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
LO, HI = 9.5, 11.0
rng = np.random.default_rng(19)


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


print("Adding Gaussian noise to log M* and re-running the screening test.")
print('"flip" = r(Mbh,Mhalo|M*) rises above r(Mbh,M*|Mhalo), i.e. looks halo-coupled.\n')
print(f'{"noise":>7s}' + "".join(f"{t:>34s}" for t in ["TNG50", "EAGLE", "SIMBA"]))
print(f'{"(dex)":>7s}' + "".join(f'{"r(bh,M*|Mh)":>17s}{"r(bh,Mh|M*)":>17s}' for _ in range(3)))
for nz in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    row = f"{nz:>7.2f}"
    for t in ["TNG50", "EAGLE", "SIMBA"]:
        d = pickle.load(open(D + f"{t}_clean.pkl", "rb"))
        m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
        bh, ms, mh = d["BH_MASS"][m], d["STELLAR_MASS"][m].copy(), d["DM_MASS"][m]
        if nz > 0:
            ms = ms + rng.normal(0, nz, len(ms))
        a, b = pc(bh, ms, [mh]), pc(bh, mh, [ms])
        star = "*" if b > a else " "
        row += f"{a:>17.3f}{b:>16.3f}{star}"
    print(row)

print("\n" + "=" * 70)
print("Mirror: add noise to log M_halo instead. Does EAGLE lose its halo link?")
print("=" * 70)
print(f'{"noise":>7s}' + "".join(f"{t:>34s}" for t in ["TNG50", "EAGLE", "SIMBA"]))
for nz in [0.0, 0.05, 0.10, 0.20, 0.30]:
    row = f"{nz:>7.2f}"
    for t in ["TNG50", "EAGLE", "SIMBA"]:
        d = pickle.load(open(D + f"{t}_clean.pkl", "rb"))
        m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
        bh, ms, mh = d["BH_MASS"][m], d["STELLAR_MASS"][m], d["DM_MASS"][m].copy()
        if nz > 0:
            mh = mh + rng.normal(0, nz, len(mh))
        a, b = pc(bh, ms, [mh]), pc(bh, mh, [ms])
        row += f"{a:>17.3f}{b:>17.3f}"
    print(row)

print("\nIntrinsic scatter of each variable for scale:")
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = pickle.load(open(D + f"{t}_clean.pkl", "rb"))
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    print(
        f'  {t:7s} sd(logM*)={np.std(d["STELLAR_MASS"][m]):.3f}  sd(logMhalo)={np.std(d["DM_MASS"][m]):.3f}'
    )
