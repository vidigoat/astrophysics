"""Is the EAGLE high-mass FMR inversion an artefact of the SFR>0 selection?"""

import pickle, numpy as np

P = "/Users/vidigoat/astrophysics/Data/"
rng = np.random.default_rng(404)


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


E = {k: np.asarray(v, float) for k, v in pickle.load(open(P + "eagle_final.pkl", "rb")).items()}
S = {k: np.asarray(v, float) for k, v in pickle.load(open(P + "simba_final.pkl", "rb")).items()}
print("Quenched fraction (SFR=0) by stellar mass -- how severe is the SF-only cut?")
print(f'{"bin":>12s}{"EAGLE":>22s}{"SIMBA":>22s}')
for lo, hi in [(9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]:
    row = f'{f"{lo}-{hi}":>12s}'
    for d in (E, S):
        m = (d["STELLAR_MASS"] > lo) & (d["STELLAR_MASS"] < hi)
        n = int(m.sum())
        q = int(np.sum(d["SFR"][m] <= 0))
        row += f"{q:>8d}/{n:<6d} ({100*q/max(n,1):4.1f}%)"
    print(row)
print()
print("Does the inversion survive if we use SFR (not sSFR), or add a floor instead of a cut?")
for lab, tag in [("EAGLE", None), ("SIMBA", None)]:
    d = E if lab == "EAGLE" else S
    print(f"\n{lab}:")
    for lo, hi in [(10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]:
        m = (d["STELLAR_MASS"] > lo) & (d["STELLAR_MASS"] < hi) & (d["GAS_METALLICITY"] > 0)
        ms = d["STELLAR_MASS"][m]
        z = np.log10(d["GAS_METALLICITY"][m])
        sfr = d["SFR"][m]
        sf = sfr > 0
        if sf.sum() < 80:
            print(f"   {lo}-{hi}: too few")
            continue
        r_ssfr = pc(z[sf], np.log10(sfr[sf]) - ms[sf], [ms[sf]])
        r_sfr = pc(z[sf], np.log10(sfr[sf]), [ms[sf]])
        # floor the quenched at 1e-4 instead of dropping them
        sfrf = np.maximum(sfr, 1e-4)
        r_floor = pc(z, np.log10(sfrf) - ms, [ms])
        print(
            f"   {lo}-{hi}  N_SF={sf.sum():5d}   r(Z,sSFR|M*)={r_ssfr:+.3f}   r(Z,SFR|M*)={r_sfr:+.3f}   with quenched floored={r_floor:+.3f}"
        )
