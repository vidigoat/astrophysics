"""The FMR properly. Use the FULL star-forming sample (no BH requirement -- the
BH cut removed most massive galaxies and destroyed the statistics).
TNG50's GAS_METALLICITY is ALL bound gas incl. hot halo; EAGLE (SF_Metallicity)
and SIMBA (sfr_weighted) both track star-forming gas, i.e. what HII-region
observations measure. So EAGLE vs SIMBA is the defensible comparison."""

import pickle, numpy as np

P = "/Users/vidigoat/astrophysics/Data/"
rng = np.random.default_rng(303)


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


def sf_sample(name, fix_tng=False):
    d = {k: np.asarray(v, float) for k, v in pickle.load(open(P + name + ".pkl", "rb")).items()}
    if fix_tng:
        g, dm, st = d["DM_MASS"].copy(), d["STELLAR_MASS"].copy(), d["GAS_MASS"].copy()
        d["GAS_MASS"], d["DM_MASS"], d["STELLAR_MASS"] = g, dm, st  # relabel: 0=gas,1=DM,4=stars
        d["GAS_MASS"], d["DM_MASS"], d["STELLAR_MASS"] = g, st, d["GAS_MASS"]
    return d


tng = pickle.load(open(P + "tng50_final.pkl", "rb"))
tng = {k: np.asarray(v, float) for k, v in tng.items()}
# corrected TNG labels
TNG = {"STELLAR_MASS": tng["GAS_MASS"], "GAS_METALLICITY": tng["GAS_METALLICITY"], "SFR": tng["SFR"]}
EAG = pickle.load(open(P + "eagle_final.pkl", "rb"))
EAG = {k: np.asarray(v, float) for k, v in EAG.items()}
SIM = pickle.load(open(P + "simba_final.pkl", "rb"))
SIM = {k: np.asarray(v, float) for k, v in SIM.items()}


def prep(d):
    ms, z, sfr = d["STELLAR_MASS"], d["GAS_METALLICITY"], d["SFR"]
    ok = (sfr > 0) & (z > 0) & np.isfinite(ms)
    ms, z, sfr = ms[ok], z[ok], sfr[ok]
    return ms, np.log10(z), np.log10(sfr) - ms


S = {"TNG50": prep(TNG), "EAGLE": prep(EAG), "SIMBA": prep(SIM)}
print("Star-forming samples (no BH cut):")
for t, (ms, z, ss) in S.items():
    print(f"   {t:7s} N={len(ms):6d}   median logM* = {np.median(ms):.2f}")
print()
print("THE FMR:  r(log Z_gas, log sSFR | log M*)   -- observed sign is NEGATIVE")
print("Observed: Mannucci+2010 / Curti+2020 find the SFR anti-correlation strong at low")
print("mass and FLATTENING toward zero above log M* ~ 10.5-10.9. It does not invert.")
print()
BINS = [(8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]
print(f'{"log M* bin":>12s}' + "".join(f"{t:>24s}" for t in ["TNG50*", "EAGLE", "SIMBA"]))
print(f'{"":>12s}' + "".join(f'{"r":>10s}{"+-":>7s}{"N":>7s}' for _ in range(3)))
for lo, hi in BINS:
    row = f'{f"{lo}-{hi}":>12s}'
    for t in ["TNG50", "EAGLE", "SIMBA"]:
        ms, z, ss = S[t]
        m = (ms > lo) & (ms < hi)
        n = int(m.sum())
        if n < 120:
            row += f'{"--":>10s}{"":>7s}{n:>7d}'
            continue
        v = pc(z[m], ss[m], [ms[m]])
        bs = [pc(z[m][i], ss[m][i], [ms[m][i]]) for i in (rng.choice(n, n, replace=True) for _ in range(200))]
        row += f"{v:>10.3f}{np.std(bs):>7.3f}{n:>7d}"
    print(row)
print()
print("* TNG50 metallicity is ALL bound gas (incl. hot halo), not star-forming gas,")
print("  so it is NOT comparable with EAGLE/SIMBA or with HII-region observations.")
