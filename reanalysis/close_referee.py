"""Close three open referee points."""

import pickle, numpy as np

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
P = "/Users/vidigoat/astrophysics/Data/"
rng = np.random.default_rng(1111)


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    return d


print("#" * 78)
print('# REFEREE 4b: "Could discrepancies simply be driven by morphology mismatches?"')
print("# Match on a morphology proxy (sSFR, i.e. star-forming vs quenched-ish) as well")
print("# as stellar mass, and re-run the black hole result.")
print("#" * 78)
print(f'{"":8s}{"r(bh,Mh|M*) raw":>18s}{"sSFR-matched":>16s}{"N matched":>12s}')
# build a common sSFR window too
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    raw = pc(d["BH_MASS"][m], d["DM_MASS"][m], [d["STELLAR_MASS"][m]])
    m2 = m & (d["SSFR"] > -10.6) & (d["SSFR"] < -9.4)  # common main-sequence slice
    v = pc(d["BH_MASS"][m2], d["DM_MASS"][m2], [d["STELLAR_MASS"][m2]]) if m2.sum() > 150 else np.nan
    print(f"{t:8s}{raw:>18.3f}{v:>16.3f}{int(m2.sum()):>12d}")

print()
print("#" * 78)
print("# GAS-DEPLETION FINGERPRINT with a COMPARABLE gas measure.")
print("# TNG GAS_MASS is all bound gas (hot halo included); at LOW halo mass the hot")
print("# component is negligible, so restricting there makes the three comparable.")
print("#" * 78)
print(f'{"":8s}{"all masses":>14s}{"log Mhalo<11.5":>18s}{"N":>8s}   feedback mode')
MODE = {"TNG50": "thermal + KINETIC wind", "EAGLE": "thermal only", "SIMBA": "thermal + KINETIC jets"}
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    ctrl = lambda mm: [d["STELLAR_MASS"][mm], d["DM_MASS"][mm], d["LOG_SIGMA"][mm]]
    a = pc(d["BH_MASS"][m], d["GAS_MASS"][m], ctrl(m))
    m2 = m & (d["DM_MASS"] < 11.5)
    b = pc(d["BH_MASS"][m2], d["GAS_MASS"][m2], ctrl(m2)) if m2.sum() > 150 else np.nan
    print(f"{t:8s}{a:>14.3f}{b:>18.3f}{int(m2.sum()):>8d}   {MODE[t]}")

print()
print("#" * 78)
print("# FMR HIGH-MASS TWIST vs OBSERVATION (Yates, Kauffmann & Guo 2012 found the")
print("# SFR-metallicity anticorrelation REVERSES above log M* ~ 10.5 in 177k SDSS")
print("# galaxies). Which codes reproduce the twist?")
print("#" * 78)


def prep(d):
    ms, z, sfr = d["STELLAR_MASS"], d["GAS_METALLICITY"], d["SFR"]
    ok = (sfr > 0) & (z > 0) & np.isfinite(ms)
    return ms[ok], np.log10(z[ok]), np.log10(sfr[ok]) - ms[ok]


E = {k: np.asarray(v, float) for k, v in pickle.load(open(P + "eagle_final.pkl", "rb")).items()}
S = {k: np.asarray(v, float) for k, v in pickle.load(open(P + "simba_final.pkl", "rb")).items()}
print(f'{"":10s}{"low mass 9-10":>18s}{"high mass >10.5":>20s}{"TWIST?":>12s}')
for lab, d in [("EAGLE", E), ("SIMBA", S)]:
    ms, z, ss = prep(d)
    lo = (ms > 9.0) & (ms < 10.0)
    hi = ms > 10.5
    a = pc(z[lo], ss[lo], [ms[lo]])
    b = pc(z[hi], ss[hi], [ms[hi]])
    tw = "YES" if (a < -0.2 and b > 0.1) else "NO"
    print(f"{lab:10s}{a:>13.3f}({lo.sum():4d}){b:>15.3f}({hi.sum():4d}){tw:>12s}")
print("   observed (Yates+2012, SDSS): negative at low mass, POSITIVE above ~10.5  -> TWIST")
