"""
Build matched samples for the causal-structure comparison across observations
and three independent cosmological simulations (TNG50, EAGLE, SIMBA).

Two comparisons:
  A (observation vs simulations): NSA, TNG50, EAGLE, SIMBA on the properties all
     four measure comparably -- stellar mass, r-band absolute magnitude, U-R
     colour, stellar metallicity, physical size. NSA's angular half-light radius
     is converted to physical size using its distances.
  B (model vs model): TNG50, EAGLE, SIMBA on the full intrinsic set all three
     provide -- halo / stellar / gas / BH / baryonic mass, size, velocity
     dispersion, stellar & gas metallicity, r/u magnitude, colour, SFR.

Every variable is standardised within each sample; heavy-tailed positives (size,
SFR) are log-scaled first; each sample is subsampled to a common N so graph
differences cannot be attributed to sample size. Outputs Data/match{A,B}_*.pkl.
"""
import os
import pickle
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
SEED = 42
N_COMMON = 10000
C = 299792.458
H0 = 70.0


def load(fn):
    return pickle.load(open(os.path.join(DATA, fn), "rb"))


def zscore(x):
    x = np.asarray(x, float)
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)


def subsample(d, n, seed=SEED):
    m = len(next(iter(d.values())))
    if m <= n:
        return d
    idx = np.random.RandomState(seed).choice(m, size=n, replace=False)
    return {k: v[idx] for k, v in d.items()}


def finite_mask(cols):
    m = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        m &= np.isfinite(c)
    return m


def nsa_physical_size(th50_arcsec, zdist):
    zdist = np.asarray(zdist, float)
    d_mpc = zdist * C / H0 if np.nanmedian(zdist) < 1.0 else zdist
    return np.asarray(th50_arcsec, float) * d_mpc * 1000.0 / 206265.0


def standardise(d, logvars=()):
    out = {}
    for k, v in d.items():
        v = np.asarray(v, float)
        if k in logvars:
            v = np.log10(np.maximum(v, 1e-3))
        out[k] = zscore(v)
    return out


def obs_A(nsa):
    return {
        "stellar_mass": nsa["ELPETRO_MASS"], "r_mag": nsa["ELPETRO_ABSMAG_R"],
        "colour": nsa["COLOR_U_R"], "metallicity": nsa["ELPETRO_METS"],
        "size": nsa_physical_size(nsa["ELPETRO_TH50_R"], nsa["ZDIST"]),
    }


def sim_A(s):
    return {
        "stellar_mass": s["STELLAR_MASS"], "r_mag": s["PHOTOMETRIC_R"],
        "colour": s["COLOUR"], "metallicity": s["STAR_METALLICITY"],
        "size": s["HALFMASS_RAD"],
    }


BNAME = {"DM_MASS": "halo_mass", "STELLAR_MASS": "stellar_mass",
         "GAS_MASS": "gas_mass", "BH_MASS": "bh_mass",
         "BARYONIC_MASS": "baryon_mass", "HALFMASS_RAD": "size",
         "VEL_DISP": "veldisp", "STAR_METALLICITY": "Z_star",
         "GAS_METALLICITY": "Z_gas", "PHOTOMETRIC_R": "r_mag",
         "PHOTOMETRIC_U": "u_mag", "COLOUR": "colour", "SFR": "sfr"}


def sim_B(s):
    return {BNAME[k]: s[k] for k in BNAME}


def main():
    np.random.seed(SEED)
    nsa = load("nsa_final_10props.pkl")
    sims = {n: load(f"{n.lower()}_final.pkl") for n in ("TNG50", "EAGLE", "SIMBA")}

    jobs = [("matchA_NSA", obs_A(nsa), ("size",))]
    for n, s in sims.items():
        jobs.append((f"matchA_{n}", sim_A(s), ("size",)))
    for n, s in sims.items():
        jobs.append((f"matchB_{n}", sim_B(s), ("size", "sfr")))

    print("NSA physical size (kpc): median %.2f"
          % np.nanmedian(obs_A(nsa)["size"]))
    for outname, d, logvars in jobs:
        m = finite_mask(list(d.values()))
        d = {k: np.asarray(v, float)[m] for k, v in d.items()}
        d = standardise(d, logvars=logvars)
        d = subsample(d, N_COMMON)
        n = len(next(iter(d.values())))
        with open(os.path.join(DATA, outname + ".pkl"), "wb") as f:
            pickle.dump(d, f)
        print(f"{outname}: N={n}, vars={list(d.keys())}")


if __name__ == "__main__":
    main()
