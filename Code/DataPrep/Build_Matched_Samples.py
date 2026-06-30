"""
Build matched samples for the simulation-vs-observation causal comparison.

Two comparisons:
  A (validation):  observations (NSA) vs TNG50 vs SIMBA, on the set of
                   variables all three measure comparably -- stellar mass,
                   r-band absolute magnitude, U-R colour, stellar metallicity,
                   and physical half-light/half-mass size. NSA's angular size is
                   converted to a physical size using its distances.
  B (model test):  TNG50 vs SIMBA on the full intrinsic set both simulations
                   provide (halo / gas / BH / baryon mass, size, velocity
                   dispersion, metallicities, photometry, SFR).

For a fair comparison every variable is standardised (z-score) within each
sample, heavy-tailed positive quantities (size, SFR) are log-scaled first, and
each sample is subsampled to a common N so detection power is equal (this
removes the "the simulation graph is sparser only because it has fewer
galaxies" objection). Outputs: Data/matchA_*.pkl and Data/matchB_*.pkl.
"""
import os
import pickle
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
SEED = 42
N_COMMON = 10000     # subsample each sample to this many galaxies

C = 299792.458       # km/s
H0 = 70.0            # km/s/Mpc (only used to turn NSA redshift-distance into Mpc)


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
    """NSA angular half-light radius (arcsec) -> physical kpc using ZDIST."""
    zdist = np.asarray(zdist, float)
    # ZDIST is a redshift-like distance; if it looks like a redshift (<~1)
    # convert to Mpc, otherwise assume it is already in Mpc.
    if np.nanmedian(zdist) < 1.0:
        d_mpc = zdist * C / H0
    else:
        d_mpc = zdist
    return np.asarray(th50_arcsec, float) * d_mpc * 1000.0 / 206265.0


def standardise(d, logvars=()):
    out = {}
    for k, v in d.items():
        v = np.asarray(v, float)
        if k in logvars:
            v = np.log10(np.maximum(v, 1e-3))
        out[k] = zscore(v)
    return out


def main():
    np.random.seed(SEED)
    nsa = load("nsa_final_10props.pkl")
    tng = load("tng50_final.pkl")
    sim = load("simba_final.pkl")

    # ---------- Comparison A: common observable variables ----------
    size_nsa = nsa_physical_size(nsa["ELPETRO_TH50_R"], nsa["ZDIST"])
    A_nsa = {
        "stellar_mass": nsa["ELPETRO_MASS"],
        "r_mag":        nsa["ELPETRO_ABSMAG_R"],
        "colour":       nsa["COLOR_U_R"],
        "metallicity":  nsa["ELPETRO_METS"],
        "size":         size_nsa,
    }
    A_tng = {
        "stellar_mass": tng["STELLAR_MASS"], "r_mag": tng["PHOTOMETRIC_R"],
        "colour": tng["COLOUR"], "metallicity": tng["STAR_METALLICITY"],
        "size": tng["HALFMASS_RAD"],
    }
    A_sim = {
        "stellar_mass": sim["STELLAR_MASS"], "r_mag": sim["PHOTOMETRIC_R"],
        "colour": sim["COLOUR"], "metallicity": sim["STAR_METALLICITY"],
        "size": sim["HALFMASS_RAD"],
    }
    print("NSA physical size (kpc): median %.2f  range [%.2f, %.2f]"
          % (np.nanmedian(size_nsa), np.nanmin(size_nsa), np.nanmax(size_nsa)))

    # ---------- Comparison B: full intrinsic set (TNG50 vs SIMBA) ----------
    bvars = ["DM_MASS", "STELLAR_MASS", "GAS_MASS", "BH_MASS", "BARYONIC_MASS",
             "HALFMASS_RAD", "VEL_DISP", "STAR_METALLICITY", "GAS_METALLICITY",
             "PHOTOMETRIC_R", "PHOTOMETRIC_U", "COLOUR", "SFR"]
    bname = {"DM_MASS": "halo_mass", "STELLAR_MASS": "stellar_mass",
             "GAS_MASS": "gas_mass", "BH_MASS": "bh_mass",
             "BARYONIC_MASS": "baryon_mass", "HALFMASS_RAD": "size",
             "VEL_DISP": "veldisp", "STAR_METALLICITY": "Z_star",
             "GAS_METALLICITY": "Z_gas", "PHOTOMETRIC_R": "r_mag",
             "PHOTOMETRIC_U": "u_mag", "COLOUR": "colour", "SFR": "sfr"}
    B_tng = {bname[k]: tng[k] for k in bvars}
    B_sim = {bname[k]: sim[k] for k in bvars}

    jobs = [
        ("matchA_NSA",   A_nsa, ("size",)),
        ("matchA_TNG50", A_tng, ("size",)),
        ("matchA_SIMBA", A_sim, ("size",)),
        ("matchB_TNG50", B_tng, ("size", "sfr")),
        ("matchB_SIMBA", B_sim, ("size", "sfr")),
    ]
    for outname, d, logvars in jobs:
        cols = list(d.values())
        m = finite_mask(cols)
        d = {k: np.asarray(v, float)[m] for k, v in d.items()}
        d = standardise(d, logvars=logvars)
        d = subsample(d, N_COMMON)
        n = len(next(iter(d.values())))
        with open(os.path.join(DATA, outname + ".pkl"), "wb") as f:
            pickle.dump(d, f)
        print(f"{outname}: N={n}, vars={list(d.keys())}")


if __name__ == "__main__":
    main()
