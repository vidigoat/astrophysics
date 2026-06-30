"""
Data preparation for the SIMBA cosmological simulation (Dave et al. 2019).

Reads the public CAESAR galaxy catalogue (m100n1024, s50 full-physics run,
snapshot 151 = z=0) and extracts the same first-order galaxy properties used for
TNG50 and EAGLE, with dust-free synthetic SDSS photometry (for consistency with
the TNG50 and EAGLE photometry). Dark-matter halo mass is taken from each
galaxy's parent halo. Galaxies with no resolved central black hole are floored.

Output: Data/simba_final.pkl  (dict of variable -> array).
"""
import os
import pickle
import numpy as np
import h5py

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
CAT = os.path.join(DATA, "simba_m100n1024_151.hdf5")
OUT = os.path.join(DATA, "simba_final.pkl")
BH_FLOOR_LOG = -10.0


def _log10(x, floor=1e-10):
    return np.log10(np.maximum(np.asarray(x, float), floor))


def main():
    if not os.path.exists(CAT):
        raise FileNotFoundError(f"SIMBA catalogue not found: {CAT}")
    with h5py.File(CAT, "r") as f:
        g = f["galaxy_data"]
        gd = g["dicts"]
        n = g["GroupID"].shape[0]
        m_star = np.asarray(gd["masses.stellar"])
        m_gas = np.asarray(gd["masses.gas"])
        m_bh = np.asarray(gd["masses.bh"])
        m_bary = np.asarray(gd["masses.baryon"])
        z_star = np.asarray(gd["metallicities.stellar"])
        z_gas = np.asarray(gd["metallicities.sfr_weighted"])
        r_half = np.asarray(gd["radii.stellar_half_mass"])
        sigma = np.asarray(gd["velocity_dispersions.stellar"])
        absmag_r = np.asarray(gd["absmag_nodust.sdss_r"])  # dust-free, like TNG50/EAGLE
        absmag_u = np.asarray(gd["absmag_nodust.sdss_u"])
        sfr = np.asarray(g["sfr_100"])
        central = np.asarray(g["central"]).astype(bool)
        parent = np.asarray(g["parent_halo_index"])
        halo_dm = np.asarray(f["halo_data/dicts"]["masses.dm"])
        m_halo = np.full(n, np.nan)
        ok = (parent >= 0) & (parent < halo_dm.shape[0])
        m_halo[ok] = halo_dm[parent[ok]]

    colour = absmag_u - absmag_r
    d = {
        "DM_MASS": _log10(m_halo),
        "STELLAR_MASS": _log10(m_star),
        "GAS_MASS": _log10(m_gas),
        "BH_MASS": np.where(m_bh > 0, _log10(m_bh), BH_FLOOR_LOG),
        "BARYONIC_MASS": _log10(m_bary),
        "HALFMASS_RAD": np.asarray(r_half, float),
        "VEL_DISP": np.asarray(sigma, float),
        "STAR_METALLICITY": np.asarray(z_star, float),
        "GAS_METALLICITY": np.asarray(z_gas, float),
        "PHOTOMETRIC_R": np.asarray(absmag_r, float),
        "PHOTOMETRIC_U": np.asarray(absmag_u, float),
        "COLOUR": np.asarray(colour, float),
        "SFR": np.asarray(sfr, float),
    }
    mask = central & np.isfinite(d["DM_MASS"])
    for k, v in d.items():
        mask &= np.isfinite(v)
    mask &= (d["STELLAR_MASS"] > 8.5) & (d["STELLAR_MASS"] < 13.0)
    mask &= (d["GAS_MASS"] > 6.5) & (d["GAS_MASS"] < 12.0)
    mask &= (d["HALFMASS_RAD"] > 0.0) & (d["HALFMASS_RAD"] < 100.0)
    mask &= (d["VEL_DISP"] > 2.0) & (d["VEL_DISP"] < 500.0)
    mask &= (d["PHOTOMETRIC_R"] > -26.0) & (d["PHOTOMETRIC_R"] < -8.0)
    mask &= (d["COLOUR"] > -1.0) & (d["COLOUR"] < 4.0)
    for k in list(d.keys()):
        d[k] = d[k][mask]
    n_final = len(d["STELLAR_MASS"])
    print(f"SIMBA central galaxies: {n} -> {n_final} after selection")
    for k, v in d.items():
        print(f"  {k:16s} [{np.min(v):8.3f}, {np.max(v):8.3f}]  mean {np.mean(v):8.3f}")
    with open(OUT, "wb") as fp:
        pickle.dump(d, fp)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
