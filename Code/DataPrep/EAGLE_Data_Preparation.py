"""
Data preparation for the EAGLE simulation (Schaye et al. 2015; McAlpine et al.
2016), reference run RefL0100N1504 at z=0 (SnapNum 28).

Reads the CSV pulled from the public EAGLE database (central galaxies, 30 kpc
aperture quantities + dust-free SDSS photometry) and writes the same first-order
property set used for TNG50 and SIMBA, so the existing FCIT/consensus pipeline
can consume it. Output: Data/eagle_final.pkl.
"""
import os
import pickle
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
CSV = os.path.join(DATA, "eagle_raw.csv")
OUT = os.path.join(DATA, "eagle_final.pkl")
BH_FLOOR_LOG = -10.0


def _log10(x, floor=1e-10):
    return np.log10(np.maximum(np.asarray(x, float), floor))


def main():
    df = pd.read_csv(CSV, comment="#")
    n0 = len(df)
    m_star = df["Mass_Star"].to_numpy(float)
    m_gas = df["Mass_Gas"].to_numpy(float)
    m_bh = df["MassType_BH"].to_numpy(float)
    m_dm = df["MassType_DM"].to_numpy(float)
    u = df["u_nodust"].to_numpy(float)
    r = df["r_nodust"].to_numpy(float)

    d = {
        "DM_MASS": _log10(m_dm),
        "STELLAR_MASS": _log10(m_star),
        "GAS_MASS": _log10(m_gas),
        "BH_MASS": np.where(m_bh > 0, _log10(m_bh), BH_FLOOR_LOG),
        "BARYONIC_MASS": _log10(m_star + m_gas),
        "HALFMASS_RAD": df["R_halfmass30"].to_numpy(float),
        "VEL_DISP": df["VelDisp"].to_numpy(float),
        "VMAX": df["Vmax"].to_numpy(float),
        "STAR_METALLICITY": df["Stars_Metallicity"].to_numpy(float),
        "GAS_METALLICITY": df["SF_Metallicity"].to_numpy(float),
        "PHOTOMETRIC_R": r,
        "PHOTOMETRIC_U": u,
        "COLOUR": u - r,
        "SFR": df["SFR"].to_numpy(float),
    }

    mask = np.ones(n0, dtype=bool)
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
    n = len(d["STELLAR_MASS"])
    print(f"EAGLE central galaxies: {n0} -> {n} after selection")
    for k, v in d.items():
        print(f"  {k:16s} [{np.min(v):8.3f}, {np.max(v):8.3f}]  mean {np.mean(v):8.3f}")
    with open(OUT, "wb") as f:
        pickle.dump(d, f)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
