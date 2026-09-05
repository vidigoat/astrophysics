"""Load a SIMBA Caesar catalogue into the common variable set, with halo-level
quantities (M200c, concentration proxy, spin) and the black-hole accretion rate.

Works for the m100n1024 flagship and the m50n512 feedback variants alike.
"""

import numpy as np
import h5py

G_KPC = 4.30091e-6  # kpc (km/s)^2 / Msun


def load_simba(path, central_only=True, seeded_only=True):
    with h5py.File(path, "r") as f:
        g = f["galaxy_data"]
        gd = g["dicts"]
        h = f["halo_data"]
        hd = h["dicts"]
        n = g["GroupID"].shape[0]
        parent = np.asarray(g["parent_halo_index"])
        central = np.asarray(g["central"]).astype(bool)

        def gal(k):
            return np.asarray(gd[k], float)

        def halo(k):
            arr = np.asarray(hd[k], float)
            out = np.full(n, np.nan)
            ok = (parent >= 0) & (parent < arr.shape[0])
            out[ok] = arr[parent[ok]]
            return out

        m_star = gal("masses.stellar")
        m_gas = gal("masses.gas")
        m_bh = gal("masses.bh")
        m_halo_dm = halo("masses.dm")
        m200 = halo("virial_quantities.m200c")
        m2500 = halo("virial_quantities.m2500c")
        r200 = halo("virial_quantities.r200c")  # kpc (physical? Caesar stores kpccm)
        vcirc = halo("virial_quantities.circular_velocity")
        spin = halo("virial_quantities.spin_param")
        sigma = gal("velocity_dispersions.stellar")
        z_star = gal("metallicities.stellar")
        z_gas = gal("metallicities.sfr_weighted")
        sfr = np.asarray(g["sfr_100"], float)
        bhmdot = np.asarray(g["bhmdot"], float)
        fedd = np.asarray(g["bh_fedd"], float)
        age = gal("ages.mass_weighted")
        z_attr = (
            float(f["simulation_attributes"].attrs.get("redshift", 0.0))
            if "simulation_attributes" in f
            else 0.0
        )
        hpar = (
            float(f["simulation_attributes"].attrs.get("hubble_constant", 0.68))
            if "simulation_attributes" in f
            else 0.68
        )

    v200 = np.sqrt(G_KPC * m200 / np.maximum(r200, 1e-3))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = {
            "STELLAR_MASS": np.log10(m_star),
            "GAS_MASS": np.log10(np.maximum(m_gas, 1.0)),
            "BH_MASS": np.where(m_bh > 0, np.log10(np.maximum(m_bh, 1.0)), -10.0),
            "DM_MASS": np.log10(m_halo_dm),
            "M200": np.log10(m200),
            "CONC_M": np.log10(m2500 / m200),  # concentration proxy: inner/outer mass
            "CONC_V": np.log10(vcirc / v200),  # Vmax/V200 style proxy
            "SPIN": np.log10(spin),
            "LOG_SIGMA": np.log10(np.maximum(sigma, 1e-3)),
            "STAR_METALLICITY": np.log10(np.maximum(z_star, 1e-8)),
            "GAS_METALLICITY": np.log10(np.maximum(z_gas, 1e-8)),
            "SSFR": np.log10(sfr) - np.log10(m_star),
            "BHMDOT": np.log10(np.maximum(bhmdot, 1e-12)),
            "FEDD": np.log10(np.maximum(fedd, 1e-12)),
            "AGE": age,
        }
    mask = np.isfinite(d["STELLAR_MASS"]) & np.isfinite(d["DM_MASS"]) & np.isfinite(d["M200"]) & (m200 > 0)
    mask &= d["STELLAR_MASS"] > 8.5
    if central_only:
        mask &= central
    if seeded_only:
        mask &= m_bh > 0
    return {k: v[mask] for k, v in d.items()}, {"z": z_attr, "h": hpar}
