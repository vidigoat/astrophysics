# Dataset Properties and Quality Cuts

## 1. NSA (NASA-Sloan Atlas) — `nsa_v1_0_1.fits`

| Property | Description | Cut |
|---|---|---|
| `COLOR_U_R` | U−R colour (from ELPETRO_ABSMAG bands 2 & 4) | −0.5 ≤ x ≤ 4.0 |
| `ELPETRO_B300` | Star formation rate (300 Myr avg, elliptical Petrosian) | 1×10⁻⁸ < x < 10.0 |
| `SERSIC_N` | Sérsic index | 0 < x < 6.0 |
| `ELPETRO_METS` | Stellar metallicity | −2.5 < x < 0.5 |
| `ELPETRO_MTOL` | Mass-to-light ratio (r-band, index 4) | 0.1 ≤ x ≤ 10.0 |
| `ELPETRO_BA` | Axis ratio (b/a) | 0 < x < 1.0 |
| `ELPETRO_TH50_R` | Half-light radius in r-band (arcsec) | 0 < x < 25.0 |
| `ZDIST` | Distance redshift | x < 0.15 |
| `ELPETRO_MASS` | Stellar mass (log₁₀ M☉) | 6.0 < x < 12.0 |
| `ELPETRO_ABSMAG_R` | Absolute magnitude, r-band (index 4) | −25.0 < x < −10.0 |

---

## 2. ALFALFA × NSA — `5asfullmatch.fits`

All NSA cuts above apply, plus:

| Property | Description | Cut |
|---|---|---|
| `COLOR_U_R` | U−R colour (from ELPETRO_ABSMAG bands 2 & 4) | −0.5 ≤ x ≤ 4.0 |
| `ELPETRO_B300` | Star formation rate (300 Myr avg, elliptical Petrosian) | 1×10⁻⁸ < x < 10.0 |
| `SERSIC_N` | Sérsic index | 0 < x < 6.0 |
| `ELPETRO_METS` | Stellar metallicity | −2.5 < x < 0.5 |
| `ELPETRO_MTOL` | Mass-to-light ratio (r-band, index 4) | 0.1 ≤ x ≤ 10.0 |
| `ELPETRO_BA` | Axis ratio (b/a) | 0 < x < 1.0 |
| `ELPETRO_TH50_R` | Half-light radius in r-band (arcsec) | 0 < x < 25.0 |
| `ZDIST` | Distance redshift | x < 0.15 |
| `ELPETRO_MASS` | Stellar mass (log₁₀ M☉) | 6.0 < x < 12.0 |
| `ELPETRO_ABSMAG_R` | Absolute magnitude, r-band (index 4) | −25.0 < x < −10.0 |
| `logMH` | HI mass (log₁₀ M☉) | 6.0 ≤ x ≤ 10.5 |
| `W50` | HI line width at 50% peak (km/s) | 20.0 < x < 500.0 |
| `BARYONIC_MASS` | Baryonic mass log₁₀(M★ + 1.4·M_HI) | 6.0 < x < 12.0 |

---

## 3. TNG50 (IllustrisTNG) — `subhalos_mstar_gt1e8.hdf5`

Masses stored in units of 10¹⁰ M☉ in the HDF5 file; converted to log₁₀(M☉) before cuts.

| Property | Description | Cut |
|---|---|---|
| `DM_MASS` | Dark matter mass (log₁₀ M☉) | x > 5.0 |
| `STELLAR_MASS` | Stellar mass (log₁₀ M☉) | 6.0 < x < 14.0 |
| `GAS_MASS` | Gas mass (log₁₀ M☉) | 6.5 < x < 11.5 |
| `BH_MASS` | Black hole mass (log₁₀ M☉) | x < 12.0 |
| `BARYONIC_MASS` | Baryonic mass log₁₀(M★ + M_gas) | 7.0 < x < 13.0 |
| `HALFMASS_RAD` | Half-mass radius (kpc/h) | −1.5 < log₁₀(x) < 2.5 |
| `VEL_DISP` | Velocity dispersion (km/s) | 2.0 < x < 300.0 |
| `VMAX` | Maximum circular velocity (km/s) | 5.0 < x < 500.0 |
| `GAS_METALLICITY` | Gas-phase metallicity (Z) | 0.0 ≤ x < 0.15 |
| `STAR_METALLICITY` | Stellar metallicity (Z) | finite values only |
| `PHOTOMETRIC_R` | r-band absolute magnitude | −26.0 < x < −9.0 |
| `PHOTOMETRIC_U` | u-band absolute magnitude | finite values only |
| `COLOUR` | U−R colour (PHOTOMETRIC_U − PHOTOMETRIC_R) | −2.0 < x < 4.0 |
| `SFR` | Star formation rate (log₁₀ M☉/yr) | log₁₀(SFR) < 2.5 |
