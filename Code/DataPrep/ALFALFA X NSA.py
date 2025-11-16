import numpy as np
from astropy.io import fits
import pickle
import os

# Get FITS file path from environment variable or use default
FITS_PATH = os.environ.get('ALFALFA_FITS_PATH', '5asfullmatch.fits')

print("Loading FITS file...")
with fits.open(FITS_PATH, memmap=True) as hdul:
    data = hdul[1].data
    n_rows = len(data)
    print(f"Total galaxies: {n_rows}")

    print("\nExtracting base properties...")
    sersic_n = np.asarray(data['SERSIC_N'])
    zdist = np.asarray(data['ZDIST'])
    elpetro_b300 = np.asarray(data['ELPETRO_B300'])
    elpetro_mets = np.asarray(data['ELPETRO_METS'])
    elpetro_mtol_array = np.asarray(data['ELPETRO_MTOL'])
    elpetro_mtol = elpetro_mtol_array[:, 4]
    elpetro_ba = np.asarray(data['ELPETRO_BA'])
    elpetro_th50_r = np.asarray(data['ELPETRO_TH50_R'])
    elpetro_mass = np.asarray(data['ELPETRO_MASS'])
    logMH = np.asarray(data['logMH'])
    w50 = np.asarray(data['W50'])  # HI velocity width at 50% peak flux (km/s)
    elpetro_absmag = np.asarray(data['ELPETRO_ABSMAG'])

print(f"ELPETRO_ABSMAG shape: {elpetro_absmag.shape}")

color_u_r = elpetro_absmag[:, 2] - elpetro_absmag[:, 4]
elpetro_absmag_r = elpetro_absmag[:, 4]  # r-band absolute magnitude
M_HI = 10 ** logMH
baryonic_mass = elpetro_mass + 1.4 * M_HI

data_dict = {
    'BARYONIC_MASS': baryonic_mass,
    'COLOR_U_R': color_u_r,
    'ELPETRO_B300': elpetro_b300,
    'SERSIC_N': sersic_n,
    'ELPETRO_METS': elpetro_mets,
    'ELPETRO_MTOL': elpetro_mtol,
    'ELPETRO_BA': elpetro_ba,
    'ELPETRO_TH50_R': elpetro_th50_r,
    'ZDIST': zdist,
    'logMH': logMH,
    'ELPETRO_MASS': elpetro_mass,
    'ELPETRO_ABSMAG_R': elpetro_absmag_r,
    'W50': w50
}

print("\n=== DATA QUALITY CHECK ===")
valid_mask = np.ones(n_rows, dtype=bool)
for var_name, var_data in data_dict.items():
    finite_mask = np.isfinite(var_data)
    n_bad = np.count_nonzero(~finite_mask)
    if n_bad > 0:
        print(f"{var_name}: {n_bad} non-finite entries")
    else:
        print(f"{var_name}: Clean")
    valid_mask &= finite_mask

n_valid = np.count_nonzero(valid_mask)
print(f"\nGalaxies with complete data: {n_valid} / {n_rows}")
print(f"Removed: {n_rows - n_valid} galaxies (non-finite values)")

for key in data_dict:
    data_dict[key] = data_dict[key][valid_mask]

# Apply physical parameter cuts
print("\n=== APPLYING PHYSICAL PARAMETER CUTS ===")
n_before_cuts = len(data_dict['ELPETRO_MASS'])
cut_mask = np.ones(n_before_cuts, dtype=bool)

# BARYONIC_MASS: 10^6 < M_baryon < 10^12
cut = (data_dict['BARYONIC_MASS'] > 1e6) & (data_dict['BARYONIC_MASS'] < 1e12)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"BARYONIC_MASS: Removed {n_removed} galaxies (outside 10^6 - 10^12 M_sun)")
cut_mask &= cut

# COLOR_U_R: -0.5 <= (u−r) <= 4
cut = (data_dict['COLOR_U_R'] >= -0.5) & (data_dict['COLOR_U_R'] <= 4.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"COLOR_U_R: Removed {n_removed} galaxies (outside -0.5 - 4 mag)")
cut_mask &= cut

# ELPETRO_B300: 10^-8 < B300 < 10
cut = (data_dict['ELPETRO_B300'] > 1e-8) & (data_dict['ELPETRO_B300'] < 10.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_B300: Removed {n_removed} galaxies (outside 10^-8 - 10)")
cut_mask &= cut

# SERSIC_N: 0 < n < 6
cut = (data_dict['SERSIC_N'] > 0) & (data_dict['SERSIC_N'] < 6.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"SERSIC_N: Removed {n_removed} galaxies (outside 0 - 6)")
cut_mask &= cut

# ELPETRO_METS: -2.5 < Z < 0.5
cut = (data_dict['ELPETRO_METS'] > -2.5) & (data_dict['ELPETRO_METS'] < 0.5)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_METS: Removed {n_removed} galaxies (outside -2.5 - 0.5 dex)")
cut_mask &= cut

# ELPETRO_MTOL: 0.1 <= M/L <= 10
cut = (data_dict['ELPETRO_MTOL'] >= 0.1) & (data_dict['ELPETRO_MTOL'] <= 10.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_MTOL: Removed {n_removed} galaxies (outside 0.1 - 10 M_sun/L_sun)")
cut_mask &= cut

# ELPETRO_BA: 0 < b/a < 1
cut = (data_dict['ELPETRO_BA'] > 0) & (data_dict['ELPETRO_BA'] < 1.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_BA: Removed {n_removed} galaxies (outside 0 - 1)")
cut_mask &= cut

# ELPETRO_TH50_R: 0 < r_50 < 25
cut = (data_dict['ELPETRO_TH50_R'] > 0) & (data_dict['ELPETRO_TH50_R'] < 25.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_TH50_R: Removed {n_removed} galaxies (outside 0 - 25 arcsec)")
cut_mask &= cut

# ZDIST: z < 0.15
cut = data_dict['ZDIST'] < 0.15
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ZDIST: Removed {n_removed} galaxies (z >= 0.15)")
cut_mask &= cut

# logMH: 6 <= log M_HI <= 10.5
cut = (data_dict['logMH'] >= 6.0) & (data_dict['logMH'] <= 10.5)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"logMH: Removed {n_removed} galaxies (outside 6 - 10.5)")
cut_mask &= cut

# ELPETRO_MASS: 10^6 < M* < 10^12
cut = (data_dict['ELPETRO_MASS'] > 1e6) & (data_dict['ELPETRO_MASS'] < 1e12)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_MASS: Removed {n_removed} galaxies (outside 10^6 - 10^12 M_sun)")
cut_mask &= cut

# ELPETRO_ABSMAG_R: -25 < M_r < -10 (reasonable absolute magnitude range)
cut = (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"ELPETRO_ABSMAG_R: Removed {n_removed} galaxies (outside -25 - -10 mag)")
cut_mask &= cut

# W50: 20 < W50 < 500 km/s (reasonable HI velocity width range for galaxies)
cut = (data_dict['W50'] > 20.0) & (data_dict['W50'] < 500.0)
n_removed = np.count_nonzero(~cut)
if n_removed > 0:
    print(f"W50: Removed {n_removed} galaxies (outside 20 - 500 km/s)")
cut_mask &= cut

# Apply cuts
n_after_cuts = np.count_nonzero(cut_mask)
n_removed_by_cuts = n_before_cuts - n_after_cuts
print(f"\nAfter parameter cuts: {n_after_cuts:,} / {n_before_cuts:,} galaxies")
print(f"Removed by parameter cuts: {n_removed_by_cuts:,} galaxies")

for key in data_dict:
    data_dict[key] = data_dict[key][cut_mask]

print("\n=== SUMMARY STATISTICS ===")
for var_name, var_data in data_dict.items():
    print(f"\n{var_name}:")
    print(f"  Min:     {np.min(var_data):.6e}")
    print(f"  Max:     {np.max(var_data):.6e}")
    print(f"  Mean:    {np.mean(var_data):.6e}")
    print(f"  Std:     {np.std(var_data):.6e}")
    print(f"  Median:  {np.median(var_data):.6e}")

# Use Data folder at project root (two levels up from this script)
output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', 'alfalfa_nsa_final_13props.pkl')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as f_out:
    pickle.dump(data_dict, f_out)

print(f"\n{'='*70}")
print(f"FINAL DATASET SAVED: {output_file}")
print(f"Properties: {len(data_dict)}")
print(f"Galaxies: {len(next(iter(data_dict.values())))}")
print("\nFinal Property List:")
for i, var in enumerate(data_dict.keys(), 1):
    print(f"  {i:2d}. {var}")
print(f"{'='*70}")