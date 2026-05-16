"""
Data preparation script for ALFALFA × NSA dataset.
Extracts galaxy properties, applies quality cuts, and saves processed data.
"""

import numpy as np
from astropy.io import fits
import pickle
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_FITS_PATH = r'C:\Users\sanji\Downloads\5asfullmatch.fits'
DEFAULT_FITS_PATH_ROOT = os.path.join(REPO_ROOT, '5asfullmatch.fits')
DEFAULT_FITS_PATH_DATA = os.path.join(REPO_ROOT, 'Data', '5asfullmatch.fits')

if 'ALFALFA_FITS_PATH' in os.environ:
    FITS_PATH = os.environ.get('ALFALFA_FITS_PATH')
elif os.path.exists(DEFAULT_FITS_PATH):
    FITS_PATH = DEFAULT_FITS_PATH
elif os.path.exists(DEFAULT_FITS_PATH_ROOT):
    FITS_PATH = DEFAULT_FITS_PATH_ROOT
else:
    FITS_PATH = DEFAULT_FITS_PATH_DATA

if not os.path.exists(FITS_PATH):
    raise FileNotFoundError(
        f"FITS file not found: {FITS_PATH}\n"
        f"Set ALFALFA_FITS_PATH environment variable or place file at:\n"
        f"  - {DEFAULT_FITS_PATH}\n"
        f"  - {DEFAULT_FITS_PATH_ROOT}\n"
        f"  - {DEFAULT_FITS_PATH_DATA}"
    )

print(f"Loading ALFALFA×NSA data from: {FITS_PATH}")

with fits.open(FITS_PATH, memmap=True) as hdul:
    data = hdul[1].data
    n_rows = len(data)
    print(f"Total galaxies in file: {n_rows:,}")

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
    w50 = np.asarray(data['W50'])
    elpetro_absmag = np.asarray(data['ELPETRO_ABSMAG'])

color_u_r = elpetro_absmag[:, 2] - elpetro_absmag[:, 4]
elpetro_absmag_r = elpetro_absmag[:, 4]
M_HI = 10 ** logMH
baryonic_mass = elpetro_mass + 1.4 * M_HI

log_baryonic_mass = np.log10(np.maximum(baryonic_mass, 1e-10))
log_stellar_mass = np.log10(np.maximum(elpetro_mass, 1e-10))

data_dict = {
    'BARYONIC_MASS': log_baryonic_mass,
    'COLOR_U_R': color_u_r,
    'ELPETRO_B300': elpetro_b300,
    'SERSIC_N': sersic_n,
    'ELPETRO_METS': elpetro_mets,
    'ELPETRO_MTOL': elpetro_mtol,
    'ELPETRO_BA': elpetro_ba,
    'ELPETRO_TH50_R': elpetro_th50_r,
    'ZDIST': zdist,
    'logMH': logMH,
    'ELPETRO_MASS': log_stellar_mass,
    'ELPETRO_ABSMAG_R': elpetro_absmag_r,
    'W50': w50
}

valid_mask = np.ones(n_rows, dtype=bool)
for var_name, var_data in data_dict.items():
    finite_mask = np.isfinite(var_data)
    valid_mask &= finite_mask

n_valid = np.count_nonzero(valid_mask)
if n_valid == 0:
    raise RuntimeError("No valid rows remain after filtering.")

print(f"Galaxies with finite values: {n_valid:,} ({100*n_valid/n_rows:.1f}%)")

for key in data_dict:
    data_dict[key] = data_dict[key][valid_mask]

n_before_cuts = len(data_dict['ELPETRO_MASS'])
cut_mask = np.ones(n_before_cuts, dtype=bool)

# Track individual cuts and their impact
cut_info = []

cut = (data_dict['BARYONIC_MASS'] > 6.0) & (data_dict['BARYONIC_MASS'] < 12.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('BARYONIC_MASS (6.0 < x < 12.0)', n_before - n_after, n_before))

cut = (data_dict['COLOR_U_R'] >= -0.5) & (data_dict['COLOR_U_R'] <= 4.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('COLOR_U_R (-0.5 <= x <= 4.0)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_B300'] > 1e-8) & (data_dict['ELPETRO_B300'] < 10.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_B300 (1e-8 < x < 10.0)', n_before - n_after, n_before))

cut = (data_dict['SERSIC_N'] > 0) & (data_dict['SERSIC_N'] < 6.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('SERSIC_N (0 < x < 6.0)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_METS'] > -2.5) & (data_dict['ELPETRO_METS'] < 0.5)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_METS (-2.5 < x < 0.5)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_MTOL'] >= 0.1) & (data_dict['ELPETRO_MTOL'] <= 10.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_MTOL (0.1 <= x <= 10.0)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_BA'] > 0) & (data_dict['ELPETRO_BA'] < 1.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_BA (0 < x < 1.0)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_TH50_R'] > 0) & (data_dict['ELPETRO_TH50_R'] < 25.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_TH50_R (0 < x < 25.0)', n_before - n_after, n_before))

cut = data_dict['ZDIST'] < 0.15
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ZDIST (x < 0.15)', n_before - n_after, n_before))

cut = (data_dict['logMH'] >= 6.0) & (data_dict['logMH'] <= 10.5)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('logMH (6.0 <= x <= 10.5)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_MASS'] > 6.0) & (data_dict['ELPETRO_MASS'] < 12.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_MASS (6.0 < x < 12.0)', n_before - n_after, n_before))

cut = (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('ELPETRO_ABSMAG_R (-25.0 < x < -10.0)', n_before - n_after, n_before))

cut = (data_dict['W50'] > 20.0) & (data_dict['W50'] < 500.0)
n_before = np.count_nonzero(cut_mask)
cut_mask &= cut
n_after = np.count_nonzero(cut_mask)
cut_info.append(('W50 (20.0 < x < 500.0)', n_before - n_after, n_before))

for key in data_dict:
    data_dict[key] = data_dict[key][cut_mask]

n_after_cuts = len(data_dict['ELPETRO_MASS'])
n_removed = n_before_cuts - n_after_cuts

print(f"\nQuality cuts applied:")
print(f"  Galaxies before cuts: {n_before_cuts:,}")
print(f"  Galaxies after cuts: {n_after_cuts:,}")
print(f"  Removed: {n_removed:,} ({100*n_removed/n_before_cuts:.1f}%)")

# Find and report the biggest cut
if cut_info:
    print(f"\nIndividual cut impacts:")
    for cut_name, n_removed_by_cut, n_before_cut in cut_info:
        pct = 100 * n_removed_by_cut / n_before_cut if n_before_cut > 0 else 0
        print(f"  {cut_name}: removed {n_removed_by_cut:,} ({pct:.1f}%)")
    
    # Find biggest cut
    biggest_cut = max(cut_info, key=lambda x: x[1])
    print(f"\n*** BIGGEST CUT: {biggest_cut[0]} ***")
    print(f"    Removed {biggest_cut[1]:,} galaxies ({100*biggest_cut[1]/biggest_cut[2]:.1f}% of {biggest_cut[2]:,} galaxies before this cut)")

output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', 'alfalfa_nsa_final_13props.pkl')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as f_out:
    pickle.dump(data_dict, f_out)

print(f"\nData saved to: {output_file}")

