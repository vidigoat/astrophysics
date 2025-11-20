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

with fits.open(FITS_PATH, memmap=True) as hdul:
    data = hdul[1].data
    n_rows = len(data)

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

valid_mask = np.ones(n_rows, dtype=bool)
for var_name, var_data in data_dict.items():
    finite_mask = np.isfinite(var_data)
    valid_mask &= finite_mask

for key in data_dict:
    data_dict[key] = data_dict[key][valid_mask]

n_before_cuts = len(data_dict['ELPETRO_MASS'])
cut_mask = np.ones(n_before_cuts, dtype=bool)

cut = (data_dict['BARYONIC_MASS'] > 1e6) & (data_dict['BARYONIC_MASS'] < 1e12)
cut_mask &= cut

cut = (data_dict['COLOR_U_R'] >= -0.5) & (data_dict['COLOR_U_R'] <= 4.0)
cut_mask &= cut

cut = (data_dict['ELPETRO_B300'] > 1e-8) & (data_dict['ELPETRO_B300'] < 10.0)
cut_mask &= cut

cut = (data_dict['SERSIC_N'] > 0) & (data_dict['SERSIC_N'] < 6.0)
cut_mask &= cut

cut = (data_dict['ELPETRO_METS'] > -2.5) & (data_dict['ELPETRO_METS'] < 0.5)
cut_mask &= cut

cut = (data_dict['ELPETRO_MTOL'] >= 0.1) & (data_dict['ELPETRO_MTOL'] <= 10.0)
cut_mask &= cut

cut = (data_dict['ELPETRO_BA'] > 0) & (data_dict['ELPETRO_BA'] < 1.0)
cut_mask &= cut

cut = (data_dict['ELPETRO_TH50_R'] > 0) & (data_dict['ELPETRO_TH50_R'] < 25.0)
cut_mask &= cut

cut = data_dict['ZDIST'] < 0.15
cut_mask &= cut

cut = (data_dict['logMH'] >= 6.0) & (data_dict['logMH'] <= 10.5)
cut_mask &= cut

cut = (data_dict['ELPETRO_MASS'] > 1e6) & (data_dict['ELPETRO_MASS'] < 1e12)
cut_mask &= cut

cut = (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0)
cut_mask &= cut

cut = (data_dict['W50'] > 20.0) & (data_dict['W50'] < 500.0)
cut_mask &= cut

for key in data_dict:
    data_dict[key] = data_dict[key][cut_mask]

output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', 'alfalfa_nsa_final_13props.pkl')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as f_out:
    pickle.dump(data_dict, f_out)

