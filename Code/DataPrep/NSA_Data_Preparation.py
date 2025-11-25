"""
Data preparation script for NSA-only dataset.
Extracts galaxy properties, applies quality cuts, and saves processed data.
"""

import os
import pickle
import numpy as np
from astropy.io import fits

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_FITS_PATH = r'C:\Users\sanji\Downloads\nsa_v1_0_1.fits'
DEFAULT_FITS_PATH_ROOT = os.path.join(REPO_ROOT, 'nsa_v1_0_1.fits')
DEFAULT_FITS_PATH_DATA = os.path.join(REPO_ROOT, 'Data', 'nsa_v1_0_1.fits')

if 'NSA_FITS_PATH' in os.environ:
    FITS_PATH = os.environ.get('NSA_FITS_PATH')
elif os.path.exists(DEFAULT_FITS_PATH):
    FITS_PATH = DEFAULT_FITS_PATH
elif os.path.exists(DEFAULT_FITS_PATH_ROOT):
    FITS_PATH = DEFAULT_FITS_PATH_ROOT
else:
    FITS_PATH = DEFAULT_FITS_PATH_DATA

OUTPUT_DIR = os.path.join(REPO_ROOT, "Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nsa_final_10props.pkl")

def main():
    if not os.path.exists(FITS_PATH):
        raise FileNotFoundError(
            f"FITS file not found: {FITS_PATH}\n"
            f"Set NSA_FITS_PATH environment variable or place file at:\n"
            f"  - {DEFAULT_FITS_PATH}\n"
            f"  - {DEFAULT_FITS_PATH_ROOT}\n"
            f"  - {DEFAULT_FITS_PATH_DATA}"
        )

    with fits.open(FITS_PATH, memmap=True) as hdul:
        data = hdul[1].data
        n_galaxies = len(data)

        sersic_n = np.asarray(data["SERSIC_N"])
        zdist = np.asarray(data["ZDIST"])
        elpetro_b300 = np.asarray(data["ELPETRO_B300"])
        elpetro_mets = np.asarray(data["ELPETRO_METS"])
        elpetro_mtol_all = np.asarray(data["ELPETRO_MTOL"])
        elpetro_ba = np.asarray(data["ELPETRO_BA"])
        elpetro_th50_r = np.asarray(data["ELPETRO_TH50_R"])
        elpetro_mass = np.asarray(data["ELPETRO_MASS"])
        elpetro_absmag = np.asarray(data["ELPETRO_ABSMAG"])

    elpetro_mtol_r = elpetro_mtol_all[:, 4]
    color_u_r = elpetro_absmag[:, 2] - elpetro_absmag[:, 4]
    elpetro_absmag_r = elpetro_absmag[:, 4]

    log_stellar_mass = np.log10(np.maximum(elpetro_mass, 1e-10))

    data_dict = {
        "COLOR_U_R": color_u_r,
        "ELPETRO_B300": elpetro_b300,
        "SERSIC_N": sersic_n,
        "ELPETRO_METS": elpetro_mets,
        "ELPETRO_MTOL": elpetro_mtol_r,
        "ELPETRO_BA": elpetro_ba,
        "ELPETRO_TH50_R": elpetro_th50_r,
        "ZDIST": zdist,
        "ELPETRO_MASS": log_stellar_mass,
        "ELPETRO_ABSMAG_R": elpetro_absmag_r,
    }

    valid_mask = np.ones(n_galaxies, dtype=bool)
    for name, values in data_dict.items():
        finite = np.isfinite(values)
        valid_mask &= finite

    n_valid = np.count_nonzero(valid_mask)
    if n_valid == 0:
        raise RuntimeError("No valid rows remain after filtering.")

    for key in data_dict:
        data_dict[key] = data_dict[key][valid_mask]

    n_before_cuts = len(data_dict['ELPETRO_MASS'])
    cut_mask = np.ones(n_before_cuts, dtype=bool)

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

    cut = (data_dict['ELPETRO_MASS'] > 6.0) & (data_dict['ELPETRO_MASS'] < 12.0)
    cut_mask &= cut

    cut = (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0)
    cut_mask &= cut

    for key in data_dict:
        data_dict[key] = data_dict[key][cut_mask]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as fp:
        pickle.dump(data_dict, fp)

if __name__ == "__main__":
    main()
