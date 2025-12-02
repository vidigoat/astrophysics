"""
Data preparation script for Illustris TNG50 simulation dataset.
Extracts subhalo properties from HDF5 file, applies quality cuts, and saves processed data.
"""

import os
import pickle
import numpy as np
import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_HDF5_PATH = r'C:\Users\sanji\Downloads\subhalos_mstar_gt1e8.hdf5'
DEFAULT_HDF5_PATH_ROOT = os.path.join(REPO_ROOT, 'subhalos_mstar_gt1e8.hdf5')
DEFAULT_HDF5_PATH_DATA = os.path.join(REPO_ROOT, 'Data', 'subhalos_mstar_gt1e8.hdf5')

if 'TNG50_HDF5_PATH' in os.environ:
    HDF5_PATH = os.environ.get('TNG50_HDF5_PATH')
elif os.path.exists(DEFAULT_HDF5_PATH):
    HDF5_PATH = DEFAULT_HDF5_PATH
elif os.path.exists(DEFAULT_HDF5_PATH_ROOT):
    HDF5_PATH = DEFAULT_HDF5_PATH_ROOT
else:
    HDF5_PATH = DEFAULT_HDF5_PATH_DATA

OUTPUT_DIR = os.path.join(REPO_ROOT, "Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tng50_final.pkl")

def main():
    if not os.path.exists(HDF5_PATH):
        raise FileNotFoundError(
            f"HDF5 file not found: {HDF5_PATH}\n"
            f"Set TNG50_HDF5_PATH environment variable or place file at:\n"
            f"  - {DEFAULT_HDF5_PATH}\n"
            f"  - {DEFAULT_HDF5_PATH_ROOT}\n"
            f"  - {DEFAULT_HDF5_PATH_DATA}"
        )

    print(f"Loading TNG50 data from: {HDF5_PATH}")
    
    with h5py.File(HDF5_PATH, 'r') as f:
        n_subhalos = len(f[list(f.keys())[0]])
        print(f"Total subhalos in file: {n_subhalos:,}")
        
        # Extract subhalo properties from HDF5 file
        mass_type = np.asarray(f['SubhaloMassType'])
        dm_mass = mass_type[:, 0]
        stellar_mass = mass_type[:, 1]
        gas_mass = mass_type[:, 4]
        bh_mass = mass_type[:, 5]
        halfmass_rad = np.asarray(f['SubhaloHalfmassRad'])
        vel_disp = np.asarray(f['SubhaloVelDisp'])
        vmax = np.asarray(f['SubhaloVmax'])
        gas_metallicity = np.asarray(f['SubhaloGasMetallicity'])
        star_metallicity = np.asarray(f['SubhaloStarMetallicity'])
        photometrics = np.asarray(f['SubhaloStellarPhotometrics'])
        photometric_u = photometrics[:, 0]
        photometric_r = photometrics[:, 2]
        sfr = np.asarray(f['SubhaloSFR'])
        
        data_dict_raw = {
            'DM_MASS': dm_mass,
            'STELLAR_MASS': stellar_mass,
            'GAS_MASS': gas_mass,
            'BH_MASS': bh_mass,
            'HALFMASS_RAD': halfmass_rad,
            'VEL_DISP': vel_disp,
            'VMAX': vmax,
            'GAS_METALLICITY': gas_metallicity,
            'STAR_METALLICITY': star_metallicity,
            'PHOTOMETRIC_U': photometric_u,
            'PHOTOMETRIC_R': photometric_r,
            'SFR': sfr
        }

    # Filter for finite values
    valid_mask = np.ones(n_subhalos, dtype=bool)
    for name, values in data_dict_raw.items():
        finite = np.isfinite(values)
        valid_mask &= finite

    n_valid = np.count_nonzero(valid_mask)
    if n_valid == 0:
        raise RuntimeError("No valid rows remain after filtering.")
    
    print(f"Subhalos with finite values: {n_valid:,} ({100*n_valid/n_subhalos:.1f}%)")

    for key in data_dict_raw:
        data_dict_raw[key] = data_dict_raw[key][valid_mask]

    # Convert masses to log space (TNG50 masses need scaling by 10^10 before log conversion)
    dm_mass_log = np.log10(np.maximum(data_dict_raw['DM_MASS'] * 1e10, 1e-10))
    stellar_mass_log = np.log10(np.maximum(data_dict_raw['STELLAR_MASS'] * 1e10, 1e-10))
    gas_mass_log = np.log10(np.maximum(data_dict_raw['GAS_MASS'] * 1e10, 1e-10))
    bh_mass_log = np.log10(np.maximum(data_dict_raw['BH_MASS'] * 1e10, 1e-10))
    
    # Calculate baryonic mass as sum of stellar and gas mass
    stellar_mass_linear = 10**stellar_mass_log
    gas_mass_linear = 10**gas_mass_log
    baryonic_mass_linear = stellar_mass_linear + gas_mass_linear
    baryonic_mass_log = np.log10(np.maximum(baryonic_mass_linear, 1e-10))
    
    # Calculate color as difference between U and R band magnitudes
    colour = data_dict_raw['PHOTOMETRIC_U'] - data_dict_raw['PHOTOMETRIC_R']
    data_dict = {
        'DM_MASS': dm_mass_log,
        'STELLAR_MASS': stellar_mass_log,
        'GAS_MASS': gas_mass_log,
        'BH_MASS': bh_mass_log,
        'BARYONIC_MASS': baryonic_mass_log,
        'HALFMASS_RAD': data_dict_raw['HALFMASS_RAD'],
        'VEL_DISP': data_dict_raw['VEL_DISP'],
        'VMAX': data_dict_raw['VMAX'],
        'GAS_METALLICITY': data_dict_raw['GAS_METALLICITY'],
        'STAR_METALLICITY': data_dict_raw['STAR_METALLICITY'],
        'PHOTOMETRIC_U': data_dict_raw['PHOTOMETRIC_U'],
        'PHOTOMETRIC_R': data_dict_raw['PHOTOMETRIC_R'],
        'SFR': data_dict_raw['SFR'],
        'COLOUR': colour
    }

    # Apply quality cuts to remove outliers and invalid values
    n_before_cuts = len(list(data_dict.values())[0])
    cut_mask = np.ones(n_before_cuts, dtype=bool)

    # Mass cuts (log space, after scaling)
    cut_mask &= (data_dict['DM_MASS'] > 5.0) & (data_dict['DM_MASS'] < 13.0)
    cut_mask &= (data_dict['STELLAR_MASS'] > 6.0) & (data_dict['STELLAR_MASS'] < 14.0)
    cut_mask &= (data_dict['GAS_MASS'] > 6.5) & (data_dict['GAS_MASS'] < 11.5)
    cut_mask &= (data_dict['BH_MASS'] < 12.0)
    
    # Size cuts
    halfmass_rad_log = np.log10(np.maximum(data_dict['HALFMASS_RAD'], 1e-10))
    cut_mask &= (halfmass_rad_log > -1.5) & (halfmass_rad_log < 2.5)
    
    # Kinematic cuts
    cut_mask &= (data_dict['VEL_DISP'] > 2.0) & (data_dict['VEL_DISP'] < 300.0)
    cut_mask &= (data_dict['VMAX'] > 5.0) & (data_dict['VMAX'] < 500.0)
    
    # Metallicity cuts
    cut_mask &= (data_dict['GAS_METALLICITY'] >= 0.0) & (data_dict['GAS_METALLICITY'] < 0.15)
    cut_mask &= (data_dict['STAR_METALLICITY'] > 0.0005) & (data_dict['STAR_METALLICITY'] < 0.15)
    
    # Photometric cuts
    cut_mask &= (data_dict['PHOTOMETRIC_U'] > -26.0) & (data_dict['PHOTOMETRIC_U'] < -9.0)
    cut_mask &= (data_dict['PHOTOMETRIC_R'] > -27.0) & (data_dict['PHOTOMETRIC_R'] < -9.0)
    
    # Star formation and color cuts
    sfr_log = np.log10(np.maximum(data_dict['SFR'], 1e-10))
    cut_mask &= (sfr_log < 2.5)
    cut_mask &= (data_dict['COLOUR'] > -2.0) & (data_dict['COLOUR'] < 4.0)
    cut_mask &= (data_dict['BARYONIC_MASS'] > 7.0) & (data_dict['BARYONIC_MASS'] < 13.0)

    # Apply cuts
    for key in data_dict:
        data_dict[key] = data_dict[key][cut_mask]

    n_after_cuts = len(list(data_dict.values())[0])
    n_removed = n_before_cuts - n_after_cuts
    
    print(f"\nQuality cuts applied:")
    print(f"  Subhalos before cuts: {n_before_cuts:,}")
    print(f"  Subhalos after cuts: {n_after_cuts:,}")
    print(f"  Removed: {n_removed:,} ({100*n_removed/n_before_cuts:.1f}%)")
    print(f"\nProperties extracted: {list(data_dict.keys())}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as fp:
        pickle.dump(data_dict, fp)
    
    print(f"\nData saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

