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

    with h5py.File(HDF5_PATH, 'r') as f:
        # TODO: Extract properties from HDF5 file
        # Example structure (to be updated based on actual file structure):
        # property1 = np.asarray(f['group_name/property1'])
        # property2 = np.asarray(f['group_name/property2'])
        # ...
        
        # Placeholder: will be updated when actual properties are known
        print("Reading HDF5 file structure...")
        print("Top-level keys:", list(f.keys()))
        
        # TODO: Extract actual properties here
        # For now, creating empty structure
        data_dict = {}
        
        # Example structure (to be filled):
        # data_dict = {
        #     'PROPERTY1': property1,
        #     'PROPERTY2': property2,
        #     ...
        # }

    # Filter for finite values
    valid_mask = np.ones(len(list(data_dict.values())[0]) if data_dict else 0, dtype=bool)
    for name, values in data_dict.items():
        finite = np.isfinite(values)
        valid_mask &= finite

    n_valid = np.count_nonzero(valid_mask)
    if n_valid == 0:
        raise RuntimeError("No valid rows remain after filtering.")

    for key in data_dict:
        data_dict[key] = data_dict[key][valid_mask]

    # Apply quality cuts
    n_before_cuts = len(list(data_dict.values())[0]) if data_dict else 0
    cut_mask = np.ones(n_before_cuts, dtype=bool)

    # TODO: Add quality cuts based on physical constraints
    # Example:
    # cut = (data_dict['PROPERTY1'] > min_val) & (data_dict['PROPERTY1'] < max_val)
    # cut_mask &= cut

    for key in data_dict:
        data_dict[key] = data_dict[key][cut_mask]

    n_after_cuts = len(list(data_dict.values())[0]) if data_dict else 0
    print(f"Galaxies before cuts: {n_before_cuts:,}")
    print(f"Galaxies after cuts: {n_after_cuts:,}")
    print(f"Properties: {list(data_dict.keys())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as fp:
        pickle.dump(data_dict, fp)

    print(f"Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

