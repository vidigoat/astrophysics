import os
import pickle
import numpy as np
from astropy.io import fits

# Get FITS file path from environment variable or use default
FITS_PATH = os.environ.get('NSA_FITS_PATH', 'nsa_v1_0_1.fits')
# Use Data folder at project root (two levels up from this script)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nsa_final_10props.pkl")

def main():
    print("=" * 70)
    print("NSA DATA PREPARATION")
    print("=" * 70)
    print(f"\nLoading FITS file:\n  {FITS_PATH}")

    with fits.open(FITS_PATH, memmap=True) as hdul:
        data = hdul[1].data
        n_galaxies = len(data)
        print(f"\nLoaded HDU 1 with {n_galaxies:,} rows.")

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
    elpetro_absmag_r = elpetro_absmag[:, 4]  # r-band absolute magnitude

    data_dict = {
        "COLOR_U_R": color_u_r,
        "ELPETRO_B300": elpetro_b300,
        "SERSIC_N": sersic_n,
        "ELPETRO_METS": elpetro_mets,
        "ELPETRO_MTOL": elpetro_mtol_r,
        "ELPETRO_BA": elpetro_ba,
        "ELPETRO_TH50_R": elpetro_th50_r,
        "ZDIST": zdist,
        "ELPETRO_MASS": elpetro_mass,
        "ELPETRO_ABSMAG_R": elpetro_absmag_r,
    }

    print("\nRunning data quality checks...")
    valid_mask = np.ones(n_galaxies, dtype=bool)
    for name, values in data_dict.items():
        finite = np.isfinite(values)
        if not np.all(finite):
            print(f"  {name}: {np.count_nonzero(~finite)} non-finite entries")
        valid_mask &= finite

    n_valid = np.count_nonzero(valid_mask)
    print(f"\nGalaxies with complete data: {n_valid:,} / {n_galaxies:,}")
    if n_valid == 0:
        raise RuntimeError("No valid rows remain after filtering.")

    for key in data_dict:
        data_dict[key] = data_dict[key][valid_mask]

    # Apply physical parameter cuts
    print("\n=== APPLYING PHYSICAL PARAMETER CUTS ===")
    n_before_cuts = len(data_dict['ELPETRO_MASS'])
    cut_mask = np.ones(n_before_cuts, dtype=bool)

    # COLOR_U_R: -0.5 <= (u−r) <= 4
    cut = (data_dict['COLOR_U_R'] >= -0.5) & (data_dict['COLOR_U_R'] <= 4.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"COLOR_U_R: Removed {n_removed:,} galaxies (outside -0.5 - 4 mag)")
    cut_mask &= cut

    # ELPETRO_B300: 10^-8 < B300 < 10
    cut = (data_dict['ELPETRO_B300'] > 1e-8) & (data_dict['ELPETRO_B300'] < 10.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_B300: Removed {n_removed:,} galaxies (outside 10^-8 - 10)")
    cut_mask &= cut

    # SERSIC_N: 0 < n < 6
    cut = (data_dict['SERSIC_N'] > 0) & (data_dict['SERSIC_N'] < 6.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"SERSIC_N: Removed {n_removed:,} galaxies (outside 0 - 6)")
    cut_mask &= cut

    # ELPETRO_METS: -2.5 < Z < 0.5
    cut = (data_dict['ELPETRO_METS'] > -2.5) & (data_dict['ELPETRO_METS'] < 0.5)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_METS: Removed {n_removed:,} galaxies (outside -2.5 - 0.5 dex)")
    cut_mask &= cut

    # ELPETRO_MTOL: 0.1 <= M/L <= 10
    cut = (data_dict['ELPETRO_MTOL'] >= 0.1) & (data_dict['ELPETRO_MTOL'] <= 10.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_MTOL: Removed {n_removed:,} galaxies (outside 0.1 - 10 M_sun/L_sun)")
    cut_mask &= cut

    # ELPETRO_BA: 0 < b/a < 1
    cut = (data_dict['ELPETRO_BA'] > 0) & (data_dict['ELPETRO_BA'] < 1.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_BA: Removed {n_removed:,} galaxies (outside 0 - 1)")
    cut_mask &= cut

    # ELPETRO_TH50_R: 0 < r_50 < 25
    cut = (data_dict['ELPETRO_TH50_R'] > 0) & (data_dict['ELPETRO_TH50_R'] < 25.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_TH50_R: Removed {n_removed:,} galaxies (outside 0 - 25 arcsec)")
    cut_mask &= cut

    # ZDIST: z < 0.15
    cut = data_dict['ZDIST'] < 0.15
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ZDIST: Removed {n_removed:,} galaxies (z >= 0.15)")
    cut_mask &= cut

    # ELPETRO_MASS: 10^6 < M* < 10^12
    cut = (data_dict['ELPETRO_MASS'] > 1e6) & (data_dict['ELPETRO_MASS'] < 1e12)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_MASS: Removed {n_removed:,} galaxies (outside 10^6 - 10^12 M_sun)")
    cut_mask &= cut

    # ELPETRO_ABSMAG_R: -25 < M_r < -10 (reasonable absolute magnitude range)
    cut = (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0)
    n_removed = np.count_nonzero(~cut)
    if n_removed > 0:
        print(f"ELPETRO_ABSMAG_R: Removed {n_removed:,} galaxies (outside -25 - -10 mag)")
    cut_mask &= cut

    # Apply cuts
    n_after_cuts = np.count_nonzero(cut_mask)
    n_removed_by_cuts = n_before_cuts - n_after_cuts
    print(f"\nAfter parameter cuts: {n_after_cuts:,} / {n_before_cuts:,} galaxies")
    print(f"Removed by parameter cuts: {n_removed_by_cuts:,} galaxies")

    for key in data_dict:
        data_dict[key] = data_dict[key][cut_mask]

    print("\nSummary statistics (after filtering):")
    for name, values in data_dict.items():
        print(f"\n{name}")
        print(f"  Min    : {np.min(values):.6e}")
        print(f"  Max    : {np.max(values):.6e}")
        print(f"  Mean   : {np.mean(values):.6e}")
        print(f"  Median : {np.median(values):.6e}")
        print(f"  Std    : {np.std(values):.6e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as fp:
        pickle.dump(data_dict, fp)

    print("\n" + "=" * 70)
    print(f"Saved cleaned dataset: {OUTPUT_FILE}")
    print(f"Properties: {len(data_dict)}")
    print(f"Galaxies  : {len(next(iter(data_dict.values()))):,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
