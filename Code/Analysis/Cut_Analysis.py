"""
Analyze the effects of physical parameter cuts on both datasets.
Tracks how many galaxies each cut removes and documents their effects.
"""
import os
import pickle
import numpy as np
from astropy.io import fits
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Get FITS file paths from environment variables or use defaults
ALFALFA_FITS = os.environ.get('ALFALFA_FITS_PATH', os.path.join(REPO_ROOT, 'Data', '5asfullmatch.fits'))
NSA_FITS = os.environ.get('NSA_FITS_PATH', os.path.join(REPO_ROOT, 'Data', 'nsa_v1_0_1.fits'))
OUTPUT_DIR = os.path.join(REPO_ROOT, "Docs")

def analyze_alfalfa_cuts():
    """Analyze cuts for ALFALFA × NSA dataset."""
    print("=" * 70)
    print("ALFALFA × NSA CUT ANALYSIS")
    print("=" * 70)
    
    with fits.open(ALFALFA_FITS, memmap=True) as hdul:
        data = hdul[1].data
        n_total = len(data)
        print(f"\nTotal galaxies in FITS file: {n_total:,}")
        
        # Extract properties
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
        'ELPETRO_ABSMAG_R': elpetro_absmag_r
    }
    
    # Track non-finite values
    print("\n=== NON-FINITE VALUE REMOVAL ===")
    valid_mask = np.ones(n_total, dtype=bool)
    non_finite_stats = {}
    for var_name, var_data in data_dict.items():
        finite_mask = np.isfinite(var_data)
        n_bad = np.count_nonzero(~finite_mask)
        if n_bad > 0:
            non_finite_stats[var_name] = n_bad
            print(f"{var_name}: {n_bad:,} non-finite entries ({100*n_bad/n_total:.2f}%)")
        valid_mask &= finite_mask
    
    n_after_finite = np.count_nonzero(valid_mask)
    print(f"\nAfter removing non-finite values: {n_after_finite:,} / {n_total:,} galaxies")
    print(f"Removed: {n_total - n_after_finite:,} galaxies ({100*(n_total - n_after_finite)/n_total:.2f}%)")
    
    # Apply finite mask
    for key in data_dict:
        data_dict[key] = data_dict[key][valid_mask]
    
    # Track individual cuts
    print("\n=== PHYSICAL PARAMETER CUTS ===")
    n_before_cuts = len(data_dict['ELPETRO_MASS'])
    cut_mask = np.ones(n_before_cuts, dtype=bool)
    cut_stats = []
    
    cuts = [
        ('BARYONIC_MASS', (data_dict['BARYONIC_MASS'] > 1e6) & (data_dict['BARYONIC_MASS'] < 1e12), 
         '10^6 < M_baryon < 10^12 M_sun', 'Removes extreme mass outliers'),
        ('COLOR_U_R', (data_dict['COLOR_U_R'] >= -0.5) & (data_dict['COLOR_U_R'] <= 4.0),
         '-0.5 <= (u-r) <= 4.0 mag', 'Removes unrealistic colors, keeps blue to red sequence'),
        ('ELPETRO_B300', (data_dict['ELPETRO_B300'] > 1e-8) & (data_dict['ELPETRO_B300'] < 10.0),
         '10^-8 < B300 < 10', 'Removes extreme star formation rate outliers'),
        ('SERSIC_N', (data_dict['SERSIC_N'] > 0) & (data_dict['SERSIC_N'] < 6.0),
         '0 < n < 6', 'Removes unphysical Sersic indices'),
        ('ELPETRO_METS', (data_dict['ELPETRO_METS'] > -2.5) & (data_dict['ELPETRO_METS'] < 0.5),
         '-2.5 < Z < 0.5 dex', 'Removes extreme metallicity values'),
        ('ELPETRO_MTOL', (data_dict['ELPETRO_MTOL'] >= 0.1) & (data_dict['ELPETRO_MTOL'] <= 10.0),
         '0.1 <= M/L <= 10 M_sun/L_sun', 'Removes unrealistic mass-to-light ratios'),
        ('ELPETRO_BA', (data_dict['ELPETRO_BA'] > 0) & (data_dict['ELPETRO_BA'] < 1.0),
         '0 < b/a < 1', 'Removes unphysical axis ratios'),
        ('ELPETRO_TH50_R', (data_dict['ELPETRO_TH50_R'] > 0) & (data_dict['ELPETRO_TH50_R'] < 25.0),
         '0 < r_50 < 25 arcsec', 'Removes extreme size measurements'),
        ('ZDIST', data_dict['ZDIST'] < 0.15,
         'z < 0.15', 'Focuses on local universe, avoids evolution effects'),
        ('logMH', (data_dict['logMH'] >= 6.0) & (data_dict['logMH'] <= 10.5),
         '6 <= log M_HI <= 10.5', 'Removes extreme HI mass outliers'),
        ('ELPETRO_MASS', (data_dict['ELPETRO_MASS'] > 1e6) & (data_dict['ELPETRO_MASS'] < 1e12),
         '10^6 < M* < 10^12 M_sun', 'Removes extreme stellar mass outliers'),
        ('ELPETRO_ABSMAG_R', (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0),
         '-25 < M_r < -10 mag', 'Removes extreme absolute magnitude values'),
    ]
    
    for var_name, cut, description, effect in cuts:
        n_removed = np.count_nonzero(~cut)
        n_remaining = np.count_nonzero(cut_mask & cut)
        pct_removed = 100 * n_removed / n_before_cuts
        pct_remaining = 100 * n_remaining / n_before_cuts
        cut_stats.append({
            'Property': var_name,
            'Cut': description,
            'Removed': n_removed,
            'Removed_%': pct_removed,
            'Remaining': n_remaining,
            'Remaining_%': pct_remaining,
            'Effect': effect
        })
        cut_mask &= cut
        print(f"{var_name}: Removed {n_removed:,} ({pct_removed:.2f}%), Remaining: {n_remaining:,} ({pct_remaining:.2f}%)")
    
    n_final = np.count_nonzero(cut_mask)
    print(f"\nFinal dataset: {n_final:,} / {n_total:,} galaxies")
    print(f"Total removed: {n_total - n_final:,} ({100*(n_total - n_final)/n_total:.2f}%)")
    
    return pd.DataFrame(cut_stats), n_total, n_after_finite, n_final

def analyze_nsa_cuts():
    """Analyze cuts for NSA-only dataset."""
    print("\n" + "=" * 70)
    print("NSA-ONLY CUT ANALYSIS")
    print("=" * 70)
    
    with fits.open(NSA_FITS, memmap=True) as hdul:
        data = hdul[1].data
        n_total = len(data)
        print(f"\nTotal galaxies in FITS file: {n_total:,}")
        
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
    
    # Track non-finite values
    print("\n=== NON-FINITE VALUE REMOVAL ===")
    valid_mask = np.ones(n_total, dtype=bool)
    for var_name, var_data in data_dict.items():
        finite_mask = np.isfinite(var_data)
        n_bad = np.count_nonzero(~finite_mask)
        if n_bad > 0:
            print(f"{var_name}: {n_bad:,} non-finite entries ({100*n_bad/n_total:.2f}%)")
        valid_mask &= finite_mask
    
    n_after_finite = np.count_nonzero(valid_mask)
    print(f"\nAfter removing non-finite values: {n_after_finite:,} / {n_total:,} galaxies")
    print(f"Removed: {n_total - n_after_finite:,} galaxies ({100*(n_total - n_after_finite)/n_total:.2f}%)")
    
    for key in data_dict:
        data_dict[key] = data_dict[key][valid_mask]
    
    # Track individual cuts
    print("\n=== PHYSICAL PARAMETER CUTS ===")
    n_before_cuts = len(data_dict['ELPETRO_MASS'])
    cut_mask = np.ones(n_before_cuts, dtype=bool)
    cut_stats = []
    
    cuts = [
        ('COLOR_U_R', (data_dict['COLOR_U_R'] >= -0.5) & (data_dict['COLOR_U_R'] <= 4.0),
         '-0.5 <= (u-r) <= 4.0 mag', 'Removes unrealistic colors'),
        ('ELPETRO_B300', (data_dict['ELPETRO_B300'] > 1e-8) & (data_dict['ELPETRO_B300'] < 10.0),
         '10^-8 < B300 < 10', 'Removes extreme star formation rate outliers'),
        ('SERSIC_N', (data_dict['SERSIC_N'] > 0) & (data_dict['SERSIC_N'] < 6.0),
         '0 < n < 6', 'Removes unphysical Sersic indices'),
        ('ELPETRO_METS', (data_dict['ELPETRO_METS'] > -2.5) & (data_dict['ELPETRO_METS'] < 0.5),
         '-2.5 < Z < 0.5 dex', 'Removes extreme metallicity values'),
        ('ELPETRO_MTOL', (data_dict['ELPETRO_MTOL'] >= 0.1) & (data_dict['ELPETRO_MTOL'] <= 10.0),
         '0.1 <= M/L <= 10 M_sun/L_sun', 'Removes unrealistic mass-to-light ratios'),
        ('ELPETRO_BA', (data_dict['ELPETRO_BA'] > 0) & (data_dict['ELPETRO_BA'] < 1.0),
         '0 < b/a < 1', 'Removes unphysical axis ratios'),
        ('ELPETRO_TH50_R', (data_dict['ELPETRO_TH50_R'] > 0) & (data_dict['ELPETRO_TH50_R'] < 25.0),
         '0 < r_50 < 25 arcsec', 'Removes extreme size measurements'),
        ('ZDIST', data_dict['ZDIST'] < 0.15,
         'z < 0.15', 'Focuses on local universe'),
        ('ELPETRO_MASS', (data_dict['ELPETRO_MASS'] > 1e6) & (data_dict['ELPETRO_MASS'] < 1e12),
         '10^6 < M* < 10^12 M_sun', 'Removes extreme stellar mass outliers'),
        ('ELPETRO_ABSMAG_R', (data_dict['ELPETRO_ABSMAG_R'] > -25.0) & (data_dict['ELPETRO_ABSMAG_R'] < -10.0),
         '-25 < M_r < -10 mag', 'Removes extreme absolute magnitude values'),
    ]
    
    for var_name, cut, description, effect in cuts:
        n_removed = np.count_nonzero(~cut)
        n_remaining = np.count_nonzero(cut_mask & cut)
        pct_removed = 100 * n_removed / n_before_cuts
        pct_remaining = 100 * n_remaining / n_before_cuts
        cut_stats.append({
            'Property': var_name,
            'Cut': description,
            'Removed': n_removed,
            'Removed_%': pct_removed,
            'Remaining': n_remaining,
            'Remaining_%': pct_remaining,
            'Effect': effect
        })
        cut_mask &= cut
        print(f"{var_name}: Removed {n_removed:,} ({pct_removed:.2f}%), Remaining: {n_remaining:,} ({pct_remaining:.2f}%)")
    
    n_final = np.count_nonzero(cut_mask)
    print(f"\nFinal dataset: {n_final:,} / {n_total:,} galaxies")
    print(f"Total removed: {n_total - n_final:,} ({100*(n_total - n_final)/n_total:.2f}%)")
    
    return pd.DataFrame(cut_stats), n_total, n_after_finite, n_final

def main():
    """Run cut analysis for both datasets."""
    alfalfa_df, alfalfa_total, alfalfa_after_finite, alfalfa_final = analyze_alfalfa_cuts()
    nsa_df, nsa_total, nsa_after_finite, nsa_final = analyze_nsa_cuts()
    
    # Create markdown document
    output_path = os.path.join(OUTPUT_DIR, "Cut_Analysis.md")
    
    with open(output_path, 'w') as f:
        f.write("# Physical Parameter Cut Analysis\n\n")
        f.write("This document analyzes the effects of physical parameter cuts applied to both datasets.\n\n")
        
        f.write("## ALFALFA × NSA Dataset\n\n")
        f.write(f"- **Initial galaxies**: {alfalfa_total:,}\n")
        f.write(f"- **After non-finite removal**: {alfalfa_after_finite:,} ({100*alfalfa_after_finite/alfalfa_total:.2f}%)\n")
        f.write(f"- **Final galaxies**: {alfalfa_final:,} ({100*alfalfa_final/alfalfa_total:.2f}%)\n")
        f.write(f"- **Total removed**: {alfalfa_total - alfalfa_final:,} ({100*(alfalfa_total - alfalfa_final)/alfalfa_total:.2f}%)\n\n")
        
        f.write("### Cut Statistics (Ordered by Number Removed)\n\n")
        alfalfa_sorted = alfalfa_df.sort_values('Removed', ascending=False)
        f.write("| Property | Cut Range | Removed | Removed % | Remaining | Remaining % | Effect |\n")
        f.write("|----------|-----------|---------|-----------|-----------|-------------|--------|\n")
        for _, row in alfalfa_sorted.iterrows():
            f.write(f"| {row['Property']} | {row['Cut']} | {row['Removed']:,} | {row['Removed_%']:.2f}% | "
                   f"{row['Remaining']:,} | {row['Remaining_%']:.2f}% | {row['Effect']} |\n")
        
        f.write("\n## NSA-only Dataset\n\n")
        f.write(f"- **Initial galaxies**: {nsa_total:,}\n")
        f.write(f"- **After non-finite removal**: {nsa_after_finite:,} ({100*nsa_after_finite/nsa_total:.2f}%)\n")
        f.write(f"- **Final galaxies**: {nsa_final:,} ({100*nsa_final/nsa_total:.2f}%)\n")
        f.write(f"- **Total removed**: {nsa_total - nsa_final:,} ({100*(nsa_total - nsa_final)/nsa_total:.2f}%)\n\n")
        
        f.write("### Cut Statistics (Ordered by Number Removed)\n\n")
        nsa_sorted = nsa_df.sort_values('Removed', ascending=False)
        f.write("| Property | Cut Range | Removed | Removed % | Remaining | Remaining % | Effect |\n")
        f.write("|----------|-----------|---------|-----------|-----------|-------------|--------|\n")
        for _, row in nsa_sorted.iterrows():
            f.write(f"| {row['Property']} | {row['Cut']} | {row['Removed']:,} | {row['Removed_%']:.2f}% | "
                   f"{row['Remaining']:,} | {row['Remaining_%']:.2f}% | {row['Effect']} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("### Most Restrictive Cuts\n\n")
        f.write("**ALFALFA × NSA:**\n")
        top_3_alfalfa = alfalfa_sorted.head(3)
        for _, row in top_3_alfalfa.iterrows():
            f.write(f"- **{row['Property']}**: Removes {row['Removed']:,} galaxies ({row['Removed_%']:.2f}%)\n")
        
        f.write("\n**NSA-only:**\n")
        top_3_nsa = nsa_sorted.head(3)
        for _, row in top_3_nsa.iterrows():
            f.write(f"- **{row['Property']}**: Removes {row['Removed']:,} galaxies ({row['Removed_%']:.2f}%)\n")
        
        f.write("\n### Scientific Rationale\n\n")
        f.write("These cuts are designed to:\n")
        f.write("1. Remove measurement errors and unphysical values\n")
        f.write("2. Focus on the local universe (z < 0.15) to minimize evolution effects\n")
        f.write("3. Ensure all properties are within physically reasonable ranges\n")
        f.write("4. Maintain a clean sample for causal inference analysis\n")
    
    print(f"\n{'='*70}")
    print(f"Cut analysis saved to: {output_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

