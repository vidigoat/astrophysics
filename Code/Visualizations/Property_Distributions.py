"""
Create multi-panel histogram figure showing distributions of all 13 properties
with cut ranges marked, suitable for MNRAS publication.
Shows full histogram including cut portions, with red lines marking cut limits.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from astropy.io import fits

# Set up paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'Plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get FITS file path
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

with fits.open(FITS_PATH, memmap=True) as hdul:
    data = hdul[1].data
    n_rows = len(data)

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
w50 = np.asarray(data['W50'])
elpetro_absmag = np.asarray(data['ELPETRO_ABSMAG'])

color_u_r = elpetro_absmag[:, 2] - elpetro_absmag[:, 4]
elpetro_absmag_r = elpetro_absmag[:, 4]
M_HI = 10 ** logMH
baryonic_mass = elpetro_mass + 1.4 * M_HI

# Create data dictionary with raw values
raw_data = {
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

for key in raw_data:
    mask = np.isfinite(raw_data[key])
    raw_data[key] = raw_data[key][mask]

# Define property information: (property_name, label, unit, cut_low, cut_high, log_scale)
# Using clearer, more descriptive labels for paper
PROPERTIES = [
    ('BARYONIC_MASS', 'Baryonic Mass', r'$M_{\odot}$', 1e6, 1e12, True),
    ('COLOR_U_R', 'Colour (u-r)', 'mag', -0.5, 4.0, False),
    ('ELPETRO_B300', r'$B_{300}$', '', 1e-8, 10.0, True),
    ('SERSIC_N', 'Sersic n', '', 0.0, 6.0, False),
    ('ELPETRO_METS', 'Metallicity', 'dex', -2.5, 0.5, False),
    ('ELPETRO_MTOL', 'Mass-to-Light Ratio', r'$M_{\odot}/L_{\odot}$', 0.1, 10.0, False),
    ('ELPETRO_BA', 'Axis Ratio (b/a)', '', 0.0, 1.0, False),
    ('ELPETRO_TH50_R', 'Half-light Radius', 'arcsec', 0.0, 25.0, False),
    ('ZDIST', 'Redshift', '', 0.0, 0.15, False),
    ('logMH', r'$\log(M_{\rm HI}/M_{\odot})$', '', 6.0, 10.5, False),
    ('ELPETRO_MASS', 'Stellar Mass', r'$M_{\odot}$', 1e6, 1e12, True),
    ('ELPETRO_ABSMAG_R', r'$M_r$', 'mag', -25.0, -10.0, False),
    ('W50', r'$W_{50}$', 'km s$^{-1}$', 20.0, 500.0, False),
]

# Create figure with 4x4 grid (13 panels + 3 empty)
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

for idx, (prop_name, label, unit, cut_low, cut_high, use_log) in enumerate(PROPERTIES):
    ax = axes[idx]
    values = raw_data[prop_name]
    
    # Determine binning - show full range including cut portions
    if use_log:
        # Log scale: filter out non-positive values for log calculation
        positive_mask = values > 0
        if np.any(positive_mask):
            log_values = np.log10(values[positive_mask])
            log_cut_low = np.log10(cut_low)
            log_cut_high = np.log10(cut_high)
            
            # Create bins covering full range, extending beyond cut limits
            log_min = min(log_values.min(), log_cut_low - 0.3)
            log_max = max(log_values.max(), log_cut_high + 0.3)
            bins = np.logspace(log_min, log_max, 60)
            
            # Filter values for histogram (only positive)
            hist_values = values[positive_mask]
        else:
            # Fallback if no positive values
            bins = np.logspace(np.log10(cut_low) - 0.5, np.log10(cut_high) + 0.5, 60)
            hist_values = values
        
        # Plot histogram of full distribution
        n, bins_edges, patches = ax.hist(hist_values, bins=bins, color='steelblue', 
                                        alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xscale('log')
        
        # Mark cut limits with red lines (zorder=3 so they're visible but don't cover labels)
        ax.axvline(cut_low, color='red', linestyle='--', linewidth=2.0, 
                  label='Cut limits', zorder=3)
        ax.axvline(cut_high, color='red', linestyle='--', linewidth=2.0, zorder=3)
        
        # Format x-axis - ensure valid limits
        if len(bins) > 0 and np.isfinite(bins[0]) and np.isfinite(bins[-1]):
            ax.set_xlim(bins[0], bins[-1])
        
    else:
        # Linear scale
        if prop_name == 'ELPETRO_ABSMAG_R':
            # Magnitude: extend range and reverse axis
            val_min = values.min()
            val_max = values.max()
            range_extend = (val_max - val_min) * 0.15
            bins = np.linspace(val_min - range_extend, val_max + range_extend, 60)
            n, bins_edges, patches = ax.hist(values, bins=bins, color='steelblue', 
                                            alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(cut_low, color='red', linestyle='--', linewidth=2.0, 
                      label='Cut limits', zorder=3)
            ax.axvline(cut_high, color='red', linestyle='--', linewidth=2.0, zorder=3)
            ax.invert_xaxis()  # Invert for magnitudes
        else:
            # Extend range to show full distribution including cut portions
            val_min = values.min()
            val_max = values.max()
            range_extend = (val_max - val_min) * 0.15
            bins = np.linspace(val_min - range_extend, val_max + range_extend, 60)
            n, bins_edges, patches = ax.hist(values, bins=bins, color='steelblue', 
                                            alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Mark cut limits with red lines (zorder=3 so they're visible but don't cover labels)
            ax.axvline(cut_low, color='red', linestyle='--', linewidth=2.0, 
                      label='Cut limits', zorder=3)
            ax.axvline(cut_high, color='red', linestyle='--', linewidth=2.0, zorder=3)
    
    # Set labels
    if unit:
        ax.set_xlabel(f'{label} ({unit})', fontsize=10)
    else:
        ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Number', fontsize=10)
    
    # Fix x-axis tick labels - rotate and adjust spacing to avoid overlap
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # For properties with many ticks, reduce number of ticks
    if prop_name in ['ELPETRO_TH50_R', 'COLOR_U_R', 'SERSIC_N', 'ELPETRO_BA']:
        # Reduce number of x-axis ticks for better readability
        ax.locator_params(axis='x', nbins=6)
    
    # Ensure red lines are behind tick labels (lower z-order for lines)
    # Redraw red lines with lower z-order so they don't cover tick labels
    if use_log:
        ax.axvline(cut_low, color='red', linestyle='--', linewidth=2.0, zorder=3)
        ax.axvline(cut_high, color='red', linestyle='--', linewidth=2.0, zorder=3)
    else:
        if prop_name == 'ELPETRO_ABSMAG_R':
            ax.axvline(cut_low, color='red', linestyle='--', linewidth=2.0, zorder=3)
            ax.axvline(cut_high, color='red', linestyle='--', linewidth=2.0, zorder=3)
        else:
            ax.axvline(cut_low, color='red', linestyle='--', linewidth=2.0, zorder=3)
            ax.axvline(cut_high, color='red', linestyle='--', linewidth=2.0, zorder=3)
    
    # Add statistics text - position in upper left
    median_val = np.median(values)
    mean_val = np.mean(values)
    if use_log:
        stats_text = f'Median: {median_val:.2e}\nMean: {mean_val:.2e}'
    else:
        if abs(median_val) < 0.01 or abs(median_val) > 1000:
            stats_text = f'Median: {median_val:.2e}\nMean: {mean_val:.2e}'
        else:
            stats_text = f'Median: {median_val:.2f}\nMean: {mean_val:.2f}'
    
    # Position stats box in upper left corner
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=0.5))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

# Remove empty subplots
for idx in range(len(PROPERTIES), 16):
    fig.delaxes(axes[idx])

# Add legend
red_line = mpatches.Patch(color='red', linestyle='--', label='Cut limits')
blue_patch = mpatches.Patch(color='steelblue', alpha=0.7, label='Distribution')

# Add legend in bottom right corner
fig.legend([blue_patch, red_line], 
          ['Distribution', 'Cut limits'],
          loc='lower right', fontsize=10, framealpha=0.9)

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 1.0])
plt.savefig(os.path.join(OUTPUT_DIR, 'property_distributions_with_cuts.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

