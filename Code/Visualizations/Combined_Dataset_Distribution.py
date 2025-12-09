"""
Generate combined histogram distribution plot comparing all three datasets.
Shows all properties in a panel layout with consistent labels.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "combined_dataset_distribution.png")

# Load data
tng50_path = os.path.join(REPO_ROOT, "Data", "tng50_final.pkl")
alfalfa_path = os.path.join(REPO_ROOT, "Data", "alfalfa_nsa_final_13props.pkl")
nsa_path = os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl")

with open(tng50_path, "rb") as f:
    tng50_data = pickle.load(f)

with open(alfalfa_path, "rb") as f:
    alfalfa_data = pickle.load(f)

with open(nsa_path, "rb") as f:
    nsa_data = pickle.load(f)

# Property mapping: (tng50_key, alfalfa_key, nsa_key, display_name)
PROPERTIES = [
    # Mass properties
    ("STELLAR_MASS", "ELPETRO_MASS", "ELPETRO_MASS", "Stellar Mass"),
    ("BARYONIC_MASS", "BARYONIC_MASS", "BARYONIC_MASS", "Baryonic Mass"),
    ("GAS_MASS", None, None, "Gas Mass"),
    ("DM_MASS", None, None, "Dark Matter Mass"),
    ("BH_MASS", None, None, "Black Hole Mass"),
    
    # Structural properties
    ("HALFMASS_RAD", "ELPETRO_TH50_R", "ELPETRO_TH50_R", "Half-light Radius"),
    (None, "SERSIC_N", "SERSIC_N", "Sersic Index"),
    (None, "ELPETRO_BA", "ELPETRO_BA", "Axis Ratio"),
    (None, "ELPETRO_MTOL", "ELPETRO_MTOL", "Mass-to-Light Ratio"),
    
    # Kinematic properties
    ("VEL_DISP", None, None, "Velocity Dispersion"),
    ("VMAX", None, None, "Vmax"),
    
    # Chemical properties
    ("GAS_METALLICITY", None, None, "Gas Metallicity"),
    ("STAR_METALLICITY", "ELPETRO_METS", "ELPETRO_METS", "Stellar Metallicity"),
    
    # Photometric properties
    ("PHOTOMETRIC_R", "ELPETRO_ABSMAG_R", "ELPETRO_ABSMAG_R", "Absolute Magnitude"),
    ("COLOUR", "COLOR_U_R", "COLOR_U_R", "Colour"),
    
    # Star formation
    ("SFR", "ELPETRO_B300", "ELPETRO_B300", "Star Formation Rate"),
    
    # Other observational properties
    (None, "ZDIST", "ZDIST", "Redshift"),
    (None, "logMH", None, "HI Mass"),
    (None, "W50", None, "W50"),
]

# Filter to only properties that exist in at least one dataset
valid_properties = []
for prop_key_tng50, prop_key_alfalfa, prop_key_nsa, display_name in PROPERTIES:
    exists = False
    if prop_key_tng50 and prop_key_tng50 in tng50_data:
        exists = True
    if prop_key_alfalfa and prop_key_alfalfa in alfalfa_data:
        exists = True
    if prop_key_nsa and prop_key_nsa in nsa_data:
        exists = True
    
    if exists:
        valid_properties.append((prop_key_tng50, prop_key_alfalfa, prop_key_nsa, display_name))

# Create figure with subplots
n_props = len(valid_properties)
n_cols = 4
n_rows = (n_props + n_cols - 1) // n_cols

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5*n_rows))
if n_props == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Color scheme
colors = {
    "TNG50": "#1f77b4",
    "ALFALFA×NSA": "#ff7f0e",
    "NSA-only": "#2ca02c"
}

# Collect all datasets for legend
all_datasets_info = {}
for prop_key_tng50, prop_key_alfalfa, prop_key_nsa, display_name in valid_properties:
    if prop_key_tng50 and prop_key_tng50 in tng50_data:
        data = tng50_data[prop_key_tng50][np.isfinite(tng50_data[prop_key_tng50])]
        if "TNG50" not in all_datasets_info:
            all_datasets_info["TNG50"] = len(data)
    if prop_key_alfalfa and prop_key_alfalfa in alfalfa_data:
        data = alfalfa_data[prop_key_alfalfa][np.isfinite(alfalfa_data[prop_key_alfalfa])]
        if "ALFALFA×NSA" not in all_datasets_info:
            all_datasets_info["ALFALFA×NSA"] = len(data)
    if prop_key_nsa and prop_key_nsa in nsa_data:
        data = nsa_data[prop_key_nsa][np.isfinite(nsa_data[prop_key_nsa])]
        if "NSA-only" not in all_datasets_info:
            all_datasets_info["NSA-only"] = len(data)

# Plot each property
for idx, (prop_key_tng50, prop_key_alfalfa, prop_key_nsa, display_name) in enumerate(valid_properties):
    ax = axes[idx]
    
    # Collect data for this property
    datasets_to_plot = []
    
    if prop_key_tng50 and prop_key_tng50 in tng50_data:
        data = tng50_data[prop_key_tng50]
        data = data[np.isfinite(data)]
        if len(data) > 0:
            datasets_to_plot.append(("TNG50", data, colors["TNG50"]))
    
    if prop_key_alfalfa and prop_key_alfalfa in alfalfa_data:
        data = alfalfa_data[prop_key_alfalfa]
        data = data[np.isfinite(data)]
        if len(data) > 0:
            datasets_to_plot.append(("ALFALFA×NSA", data, colors["ALFALFA×NSA"]))
    
    if prop_key_nsa and prop_key_nsa in nsa_data:
        data = nsa_data[prop_key_nsa]
        data = data[np.isfinite(data)]
        if len(data) > 0:
            datasets_to_plot.append(("NSA-only", data, colors["NSA-only"]))
    
    # Plot histograms
    for dataset_name, data, color in datasets_to_plot:
        ax.hist(data, bins=40, alpha=0.7, label=f"{dataset_name} (N={len(data):,})",
                color=color, edgecolor="black", linewidth=0.7, density=True)
    
    # Set labels - only xlabel, no title (removes duplication)
    ax.set_xlabel(display_name, fontweight="bold", fontsize=10)
    ax.set_ylabel("Density", fontweight="bold", fontsize=10)
    
    # Set reasonable axis limits based on data percentiles
    all_data = []
    for _, data, _ in datasets_to_plot:
        all_data.extend(data)
    if len(all_data) > 0:
        p1, p99 = np.percentile(all_data, [0.5, 99.5])
        data_range = p99 - p1
        
        # Special handling for Dark Matter Mass - extend more towards positive side
        if display_name == "Dark Matter Mass":
            p1, p99 = np.percentile(all_data, [0.1, 99.9])
            data_range = p99 - p1
            ax.set_xlim(p1 - 0.05*data_range, p99 + 0.15*data_range)
        elif data_range > 0:
            ax.set_xlim(p1 - 0.05*data_range, p99 + 0.05*data_range)
    
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9, length=4, width=1.2)

# Add legend to actual bottom right subplot (last subplot in grid)
bottom_right_idx = n_rows * n_cols - 1
legend_ax = axes[bottom_right_idx]

# Create custom legend with all datasets
from matplotlib.patches import Rectangle
handles = []
labels = []
for dataset_name in ["TNG50", "ALFALFA×NSA", "NSA-only"]:
    if dataset_name in all_datasets_info:
        handles.append(Rectangle((0,0),1,1, facecolor=colors[dataset_name], alpha=0.7, 
                                edgecolor='black', linewidth=0.7))
        labels.append(f"{dataset_name} (N={all_datasets_info[dataset_name]:,})")

legend_ax.legend(handles, labels, loc="lower right", framealpha=0.95, edgecolor="black", 
                fancybox=True, fontsize=9, frameon=True, shadow=True, borderpad=0.8)

# Hide unused subplots
for idx in range(n_props, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")  # Publication quality 300 DPI
plt.close()

print(f"Combined distribution plot saved to: {OUTPUT_PATH}")
print(f"Plotted {n_props} properties across all datasets")
