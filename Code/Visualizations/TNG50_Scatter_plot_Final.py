"""
Generate publication-quality scatter plots for TNG50 causal edges.
Creates 4 PNGs with 3x3 grids showing all 35 edges exactly once.
Matches the beautiful reference image style.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "tng50_final.pkl")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Plots", "ScatterPlots")

# All 35 edges organized: 9+9+9+8
EDGE_CONFIGS = [
    # Part 1: Edges 1-9
    [
        ("BARYONIC_MASS", "DM_MASS", "Baryonic Mass → Dark Matter Mass", "directed"),
        ("BARYONIC_MASS", "HALFMASS_RAD", "Baryonic Mass → Half-mass Radius", "directed"),
        ("BARYONIC_MASS", "VEL_DISP", "Baryonic Mass → Velocity Dispersion", "directed"),
        ("STELLAR_MASS", "BARYONIC_MASS", "Stellar Mass → Baryonic Mass", "directed"),
        ("STELLAR_MASS", "BH_MASS", "Stellar Mass → Black Hole Mass", "directed"),
        ("STELLAR_MASS", "HALFMASS_RAD", "Stellar Mass → Half-mass Radius", "directed"),
        ("STELLAR_MASS", "VEL_DISP", "Stellar Mass → Velocity Dispersion", "directed"),
        ("STELLAR_MASS", "VMAX", "Stellar Mass → Vmax", "directed"),
        ("STELLAR_MASS", "STAR_METALLICITY", "Stellar Mass → Stellar Metallicity", "directed"),
    ],
    # Part 2: Edges 10-18
    [
        ("GAS_MASS", "BH_MASS", "Gas Mass → Black Hole Mass", "directed"),
        ("GAS_MASS", "STAR_METALLICITY", "Gas Mass → Stellar Metallicity", "directed"),
        ("GAS_MASS", "VEL_DISP", "Gas Mass → Velocity Dispersion", "directed"),
        ("GAS_MASS", "VMAX", "Gas Mass → Vmax", "directed"),
        ("GAS_MASS", "HALFMASS_RAD", "Gas Mass → Half-mass Radius", "directed"),
        ("STAR_METALLICITY", "BARYONIC_MASS", "Stellar Metallicity → Baryonic Mass", "directed"),
        ("STAR_METALLICITY", "GAS_METALLICITY", "Stellar Metallicity → Gas Metallicity", "directed"),
        ("DM_MASS", "GAS_METALLICITY", "Dark Matter Mass → Gas Metallicity", "directed"),
        ("VEL_DISP", "HALFMASS_RAD", "Velocity Dispersion → Half-mass Radius", "directed"),
    ],
    # Part 3: Edges 19-27
    [
        ("COLOUR", "DM_MASS", "Colour → Dark Matter Mass", "directed"),
        ("COLOUR", "PHOTOMETRIC_U", "Colour o-o Photometric U", "undirected"),
        ("COLOUR", "STELLAR_MASS", "Colour o-o Stellar Mass", "undirected"),
        ("GAS_MASS", "COLOUR", "Gas Mass o-o Colour", "undirected"),
        ("GAS_MASS", "PHOTOMETRIC_R", "Gas Mass o-o Photometric R", "undirected"),
        ("GAS_MASS", "PHOTOMETRIC_U", "Gas Mass o-o Photometric U", "undirected"),
        ("GAS_MASS", "SFR", "Gas Mass o-o Star Formation Rate", "undirected"),
        ("PHOTOMETRIC_R", "COLOUR", "Photometric R o-o Colour", "undirected"),
        ("PHOTOMETRIC_R", "PHOTOMETRIC_U", "Photometric R o-o Photometric U", "undirected"),
    ],
    # Part 4: Edges 28-35
    [
        ("PHOTOMETRIC_R", "SFR", "Photometric R o-o Star Formation Rate", "undirected"),
        ("PHOTOMETRIC_R", "STELLAR_MASS", "Photometric R o-o Stellar Mass", "undirected"),
        ("PHOTOMETRIC_U", "SFR", "Photometric U o-o Star Formation Rate", "undirected"),
        ("VMAX", "BARYONIC_MASS", "Vmax → Baryonic Mass", "directed"),
        ("VMAX", "VEL_DISP", "Vmax → Velocity Dispersion", "directed"),
        ("BH_MASS", "STAR_METALLICITY", "Black Hole Mass o-o Stellar Metallicity", "undirected"),
        ("BH_MASS", "VMAX", "Black Hole Mass o-o Vmax", "undirected"),
        ("STAR_METALLICITY", "VMAX", "Stellar Metallicity o-o Vmax", "undirected"),
    ],
]

LABEL_MAP = {
    "DM_MASS": r"$\log_{10}(M_{\rm DM}/M_\odot)$",
    "STELLAR_MASS": r"$\log_{10}(M_*/M_\odot)$",
    "GAS_MASS": r"$\log_{10}(M_{\rm gas}/M_\odot)$",
    "BH_MASS": r"$\log_{10}(M_{\rm BH}/M_\odot)$",
    "BARYONIC_MASS": r"$\log_{10}(M_{\rm bary}/M_\odot)$",
    "HALFMASS_RAD": r"$\log_{10}(R_{1/2}/{\rm kpc})$",
    "VEL_DISP": r"$\sigma_v$ (km s$^{-1}$)",
    "VMAX": r"$V_{\rm max}$ (km s$^{-1}$)",
    "GAS_METALLICITY": r"$Z_{\rm gas}$",
    "STAR_METALLICITY": r"$Z_*$",
    "PHOTOMETRIC_U": r"$M_U$ (mag)",
    "PHOTOMETRIC_R": r"$M_R$ (mag)",
    "SFR": r"$\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})$",
    "COLOUR": r"$U-R$ (mag)",
}

# Sequential colormaps - we'll modify them to skip white/light colors
COLORMAPS_RAW = ['Blues', 'Purples', 'GnBu', 'Greys', 'Greens', 'BuPu', 'Reds', 'Oranges', 'YlGnBu']

def get_dark_colormap(cmap_name):
    """Get colormap starting from darker colors (skip white/light colors)"""
    cmap = plt.get_cmap(cmap_name)
    # Create new colormap that starts from 20% instead of 0% (skips white/light colors)
    from matplotlib.colors import ListedColormap
    colors = cmap(np.linspace(0.2, 1.0, 256))  # Start from 20% to skip white
    return ListedColormap(colors)

COLORMAPS = COLORMAPS_RAW  # Will be processed in plot_edge


def load_data(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def plot_edge(ax, x, y, title, edge_type, cmap_name, x_var, y_var):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = np.asarray(x)[mask]
    y_clean = np.asarray(y)[mask]
    
    if len(x_clean) == 0:
        ax.set_visible(False)
        return

    # Subsample for clarity
    max_points = 20000
    if len(x_clean) > max_points:
        np.random.seed(42)
        sample_idx = np.random.choice(len(x_clean), size=max_points, replace=False)
        x_clean = x_clean[sample_idx]
        y_clean = y_clean[sample_idx]

    # Calculate proper axis limits - use tighter percentiles for BH_MASS edges
    # Black hole mass often has many zeros/small values, so use tighter range
    if "BH_MASS" in (x_var, y_var):
        # For BH_MASS edges, use 2nd-98th percentiles to handle sparse/zero-heavy distributions
        x_p2, x_p98 = np.percentile(x_clean, [2, 98])
        y_p2, y_p98 = np.percentile(y_clean, [2, 98])
        x_range = x_p98 - x_p2 if (x_p98 - x_p2) != 0 else 1.0
        y_range = y_p98 - y_p2 if (y_p98 - y_p2) != 0 else 1.0
        x_pad = x_range * 0.05
        y_pad = y_range * 0.05
        ax.set_xlim(x_p2 - x_pad, x_p98 + x_pad)
        ax.set_ylim(y_p2 - y_pad, y_p98 + y_pad)
    else:
        # For other edges, use 5th-95th percentiles for optimal fit
        x_p5, x_p95 = np.percentile(x_clean, [5, 95])
        y_p5, y_p95 = np.percentile(y_clean, [5, 95])
        x_range = x_p95 - x_p5 if (x_p95 - x_p5) != 0 else 1.0
        y_range = y_p95 - y_p5 if (y_p95 - y_p5) != 0 else 1.0
        x_pad = x_range * 0.04  # Small, professional padding
        y_pad = y_range * 0.04
        ax.set_xlim(x_p5 - x_pad, x_p95 + x_pad)
        ax.set_ylim(y_p5 - y_pad, y_p95 + y_pad)
    
    # Truncate colormap to skip white/lightest colors - start from very dark shades
    # This ensures NO white/whitish shades in the color scale - start from very dark
    from matplotlib.colors import ListedColormap
    full_cmap = plt.get_cmap(cmap_name)
    # Skip the lightest 60% of colors (which are white/very light/whitish) - very dark start
    colors = full_cmap(np.linspace(0.6, 1.0, 256))  # Start from 60% for very dark colors
    dark_cmap = ListedColormap(colors)
    
    # Create hexbin with high-resolution hexagons and DARK colors (no white/whitish shades)
    hexbin = ax.hexbin(
        x_clean, y_clean,
        gridsize=150,  # High resolution hexagons for maximum detail
        cmap=dark_cmap,  # Use truncated colormap with no white
        mincnt=5,
        linewidths=0.0,
        edgecolors='none',
    )
    
    # Force the hexbin collection to be fully opaque
    hexbin.set_alpha(1.0)
    
    # Get counts and set color limits
    counts = hexbin.get_array()
    if counts is not None and len(counts) > 0:
        valid_counts = counts[counts > 0]
        if len(valid_counts) > 0:
            vmin = float(valid_counts.min())
            vmax = float(valid_counts.max())
            if vmax > vmin:
                hexbin.set_clim(vmin=vmin, vmax=vmax)
            else:
                hexbin.set_clim(vmin=max(1, vmin-1), vmax=vmax+1)

    # Colorbar
    cbar = plt.colorbar(hexbin, ax=ax, pad=0.02, fraction=0.045)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Count", fontsize=9, fontweight="bold")

    # Linear regression
    try:
        slope, intercept, r_value, _, _ = stats.linregress(x_clean, y_clean)
    except:
        slope, intercept, r_value = 0.0, np.nan, 0.0

    x_line = np.linspace(np.percentile(x_clean, 5), np.percentile(x_clean, 95), 2)
    y_line = slope * x_line + intercept

    # Dashed white line with black stroke (matching reference)
    style_lookup = {
        "directed": ("#ffffff", "--", 2.8),
        "partial": ("#ffd700", "--", 2.2),
        "undirected": ("#00ffff", "--", 2.2),
    }
    line_color, line_style, line_width = style_lookup.get(edge_type, ("#ffffff", "--", 2.2))

    ax.plot(
        x_line, y_line,
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
        zorder=10,
        path_effects=[path_effects.Stroke(linewidth=line_width+1.6, foreground="black", alpha=0.7), path_effects.Normal()]
    )

    # Labels and styling
    # Title removed as requested
    ax.set_xlabel(LABEL_MAP.get(x_var, x_var), fontsize=10, fontweight="bold")
    ax.set_ylabel(LABEL_MAP.get(y_var, y_var), fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=9, direction="in", top=True, right=True)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, color="#666666")

    # N annotation
    ax.text(0.02, 0.98, f"N = {len(x_clean):,}", transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, linewidth=0.8, edgecolor="#333333"),
            fontweight="bold")

    # Legend with R^2
    ax.legend([f"R² = {r_value**2:.3f}"], loc="best", fontsize=9, framealpha=0.95,
              edgecolor="black", fancybox=True, shadow=True)
    ax.legend_.get_frame().set_facecolor("#ffffff")


def main():
    data_dict = load_data(DATA_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,  # Publication quality 300 DPI
        "figure.dpi": 300,  # High resolution display
        "savefig.bbox": "tight",
        "font.size": 11,  # Larger fonts for readability
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    for part_idx, edge_config in enumerate(EDGE_CONFIGS, start=1):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # 12×10 inches = 3,600×3,000 pixels at 300 DPI
        axes = axes.flatten()
        
        for idx, (x_var, y_var, title, edge_type) in enumerate(edge_config):
            ax = axes[idx]
            cmap_name = COLORMAPS[idx % len(COLORMAPS)]
            plot_edge(ax, data_dict[x_var], data_dict[y_var], title, edge_type, cmap_name, x_var, y_var)

        # Hide unused axes
        for i in range(len(edge_config), 9):
            axes[i].axis("off")

        plt.tight_layout(h_pad=2.2, w_pad=2.2)
        outname = os.path.join(OUTPUT_DIR, f"tng50_causal_scatter_part{part_idx}.png")
        fig.savefig(outname, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")  # Publication quality 300 DPI
        plt.close(fig)
        print(f"Created: {outname} ({len(edge_config)} edges)")


if __name__ == "__main__":
    main()

