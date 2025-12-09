"""
Generate publication-quality scatter plots for ALFALFA×NSA causal edges.
Creates 2 PNGs with 3x3 grids showing all 13 edges exactly once.
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
DATA_PATH = os.path.join(REPO_ROOT, "Data", "alfalfa_nsa_final_13props.pkl")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Plots", "ScatterPlots")

# All 13 edges organized: 7+6
EDGE_CONFIGS = [
    # Part 1: Edges 1-7
    [
        ("BARYONIC_MASS", "ELPETRO_ABSMAG_R", "Baryonic Mass → Abs. Mag (R)", "directed"),
        ("COLOR_U_R", "ELPETRO_B300", "Colour o-> Star Formation", "partial"),
        ("ELPETRO_MASS", "BARYONIC_MASS", "Stellar Mass → Baryonic Mass", "directed"),
        ("ELPETRO_MASS", "ELPETRO_ABSMAG_R", "Stellar Mass → Abs. Mag (R)", "directed"),
        ("ELPETRO_METS", "ELPETRO_B300", "Metallicity o-> Star Formation", "partial"),
        ("ELPETRO_MTOL", "COLOR_U_R", "Mass-to-Light o-o Colour", "undirected"),
        ("ELPETRO_MTOL", "ELPETRO_ABSMAG_R", "Mass-to-Light → Abs. Mag (R)", "directed"),
    ],
    # Part 2: Edges 8-13
    [
        ("ELPETRO_MTOL", "ELPETRO_MASS", "Mass-to-Light o-> Stellar Mass", "partial"),
        ("ELPETRO_TH50_R", "ZDIST", "Half-light Radius o-> Redshift", "partial"),
        ("W50", "logMH", "W50 o-o HI Mass", "undirected"),
        ("logMH", "BARYONIC_MASS", "HI Mass → Baryonic Mass", "directed"),
        ("logMH", "ELPETRO_MASS", "HI Mass o-> Stellar Mass", "partial"),
        ("logMH", "ZDIST", "HI Mass o-> Redshift", "partial"),
    ],
]

LABEL_MAP = {
    "BARYONIC_MASS": r"$\log_{10}(M_{\rm bary}/M_\odot)$",
    "ELPETRO_MASS": r"$\log_{10}(M_*/M_\odot)$",
    "COLOR_U_R": r"$U-R$ (mag)",
    "ELPETRO_B300": r"${\rm SFR}$ ($M_\odot$ yr$^{-1}$)",
    "ELPETRO_ABSMAG_R": r"$M_R$ (mag)",
    "ELPETRO_MTOL": r"$M/L_R$ ($M_\odot/L_\odot$)",
    "logMH": r"$\log_{10}(M_{\rm HI}/M_\odot)$",
    "ZDIST": r"$z$",
    "ELPETRO_METS": r"$\log_{10}(Z/Z_\odot)$",
    "ELPETRO_BA": r"$b/a$",
    "ELPETRO_TH50_R": r"$R_{1/2}$ (arcsec)",
    "SERSIC_N": r"$n$",
    "W50": r"$W_{50}$ (km s$^{-1}$)",
}

# Sequential colormaps - will be modified to skip white/light colors
COLORMAPS = ['Blues', 'Purples', 'GnBu', 'Greys', 'Greens', 'BuPu', 'Reds', 'Oranges', 'YlGnBu']


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

    # Calculate proper axis limits - use adaptive percentiles for consistent hexagon sizes
    # For better consistency, use IQR-based approach with tighter range
    x_p10, x_p90 = np.percentile(x_clean, [10, 90])
    y_p10, y_p90 = np.percentile(y_clean, [10, 90])
    x_iqr = x_p90 - x_p10
    y_iqr = y_p90 - y_p10
    
    # Use tighter range: 1.5 * IQR beyond quartiles, but cap at 2nd-98th percentiles
    x_p2, x_p98 = np.percentile(x_clean, [2, 98])
    y_p2, y_p98 = np.percentile(y_clean, [2, 98])
    
    # Set limits: use IQR-based range but don't exceed 2nd-98th percentiles
    x_low = max(x_p10 - 1.5 * x_iqr, x_p2)
    x_high = min(x_p90 + 1.5 * x_iqr, x_p98)
    y_low = max(y_p10 - 1.5 * y_iqr, y_p2)
    y_high = min(y_p90 + 1.5 * y_iqr, y_p98)
    
    # Add small padding
    x_range = x_high - x_low if (x_high - x_low) > 0 else 1.0
    y_range = y_high - y_low if (y_high - y_low) > 0 else 1.0
    x_pad = x_range * 0.03
    y_pad = y_range * 0.03
    
    ax.set_xlim(x_low - x_pad, x_high + x_pad)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    
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
        outname = os.path.join(OUTPUT_DIR, f"alfalfa_causal_scatter_part{part_idx}.png")
        fig.savefig(outname, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")  # Publication quality 300 DPI
        plt.close(fig)
        print(f"Created: {outname} ({len(edge_config)} edges)")


if __name__ == "__main__":
    main()

