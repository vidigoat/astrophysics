"""
Generate publication-style scatter plots for TNG50 causal edges (Part 1).
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ScatterPlots", "tng50_causal_scatter.png")
DATA_PATH = os.path.join(REPO_ROOT, "Data", "tng50_final.pkl")

EDGE_CONFIG = [
    ("BARYONIC_MASS", "DM_MASS", "Baryonic Mass o-o Dark Matter Mass", "undirected"),
    ("BARYONIC_MASS", "VEL_DISP", "Baryonic Mass o-> Velocity Dispersion", "partial"),
    ("COLOUR", "PHOTOMETRIC_U", "Colour o-> Photometric U", "partial"),
    ("GAS_MASS", "PHOTOMETRIC_R", "Gas Mass o-o Photometric R", "undirected"),
    ("GAS_MASS", "STAR_METALLICITY", "Gas Mass o-o Stellar Metallicity", "undirected"),
    ("GAS_MASS", "VEL_DISP", "Gas Mass o-> Velocity Dispersion", "partial"),
    ("PHOTOMETRIC_R", "PHOTOMETRIC_U", "Photometric R o-> Photometric U", "partial"),
    ("PHOTOMETRIC_R", "SFR", "Photometric R → Star Formation Rate", "directed"),
    ("PHOTOMETRIC_U", "SFR", "Photometric U → Star Formation Rate", "directed"),
]

LABEL_MAP = {
    "DM_MASS": "Dark Matter Mass",
    "STELLAR_MASS": "Stellar Mass",
    "GAS_MASS": "Gas Mass",
    "BH_MASS": "Black Hole Mass",
    "BARYONIC_MASS": "Baryonic Mass",
    "HALFMASS_RAD": "Half-mass Radius",
    "VEL_DISP": "Velocity Dispersion",
    "VMAX": "Vmax",
    "GAS_METALLICITY": "Gas Metallicity",
    "STAR_METALLICITY": "Stellar Metallicity",
    "PHOTOMETRIC_U": "Photometric U",
    "PHOTOMETRIC_R": "Photometric R",
    "SFR": "Star Formation Rate",
    "COLOUR": "Colour",
}

AXIS_LIMITS = {
    "DM_MASS": (5.0, 13.0),
    "STELLAR_MASS": (6.0, 14.0),
    "GAS_MASS": (6.5, 11.5),
    "BH_MASS": (0.0, 12.0),
    "BARYONIC_MASS": (7.0, 13.0),
    "HALFMASS_RAD": (-1.5, 2.5),
    "VEL_DISP": (2.0, 300.0),
    "VMAX": (5.0, 500.0),
    "GAS_METALLICITY": (0.0, 0.15),
    "STAR_METALLICITY": (0.0005, 0.15),
    "PHOTOMETRIC_U": (-26.0, -9.0),
    "PHOTOMETRIC_R": (-27.0, -9.0),
    "SFR": (-3.0, 2.5),
    "COLOUR": (-2.0, 4.0),
}

COLORSETS = [
    (["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#f48c06", "#ffba08"], "Inferno Red"),
    (["#0b132b", "#1c2541", "#3a506b", "#5bc0be", "#6fffe9"], "Deep Ocean"),
    (["#2f323a", "#33658a", "#86bbd8", "#f6ae2d", "#f26419"], "Sunrise"),
    (["#231942", "#5e548e", "#9f86c0", "#be95c4", "#e0b1cb"], "Mauve"),
    (["#03045e", "#023e8a", "#0077b6", "#0096c7", "#00b4d8", "#48cae4"], "Azure"),
    (["#1b4332", "#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2"], "Forest"),
    (["#132a13", "#31572c", "#4f772d", "#90a955", "#b5c99a"], "Moss"),
    (["#590d22", "#800f2f", "#a4133c", "#c9184a", "#ff758f"], "Raspberry"),
    (["#10002b", "#240046", "#3c096c", "#5a189a", "#7b2cbf", "#9d4edd"], "Iris"),
]


def load_data(path: str) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def plot_edge(ax, x, y, title, edge_type, cmap, x_var, y_var):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) == 0:
        return

    max_points = 20_000
    if len(x_clean) > max_points:
        np.random.seed(42)
        sample_idx = np.random.choice(len(x_clean), size=max_points, replace=False)
        x_clean = x_clean[sample_idx]
        y_clean = y_clean[sample_idx]

    hexbin = ax.hexbin(
        x_clean,
        y_clean,
        gridsize=50,
        cmap=cmap,
        mincnt=1,
        linewidths=0.2,
        edgecolors="black",
        alpha=0.95,
    )

    cbar = plt.colorbar(hexbin, ax=ax, label="Count", pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Count", fontsize=9)

    slope, intercept, r_value, *_ = stats.linregress(x_clean, y_clean)
    x_p1, x_p99 = np.percentile(x_clean, [1, 99])
    x_line = np.array([x_p1, x_p99])
    y_line = slope * x_line + intercept

    style_lookup = {
        "directed": ("#ffffff", "-", 2.5),
        "partial": ("#ffe066", "--", 2.3),
        "undirected": ("#00f5d4", ":", 2.3),
    }
    line_color, line_style, line_width = style_lookup.get(edge_type, ("#ffffff", "--", 2.5))

    ax.plot(
        x_line,
        y_line,
        color=line_color,
        linewidth=line_width,
        linestyle=line_style,
        label=f"R² = {r_value**2:.3f}",
        path_effects=[path_effects.Stroke(linewidth=line_width + 1, foreground="black"), path_effects.Normal()],
    )

    legend = ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="black", fancybox=True)
    legend.get_frame().set_facecolor("#f8f9fa")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9, length=5, width=1.1)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    x_p1, x_p99 = np.percentile(x_clean, [1, 99])
    y_p1, y_p99 = np.percentile(y_clean, [1, 99])
    
    x_range = x_p99 - x_p1
    y_range = y_p99 - y_p1
    
    x_pad = x_range * 0.05
    y_pad = y_range * 0.05
    
    ax.set_xlim(x_p1 - x_pad, x_p99 + x_pad)
    ax.set_ylim(y_p1 - y_pad, y_p99 + y_pad)

    ax.set_xlabel(LABEL_MAP.get(x_var, x_var.replace("_", " ")), fontweight="bold", fontsize=9)
    ax.set_ylabel(LABEL_MAP.get(y_var, y_var.replace("_", " ")), fontweight="bold", fontsize=9)

    ax.text(
        0.02,
        0.98,
        f"N = {len(x_clean):,}",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0.4),
    )


def main() -> None:
    data_dict = load_data(DATA_PATH)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sns.set_style("ticks")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9})

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (x_var, y_var, title, edge_type) in enumerate(EDGE_CONFIG):
        ax = axes[idx]
        cmap = LinearSegmentedColormap.from_list(COLORSETS[idx][1], COLORSETS[idx][0], N=256)
        plot_edge(ax, data_dict[x_var], data_dict[y_var], title, edge_type, cmap, x_var, y_var)

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()

