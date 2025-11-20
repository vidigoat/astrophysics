"""
Generate 9 additional publication-style scatter plots for remaining NSA-only causal edges.
This is Part 2 of the NSA scatter plots (edges 10-18).
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
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ScatterPlots", "nsa_causal_scatter_part2.png")
DATA_PATH = os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl")

# Remaining 9 edges (edges 10-18 from the 18 total edges)
EDGE_CONFIG = [
    ("ELPETRO_BA", "ELPETRO_TH50_R", "Axis Ratio → Size", "partial"),
    ("ELPETRO_METS", "COLOR_U_R", "Metallicity o-o Color", "undirected"),
    ("ELPETRO_B300", "ELPETRO_METS", "Star Formation o-o Metallicity", "undirected"),
    ("ELPETRO_B300", "ELPETRO_MTOL", "Star Formation o-o Mass-to-Light", "undirected"),
    ("ELPETRO_METS", "ELPETRO_MTOL", "Metallicity o-o Mass-to-Light", "undirected"),
    ("ELPETRO_MTOL", "SERSIC_N", "Mass-to-Light o-o Sersic N", "undirected"),
    ("SERSIC_N", "ELPETRO_TH50_R", "Sersic N → Size", "partial"),
    ("ZDIST", "ELPETRO_ABSMAG_R", "Redshift → Abs. Mag (r)", "partial"),
    ("ZDIST", "ELPETRO_TH50_R", "Redshift → Size", "partial"),
]

COLORSETS = [
    (["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"], "Midnight"),
    (["#2d3436", "#636e72", "#74b9ff", "#0984e3", "#00b894"], "Turquoise"),
    (["#3d1e6d", "#4a2c7a", "#5d3a87", "#6e4a94", "#7f5aa1"], "Royal Purple"),
    (["#0c0c0c", "#1a1a1a", "#2d2d2d", "#404040", "#5a5a5a"], "Charcoal"),
    (["#1e3a5f", "#2a4a6f", "#365a7f", "#426a8f", "#4e7a9f"], "Navy"),
    (["#4a148c", "#6a1b9a", "#8e24aa", "#ab47bc", "#ce93d8"], "Violet"),
    (["#1b5e20", "#2e7d32", "#388e3c", "#43a047", "#66bb6a"], "Emerald"),
    (["#b71c1c", "#c62828", "#d32f2f", "#e53935", "#ef5350"], "Crimson"),
    (["#e65100", "#ef6c00", "#f57c00", "#fb8c00", "#ff9800"], "Amber"),
]


def load_data(path: str) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def plot_edge(ax, x, y, title, edge_type, cmap):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Downsample to 40k particles for rendering efficiency
    max_points = 40_000
    if len(x_clean) > max_points:
        np.random.seed(42)  # For reproducibility
        sample_idx = np.random.choice(len(x_clean), size=max_points, replace=False)
        x_clean = x_clean[sample_idx]
        y_clean = y_clean[sample_idx]

    hexbin = ax.hexbin(
        x_clean,
        y_clean,
        gridsize=70,
        cmap=cmap,
        mincnt=1,
        linewidths=0.2,
        edgecolors="black",
        alpha=0.92,
    )

    cbar = plt.colorbar(hexbin, ax=ax, label="Count", pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Count", fontsize=8)

    slope, intercept, r_value, *_ = stats.linregress(x_clean, y_clean)
    x_line = np.array([x_clean.min(), x_clean.max()])
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

    legend = ax.legend(loc="best", fontsize=8, framealpha=0.9, edgecolor="black", fancybox=True)
    legend.get_frame().set_facecolor("#f8f9fa")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8, length=4, width=1.0)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.45)
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
        plot_edge(ax, data_dict[x_var], data_dict[y_var], title, edge_type, cmap)
        ax.set_xlabel(x_var.replace("_", " "), fontweight="bold", fontsize=9)
        ax.set_ylabel(y_var.replace("_", " "), fontweight="bold", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()

