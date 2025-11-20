"""
Generate 6 publication-style scatter plots for ALL ALFALFA × NSA causal edges.
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
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ScatterPlots", "alfalfa_causal_scatter.png")
DATA_PATH = os.path.join(REPO_ROOT, "Data", "alfalfa_nsa_final_13props.pkl")

# All 7 edges from FCIT output
EDGE_CONFIG = [
    ("BARYONIC_MASS", "ELPETRO_MASS", "Baryonic Mass o-o Stellar Mass", "undirected"),
    ("COLOR_U_R", "ELPETRO_B300", "Color o-o Star Formation", "undirected"),
    ("ELPETRO_MASS", "ELPETRO_ABSMAG_R", "Stellar Mass o-o Abs. Mag (r)", "undirected"),
    ("ELPETRO_MTOL", "COLOR_U_R", "Mass-to-Light o-o Color", "undirected"),
    ("logMH", "BARYONIC_MASS", "HI Mass o-o Baryonic Mass", "undirected"),
    ("logMH", "ELPETRO_MASS", "HI Mass o-o Stellar Mass", "undirected"),
    ("logMH", "ZDIST", "HI Mass o-o Redshift", "undirected"),
]

COLORSETS = [
    (["#03045e", "#023e8a", "#0077b6", "#0096c7", "#00b4d8", "#48cae4", "#90e0ef", "#caf0f8"], "Atlantic Blue"),
    (["#10002b", "#240046", "#3c096c", "#5a189a", "#7b2cbf", "#9d4edd", "#c77dff", "#e0aaff"], "Velvet Purple"),
    (["#14213d", "#1f2a44", "#324a5f", "#3a6ea5", "#4ea5d9", "#56cfe1", "#80ffdb", "#a9f8ff"], "Polar Aqua"),
    (["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd"], "Steel Gray"),
    (["#2d1b69", "#11998e", "#38ef7d"], "Emerald"),
    (["#667eea", "#764ba2"], "Royal"),
    (["#f093fb", "#f5576c"], "Rose"),
    (["#641220", "#6e1423", "#85182a", "#a11d33", "#bd1f36", "#da1e37", "#f1495b", "#ff8da1"], "Crimson Flame"),
    (["#0f110c", "#3f5e5a", "#6ba292", "#98cbb4", "#c5e8cd", "#f0fff3", "#ffd6ba", "#ffb3c1"], "Pastel Breeze"),
    (["#2d1b69", "#3d2a7a", "#4e3a8b", "#5f4a9c", "#705aad", "#816abe", "#927acf", "#a38ae0"], "Royal Indigo"),
]


def load_data(path: str) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def plot_edge(ax, x, y, title, edge_type, cmap):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    hexbin = ax.hexbin(
        x_clean,
        y_clean,
        gridsize=60,
        cmap=cmap,
        mincnt=1,
        linewidths=0.25,
        edgecolors="black",
        alpha=0.95,
    )

    cbar = plt.colorbar(hexbin, ax=ax, label="Count", pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Count", fontsize=9)

    slope, intercept, r_value, *_ = stats.linregress(x_clean, y_clean)
    x_line = np.array([x_clean.min(), x_clean.max()])
    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color="#ffffff",
        linewidth=2.5,
        linestyle="--" if edge_type != "undirected" else ":",
        label=f"R² = {r_value**2:.3f}",
        path_effects=[path_effects.Stroke(linewidth=3.5, foreground="black"), path_effects.Normal()],
    )

    legend = ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="black", fancybox=True)
    legend.get_frame().set_facecolor("#f8f9fa")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9, length=5, width=1.1)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax.text(
        0.02,
        0.98,
        f"N = {len(x_clean):,}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, linewidth=0.5),
    )


def main() -> None:
    data_dict = load_data(DATA_PATH)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sns.set_style("ticks")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10})

    # Create grid for 7 edges (3x3 grid, hide 2)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (x_var, y_var, title, edge_type) in enumerate(EDGE_CONFIG):
        ax = axes[idx]
        cmap = LinearSegmentedColormap.from_list(COLORSETS[idx][1], COLORSETS[idx][0], N=256)
        plot_edge(ax, data_dict[x_var], data_dict[y_var], title, edge_type, cmap)
        ax.set_xlabel(x_var.replace("_", " "), fontweight="bold", fontsize=10)
        ax.set_ylabel(y_var.replace("_", " "), fontweight="bold", fontsize=10)

    # Hide unused subplots (7 edges, so hide last 2)
    for idx in range(len(EDGE_CONFIG), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
