"""
Generate 9 publication-style scatter plots for best NSA-only causal edges (excluding ZDIST selection effects).
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
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ScatterPlots", "nsa_causal_scatter.png")
DATA_PATH = os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl")

# 9 best edges excluding ZDIST (selection effects) - focusing on most physically meaningful
EDGE_CONFIG = [
    ("COLOR_U_R", "ELPETRO_ABSMAG_R", "Color → Abs. Mag (r)", "directed"),
    ("ELPETRO_ABSMAG_R", "ELPETRO_MASS", "Abs. Mag (r) → Stellar Mass", "directed"),
    ("ELPETRO_MTOL", "ELPETRO_MASS", "Mass-to-Light → Stellar Mass", "partial"),
    ("ELPETRO_MTOL", "COLOR_U_R", "Mass-to-Light o-o Color", "undirected"),
    ("ELPETRO_METS", "ELPETRO_ABSMAG_R", "Metallicity → Abs. Mag (r)", "directed"),
    ("COLOR_U_R", "ELPETRO_B300", "Color o-o Star Formation", "undirected"),
    ("SERSIC_N", "ELPETRO_ABSMAG_R", "Sersic N → Abs. Mag (r)", "directed"),
    ("ELPETRO_BA", "ELPETRO_ABSMAG_R", "Axis Ratio → Abs. Mag (r)", "partial"),
    ("ELPETRO_TH50_R", "ELPETRO_ABSMAG_R", "Size → Abs. Mag (r)", "partial"),
]

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

    # Create grid for 9 edges (3 rows x 4 columns, hide 3)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, (x_var, y_var, title, edge_type) in enumerate(EDGE_CONFIG):
        ax = axes[idx]
        cmap = LinearSegmentedColormap.from_list(COLORSETS[idx][1], COLORSETS[idx][0], N=256)
        plot_edge(ax, data_dict[x_var], data_dict[y_var], title, edge_type, cmap)
        ax.set_xlabel(x_var.replace("_", " "), fontweight="bold", fontsize=9)
        ax.set_ylabel(y_var.replace("_", " "), fontweight="bold", fontsize=9)

    # Hide unused subplots (9 edges, so hide last 3)
    for idx in range(len(EDGE_CONFIG), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
