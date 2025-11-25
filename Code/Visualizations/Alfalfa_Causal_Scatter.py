"""
Generate publication-style scatter plots for ALFALFA × NSA causal edges.
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

EDGE_CONFIG = [
    ("BARYONIC_MASS", "ELPETRO_MASS", "Baryonic Mass o-o Stellar Mass", "undirected"),
    ("COLOR_U_R", "ELPETRO_B300", "Color o-o Star Formation", "undirected"),
    ("ELPETRO_MASS", "ELPETRO_ABSMAG_R", "Stellar Mass o-o Abs. Mag (r)", "undirected"),
    ("ELPETRO_MTOL", "COLOR_U_R", "Mass-to-Light o-o Color", "undirected"),
    ("logMH", "BARYONIC_MASS", "HI Mass o-o Baryonic Mass", "undirected"),
    ("logMH", "ELPETRO_MASS", "HI Mass o-o Stellar Mass", "undirected"),
    ("logMH", "ZDIST", "HI Mass o-o Redshift", "undirected"),
]

LABEL_MAP = {
    "BARYONIC_MASS": r"$\log(M_{\rm baryon}/M_{\odot})$",
    "ELPETRO_MASS": r"$\log(M_*/M_{\odot})$",
    "COLOR_U_R": r"$(u-r)$",
    "ELPETRO_B300": r"$B_{300}$",
    "ELPETRO_ABSMAG_R": r"$M_r$ (mag)",
    "ELPETRO_MTOL": r"$M/L$",
    "logMH": r"$\log(M_{\rm HI}/M_{\odot})$",
    "ZDIST": r"$z$",
    "ELPETRO_METS": r"$\log(Z)$",
    "ELPETRO_BA": r"$b/a$",
    "ELPETRO_TH50_R": r"$r_{50}$ (arcsec)",
    "SERSIC_N": r"$n$",
    "W50": r"$W_{50}$ (km s$^{-1}$)",
}

AXIS_LIMITS = {
    "BARYONIC_MASS": (6.0, 12.0),
    "ELPETRO_MASS": (6.0, 12.0),
    "COLOR_U_R": (-0.5, 4.0),
    "ELPETRO_B300": (0.0, 10.0),
    "ELPETRO_ABSMAG_R": (-25.0, -10.0),
    "ELPETRO_MTOL": (0.1, 10.0),
    "logMH": (6.0, 10.5),
    "ZDIST": (0.0, 0.15),
    "ELPETRO_METS": (-2.5, 0.5),
    "ELPETRO_BA": (0.0, 1.0),
    "ELPETRO_TH50_R": (0.0, 25.0),
    "SERSIC_N": (0.0, 6.0),
    "W50": (20.0, 500.0),
}

COLORSETS = [
    (["#03045e", "#023e8a", "#0077b6", "#0096c7", "#00b4d8", "#48cae4", "#90e0ef", "#caf0f8"], "Atlantic Blue"),
    (["#10002b", "#240046", "#3c096c", "#5a189a", "#7b2cbf", "#9d4edd", "#c77dff", "#e0aaff"], "Velvet Purple"),
    (["#14213d", "#1f2a44", "#324a5f", "#3a6ea5", "#4ea5d9", "#56cfe1", "#80ffdb", "#a9f8ff"], "Polar Aqua"),
    (["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd"], "Steel Gray"),
    (["#2d1b69", "#11998e", "#38ef7d"], "Emerald"),
    (["#667eea", "#764ba2"], "Royal"),
    (["#f093fb", "#f5576c"], "Rose"),
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
        linestyle=":" if edge_type == "undirected" else "--",
        label=f"R² = {r_value**2:.3f}",
        path_effects=[path_effects.Stroke(linewidth=3.5, foreground="black"), path_effects.Normal()],
    )

    legend = ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="black", fancybox=True)
    legend.get_frame().set_facecolor("#f8f9fa")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9, length=5, width=1.1)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    
    if x_var in AXIS_LIMITS:
        ax.set_xlim(AXIS_LIMITS[x_var])
    else:
        x_pad = (x_clean.max() - x_clean.min()) * 0.05
        ax.set_xlim(x_clean.min() - x_pad, x_clean.max() + x_pad)
    
    if y_var in AXIS_LIMITS:
        ax.set_ylim(AXIS_LIMITS[y_var])
    else:
        y_pad = (y_clean.max() - y_clean.min()) * 0.05
        ax.set_ylim(y_clean.min() - y_pad, y_clean.max() + y_pad)

    ax.set_xlabel(LABEL_MAP.get(x_var, x_var.replace("_", " ")), fontweight="bold", fontsize=10)
    ax.set_ylabel(LABEL_MAP.get(y_var, y_var.replace("_", " ")), fontweight="bold", fontsize=10)

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

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (x_var, y_var, title, edge_type) in enumerate(EDGE_CONFIG):
        ax = axes[idx]
        cmap = LinearSegmentedColormap.from_list(COLORSETS[idx][1], COLORSETS[idx][0], N=256)
        plot_edge(ax, data_dict[x_var], data_dict[y_var], title, edge_type, cmap, x_var, y_var)

    for idx in range(len(EDGE_CONFIG), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
