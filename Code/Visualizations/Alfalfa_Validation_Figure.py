"""Bootstrap validation figure for ALFALFA × NSA dataset - Vertical layout."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "Results", "bootstrap_validation.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "BootstrapPlots", "alfalfa_bootstrap_validation.png")

EDGE_LABELS = {
    "ELPETRO_ABSMAG_R o-o ELPETRO_MASS": "Abs. Mag (r) ↔\nStellar Mass",
    "COLOR_U_R o-o ELPETRO_MTOL": "Color ↔\nMass-to-Light",
    "BARYONIC_MASS o-o ELPETRO_MASS": "Baryonic Mass ↔\nStellar Mass",
    "ELPETRO_MASS o-o logMH": "Stellar Mass ↔\nHI Mass",
    "BARYONIC_MASS o-o logMH": "Baryonic Mass ↔\nHI Mass",
    "ZDIST o-o logMH": "Redshift ↔\nHI Mass",
    "COLOR_U_R o-o ELPETRO_B300": "Color ↔\nStar Formation",
}

df = pd.read_csv(CSV_PATH)
df["label"] = df["edge"].map(EDGE_LABELS).fillna(df["edge"])

# Only show edges that appear in bootstrap results (≥80% threshold)
# Edges below threshold are not shown to avoid confusion

df = df.sort_values("percentage", ascending=False).reset_index(drop=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(8, 7))

labels = df["label"].tolist()
pct = df["percentage"].tolist()

colors = [
    "#27ae60" if p >= 95 else "#2ecc71" if p >= 80 else "#f39c12" if p >= 50 else "#e74c3c"
    for p in pct
]

x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, pct, color=colors, alpha=0.9, edgecolor="black", linewidth=1.5, width=0.65)

for bar, value in zip(bars, pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, wrap=True)
ax.set_ylim(0, 105)
ax.axhline(y=80, color="gray", linestyle="--", linewidth=1.8, alpha=0.7, label="80% threshold")
ax.grid(True, alpha=0.3, axis="y", linestyle=":", linewidth=0.8)
ax.set_ylabel("Bootstrap Recovery Rate (%)", fontsize=13, weight="bold", labelpad=12)
ax.set_xlabel("Causal Edge", fontsize=13, weight="bold", labelpad=12)
ax.set_title(
    "ALFALFA × NSA: Bootstrap Validation\n(50 runs, 80% subsample, N = 20,569 galaxies)",
    fontsize=14,
    weight="bold",
    pad=20,
)

legend_elements = [
    Patch(facecolor="#27ae60", edgecolor="black", linewidth=1.2, label="Very Strong (≥95%)"),
    Patch(facecolor="#2ecc71", edgecolor="black", linewidth=1.2, label="Strong (≥80%)"),
    Patch(facecolor="#f39c12", edgecolor="black", linewidth=1.2, label="Moderate (50-79%)"),
    Patch(facecolor="#e74c3c", edgecolor="black", linewidth=1.2, label="Weak (<50%)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.95, edgecolor="black", fancybox=True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved: {OUTPUT_PATH}")
