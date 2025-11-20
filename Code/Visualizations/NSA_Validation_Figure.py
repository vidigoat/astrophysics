"""Bootstrap validation figure for NSA-only dataset - Vertical layout."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "Results", "nsa_bootstrap_validation.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "BootstrapPlots", "nsa_bootstrap_validation.png")

EDGE_LABELS = {
    "COLOR_U_R --> ELPETRO_ABSMAG_R": "Color →\nAbs. Mag (r)",
    "COLOR_U_R o-o ELPETRO_B300": "Color ↔\nStar Formation",
    "ELPETRO_ABSMAG_R --> ELPETRO_MASS": "Abs. Mag (r) →\nStellar Mass",
    "ELPETRO_BA o-> ELPETRO_ABSMAG_R": "Axis Ratio →\nAbs. Mag (r)",
    "ELPETRO_BA o-> ELPETRO_TH50_R": "Axis Ratio →\nSize",
    "COLOR_U_R o-o ELPETRO_METS": "Color ↔\nMetallicity",
    "ELPETRO_METS --> ELPETRO_ABSMAG_R": "Metallicity →\nAbs. Mag (r)",
    "ELPETRO_B300 o-o ELPETRO_METS": "Star Formation ↔\nMetallicity",
    "COLOR_U_R o-o ELPETRO_MTOL": "Color ↔\nMass-to-Light",
    "ELPETRO_B300 o-o ELPETRO_MTOL": "Star Formation ↔\nMass-to-Light",
    "ELPETRO_MTOL --> ELPETRO_MASS": "Mass-to-Light →\nStellar Mass",
    "ELPETRO_METS o-o ELPETRO_MTOL": "Metallicity ↔\nMass-to-Light",
    "ELPETRO_TH50_R o-> ELPETRO_ABSMAG_R": "Size →\nAbs. Mag (r)",
    "SERSIC_N --> ELPETRO_ABSMAG_R": "Sersic N →\nAbs. Mag (r)",
    "ELPETRO_MTOL o-o SERSIC_N": "Mass-to-Light ↔\nSersic N",
    "SERSIC_N o-> ELPETRO_TH50_R": "Sersic N →\nSize",
    "ZDIST o-> ELPETRO_ABSMAG_R": "Redshift →\nAbs. Mag (r)",
    "ZDIST o-> ELPETRO_TH50_R": "Redshift →\nSize",
}

df = pd.read_csv(CSV_PATH)
df["label"] = df["edge"].map(EDGE_LABELS).fillna(df["edge"])
df = df.sort_values("percentage", ascending=False).reset_index(drop=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
})

fig, ax = plt.subplots(figsize=(14, 8))

labels = df["label"].tolist()
pct = df["percentage"].tolist()

colors = ["#27ae60" for _ in pct]  # All 100%

x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, pct, color=colors, alpha=0.9, edgecolor="black", linewidth=1.5, width=0.7)

for bar, value in zip(bars, pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.5,
        f"{value:.0f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9.5,
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylim(0, 105)
ax.axhline(y=80, color="gray", linestyle="--", linewidth=1.8, alpha=0.7, label="80% threshold")
ax.grid(True, alpha=0.3, axis="y", linestyle=":", linewidth=0.8)
ax.set_ylabel("Bootstrap Recovery Rate (%)", fontsize=12, weight="bold", labelpad=12)
ax.set_xlabel("Causal Edge", fontsize=12, weight="bold", labelpad=12)
ax.set_title(
    "NSA-only: Bootstrap Validation\n(10 runs, 80% subsample, N = 484,551 galaxies)",
    fontsize=13,
    weight="bold",
    pad=20,
)

legend_elements = [
    Patch(facecolor="#27ae60", edgecolor="black", linewidth=1.2, label="Very Strong (100%)"),
]
ax.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.4), fontsize=10, framealpha=0.95, edgecolor="black", fancybox=True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
