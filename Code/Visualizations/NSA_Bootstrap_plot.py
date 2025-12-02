"""Bootstrap validation figure for NSA dataset - Horizontal bar layout."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(__file__))
from bootstrap_plot_utils import parse_fcit_results, get_edge_display_label

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "Results", "nsa_bootstrap_validation.csv")
FCIT_RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "nsa_fcit_t14_p50.txt")
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ValidationPlots", "nsa_bootstrap_validation.png")

# Get all edges from main FCIT run (normalized to match bootstrap format)
main_run_edges = parse_fcit_results(FCIT_RESULTS_PATH)

# Load bootstrap results
bootstrap_df = pd.read_csv(CSV_PATH)
bootstrap_dict = dict(zip(bootstrap_df["edge"], bootstrap_df["percentage"]))

# Create dataframe with all main run edges
plot_data = []
for edge in main_run_edges:
    percentage = bootstrap_dict.get(edge, 0.0)
    label = get_edge_display_label(edge)
    plot_data.append({
        "edge": edge,
        "label": label,
        "percentage": percentage
    })

df = pd.DataFrame(plot_data)
df = df.sort_values("percentage", ascending=True).reset_index(drop=True)

# Set up style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "font.family": "monospace",  # Use monospace for better alignment of edge symbols
})

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(11, max(6.5, len(df) * 0.4)))

y_pos = np.arange(len(df))
pct = df["percentage"].tolist()
labels = df["label"].tolist()

colors = [
    "#27ae60" if p >= 95 else "#2ecc71" if p >= 80 else "#f39c12" if p >= 50 else "#e74c3c"
    for p in pct
]

bars = ax.barh(y_pos, pct, color=colors, alpha=0.9, edgecolor="black", linewidth=1.2, height=0.75)

# Add percentage labels on bars
for i, (bar, value) in enumerate(zip(bars, pct)):
    ax.text(
        value + 1.5 if value > 5 else 5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        ha="left",
        va="center",
        fontweight="bold",
        fontsize=9,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8.5, family="monospace")
ax.set_xlim(0, 105)
ax.axvline(x=80, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="80% threshold")
ax.grid(True, alpha=0.3, axis="x", linestyle=":", linewidth=0.8)
ax.set_xlabel("Bootstrap Recovery Rate (%)", fontsize=11, weight="bold", labelpad=10)
ax.set_ylabel("Causal Edge", fontsize=11, weight="bold", labelpad=10)

legend_elements = [
    Patch(facecolor="#27ae60", edgecolor="black", linewidth=1.0, label="Very Strong (≥95%)"),
    Patch(facecolor="#2ecc71", edgecolor="black", linewidth=1.0, label="Strong (≥80%)"),
    Patch(facecolor="#f39c12", edgecolor="black", linewidth=1.0, label="Moderate (50-79%)"),
    Patch(facecolor="#e74c3c", edgecolor="black", linewidth=1.0, label="Weak (<50%)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.95, edgecolor="black")

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

