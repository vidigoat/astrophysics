"""
Create a combined null model validation visualization for both datasets.
Shows real vs. shuffled edge counts to validate causal discoveries.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ALFALFA_NULL_PATH = os.path.join(REPO_ROOT, "Results", "null_model_test.pkl")
NSA_NULL_PATH = os.path.join(REPO_ROOT, "Results", "nsa_null_model_test.pkl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ValidationPlots", "null_model_validation.png")

def load_null_results(path):
    """Load null model test results from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def create_combined_plot():
    """Create a combined visualization comparing real vs null model edges."""
    # Load results
    alfalfa_results = load_null_results(ALFALFA_NULL_PATH)
    nsa_results = load_null_results(NSA_NULL_PATH)
    
    # Extract data
    alfalfa_real = alfalfa_results["real_edges"]
    alfalfa_null_mean = alfalfa_results["null_mean"]
    alfalfa_null_std = alfalfa_results["null_std"]
    alfalfa_null_runs = alfalfa_results.get("null_counts", [])
    
    nsa_real = nsa_results["real_edges"]
    nsa_null_mean = nsa_results["null_mean"]
    nsa_null_std = nsa_results["null_std"]
    nsa_null_runs = nsa_results.get("null_counts", [])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    
    # ALFALFA × NSA plot
    categories = ["Real Data", "Null Model\n(Shuffled)"]
    real_edges = [alfalfa_real, 0]
    null_edges = [0, alfalfa_null_mean]
    errors = [0, alfalfa_null_std]
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax1.bar(x_pos[0], real_edges[0], width, 
                    color="#2E86AB", alpha=0.8, label="Real Data", 
                    edgecolor="black", linewidth=1.5)
    bars2 = ax1.bar(x_pos[1], null_edges[1], width,
                    color="#A23B72", alpha=0.8, label="Null Model",
                    edgecolor="black", linewidth=1.5,
                    yerr=errors[1] if errors[1] > 0 else None,
                    capsize=8, error_kw={"elinewidth": 2, "capthick": 2})
    
    # Add individual null run points
    if len(alfalfa_null_runs) > 0:
        for i, null_count in enumerate(alfalfa_null_runs):
            ax1.scatter(x_pos[1] + np.random.uniform(-0.15, 0.15), 
                       null_count, color="#6B2C5E", alpha=0.6, 
                       s=50, zorder=5, edgecolors="black", linewidth=0.5)
    
    ax1.set_ylabel("Number of Edges", fontweight="bold", fontsize=13)
    ax1.set_title("ALFALFA × NSA\nNull Model Validation", 
                  fontweight="bold", fontsize=14, pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, fontweight="bold")
    ax1.set_ylim(0, max(alfalfa_real, alfalfa_null_mean + 2*alfalfa_null_std) * 1.2)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="black", linewidth=0.8)
    
    # Add text annotation
    diff = alfalfa_real - alfalfa_null_mean
    ax1.text(0.5, 0.95, f"Difference: {diff:.1f} edges",
             transform=ax1.transAxes, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                      edgecolor="black", alpha=0.8),
             fontsize=11, fontweight="bold")
    
    # NSA-only plot
    nsa_real_edges = [nsa_real, 0]
    nsa_null_edges = [0, nsa_null_mean]
    nsa_errors = [0, nsa_null_std]
    
    bars3 = ax2.bar(x_pos[0], nsa_real_edges[0], width,
                    color="#2E86AB", alpha=0.8, label="Real Data",
                    edgecolor="black", linewidth=1.5)
    bars4 = ax2.bar(x_pos[1], nsa_null_edges[1], width,
                    color="#A23B72", alpha=0.8, label="Null Model",
                    edgecolor="black", linewidth=1.5,
                    yerr=nsa_errors[1] if nsa_errors[1] > 0 else None,
                    capsize=8, error_kw={"elinewidth": 2, "capthick": 2})
    
    # Add individual null run points
    if len(nsa_null_runs) > 0:
        for i, null_count in enumerate(nsa_null_runs):
            ax2.scatter(x_pos[1] + np.random.uniform(-0.15, 0.15),
                       null_count, color="#6B2C5E", alpha=0.6,
                       s=50, zorder=5, edgecolors="black", linewidth=0.5)
    
    ax2.set_ylabel("Number of Edges", fontweight="bold", fontsize=13)
    ax2.set_title("NSA-only\nNull Model Validation",
                  fontweight="bold", fontsize=14, pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontweight="bold")
    ax2.set_ylim(0, max(nsa_real, nsa_null_mean + 2*nsa_null_std) * 1.2)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    # Set y-axis ticks to whole numbers for NSA plot
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add text annotation
    nsa_diff = nsa_real - nsa_null_mean
    ax2.text(0.5, 0.95, f"Difference: {nsa_diff:.1f} edges",
             transform=ax2.transAxes, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="black", alpha=0.8),
             fontsize=11, fontweight="bold")
    
    # Add overall title
    fig.suptitle("Null Model Validation: Real vs. Shuffled Data",
                 fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")

if __name__ == "__main__":
    create_combined_plot()

