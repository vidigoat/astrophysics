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
TNG50_NULL_PATH = os.path.join(REPO_ROOT, "Results", "tng50_null_model_test.pkl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "Plots", "ValidationPlots", "null_model_validation.png")

def load_null_results(path):
    """Load null model test results from pickle file."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def create_combined_plot():
    """Create a combined visualization comparing real vs null model edges."""
    # Load results
    alfalfa_results = load_null_results(ALFALFA_NULL_PATH)
    nsa_results = load_null_results(NSA_NULL_PATH)
    tng50_results = load_null_results(TNG50_NULL_PATH)
    
    # Determine which datasets are available
    datasets = []
    if alfalfa_results:
        datasets.append(("ALFALFA × NSA", alfalfa_results, 0))
    if nsa_results:
        datasets.append(("NSA-only", nsa_results, 1))
    if tng50_results:
        datasets.append(("TNG50", tng50_results, 2))
    
    if len(datasets) == 0:
        raise FileNotFoundError("No null model results found. Please run the null model scripts first.")
    
    # Create figure with appropriate number of subplots
    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    # Process each dataset
    for idx, (dataset_name, results, plot_idx) in enumerate(datasets):
        ax = axes[idx]
        
        real = results["real_edges"]
        null_mean = results["null_mean"]
        null_std = results["null_std"]
        null_runs = results.get("null_counts", [])
    
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
        
        categories = ["Real Data", "Null Model\n(Shuffled)"]
        x_pos = np.arange(len(categories))
        width = 0.6
        
        bars1 = ax.bar(x_pos[0], real, width, 
                        color="#2E86AB", alpha=0.8, label="Real Data", 
                        edgecolor="black", linewidth=1.5)
        bars2 = ax.bar(x_pos[1], null_mean, width,
                        color="#A23B72", alpha=0.8, label="Null Model",
                        edgecolor="black", linewidth=1.5,
                        yerr=null_std if null_std > 0 else None,
                        capsize=8, error_kw={"elinewidth": 2, "capthick": 2})
        
        # Add individual null run points
        if len(null_runs) > 0:
            for i, null_count in enumerate(null_runs):
                ax.scatter(x_pos[1] + np.random.uniform(-0.15, 0.15), 
                           null_count, color="#6B2C5E", alpha=0.6, 
                           s=50, zorder=5, edgecolors="black", linewidth=0.5)
        
        ax.set_ylabel("Number of Edges", fontweight="bold", fontsize=13)
        ax.set_title(f"{dataset_name}\nNull Model Validation", 
                      fontweight="bold", fontsize=14, pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontweight="bold")
        ax.set_ylim(0, max(real, null_mean + 2*null_std) * 1.2)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add text annotation
        diff = real - null_mean
        ax.text(0.5, 0.95, f"Difference: {diff:.1f} edges",
                 transform=ax.transAxes, ha="center", va="top",
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

