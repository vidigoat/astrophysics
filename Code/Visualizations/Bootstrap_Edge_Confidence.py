"""
Create comparative bootstrap edge confidence plots for both datasets.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ALFALFA_CSV = os.path.join(REPO_ROOT, "Results", "bootstrap_validation.csv")
NSA_CSV = os.path.join(REPO_ROOT, "Results", "nsa_bootstrap_validation.csv")
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "Plots", "BootstrapPlots", "edge_confidence_comparison.png"
)


def load_bootstrap_results(csv_path: str) -> pd.DataFrame:
    """Load bootstrap results, sort by recovery percentage."""
    df = pd.read_csv(csv_path)
    return df.sort_values("percentage", ascending=True).reset_index(drop=True)


def plot_dataset(ax, df: pd.DataFrame, title: str) -> None:
    """Plot horizontal bar chart for a dataset."""
    colors = [
        "#27ae60" if p >= 95 else "#2ecc71" if p >= 80 else "#f39c12"
        for p in df["percentage"]
    ]
    y_positions = range(len(df))
    bars = ax.barh(
        y_positions,
        df["percentage"],
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xlim(0, 105)
    ax.axvline(80, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Bootstrap Recovery (%)", fontsize=11, weight="bold")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["edge"], fontsize=9)
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    for bar, pct in zip(bars, df["percentage"]):
        ax.text(
            pct + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )


def main() -> None:
    sns.set_style("whitegrid")
    alfalfa_df = load_bootstrap_results(ALFALFA_CSV)
    nsa_df = load_bootstrap_results(NSA_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)
    plot_dataset(
        axes[0],
        alfalfa_df,
        "ALFALFA × NSA\n(50 runs, 80% subsamples)",
    )
    plot_dataset(
        axes[1],
        nsa_df,
        "NSA-only\n(10 runs, 80% subsamples)",
    )

    fig.suptitle(
        "Bootstrap Edge Confidence Comparison",
        fontsize=16,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

