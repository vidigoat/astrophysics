"""
Parameter sensitivity study for FCIT penalty_discount.

This script generates mock datasets patterned after the real ALFALFA × NSA
and NSA-only samples, runs FCIT with varying penalty_discount values, and
computes precision/recall/F1 scores relative to the known ground-truth
graph for each mock dataset.

The script is organized so that each component can be tested independently
before the full run (which is computationally intensive). DO NOT run the
full study on a laptop without reviewing the expected runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pytetrad.tools import TetradSearch as ts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALFALFA_CONFIG = {
    "name": "ALFALFA × NSA",
    "n_datasets": 500,
    "sample_size": 20569,
    "variables": [
        "BARYONIC_MASS",
        "COLOR_U_R",
        "ELPETRO_B300",
        "SERSIC_N",
        "ELPETRO_METS",
        "ELPETRO_MTOL",
        "ELPETRO_BA",
        "ELPETRO_TH50_R",
        "ZDIST",
        "logMH",
        "ELPETRO_MASS",
        "ELPETRO_ABSMAG_R",
        "W50",
    ],
}

NSA_CONFIG = {
    "name": "NSA-only",
    "n_datasets": 100,
    "sample_size": 484539,
    "variables": [
        "COLOR_U_R",
        "ELPETRO_B300",
        "ELPETRO_METS",
        "ELPETRO_MTOL",
        "ELPETRO_BA",
        "ELPETRO_TH50_R",
        "ELPETRO_MASS",
        "ELPETRO_ABSMAG_R",
        "SERSIC_N",
        "ZDIST",
    ],
}

PENALTY_GRID = [20, 30, 40, 50, 60, 70]
TRUNCATION_LIMIT = 14
ALPHA = 0.01

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Plots",
    "Analysis",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MockDatasetResult:
    penalty: int
    precision: float
    recall: float
    f1: float


# ---------------------------------------------------------------------------
# Mock data generation (placeholder implementation)
# ---------------------------------------------------------------------------

def generate_mock_data(config: Dict, seed: int) -> Tuple[pd.DataFrame, set]:
    """
    Generate a mock dataset and ground-truth edge set.

    TODO:
        - Implement SEM-based simulation using real PAG structure as template.
        - Return (dataframe, set_of_true_edges).
    """
    rng = np.random.default_rng(seed)
    n = config["sample_size"]
    variables = config["variables"]
    df = pd.DataFrame(rng.normal(size=(n, len(variables))), columns=variables)
    # Placeholder: empty ground-truth set (replace with actual structure)
    true_edges: set = set()
    return df, true_edges


# ---------------------------------------------------------------------------
# FCIT execution helper
# ---------------------------------------------------------------------------

def run_fcit_with_penalty(df: pd.DataFrame, penalty: int) -> set:
    """
    Run FCIT with the specified penalty and return discovered edges as tuples.
    """
    search = ts.TetradSearch(df)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=ALPHA)
    search.use_basis_function_bic(
        truncation_limit=TRUNCATION_LIMIT,
        penalty_discount=penalty,
    )
    search.run_fcit()
    graph_str = str(search.get_java())
    edges = {
        tuple(line.split())
        for line in graph_str.splitlines()
        if any(token in line for token in ["-->", "<--", "o-o", "o->", "<-o"])
    }
    return edges


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(discovered: set, true_edges: set) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.
    """
    true_positives = len(discovered & true_edges)
    false_positives = len(discovered - true_edges)
    false_negatives = len(true_edges - discovered)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------

def aggregate_results(results: List[MockDatasetResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    summary = (
        df.groupby("penalty")
        .agg(["mean", lambda x: np.percentile(x, 16), lambda x: np.percentile(x, 84)])
        .reset_index()
    )
    summary.columns = [
        "penalty",
        "precision_mean",
        "precision_p16",
        "precision_p84",
        "recall_mean",
        "recall_p16",
        "recall_p84",
        "f1_mean",
        "f1_p16",
        "f1_p84",
    ]
    return summary


def plot_metric_curves(summary: pd.DataFrame, config: Dict) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    for metric in ["precision", "recall", "f1"]:
        plt.plot(
            summary["penalty"],
            summary[f"{metric}_mean"],
            label=f"{metric.title()} (mean)",
        )
        plt.fill_between(
            summary["penalty"],
            summary[f"{metric}_p16"],
            summary[f"{metric}_p84"],
            alpha=0.2,
        )
    plt.axvline(50, color="gray", linestyle="--", linewidth=1.2, label="Penalty=50")
    plt.xlabel("Penalty Discount", fontsize=12, weight="bold")
    plt.ylabel("Score", fontsize=12, weight="bold")
    plt.title(
        f"{config['name']} – Precision/Recall/F1 vs. Penalty Discount\n"
        f"(Truncation Limit = {TRUNCATION_LIMIT}, Alpha = {ALPHA})",
        fontsize=13,
        weight="bold",
    )
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(
        OUTPUT_DIR,
        f"{config['name'].replace(' ', '_')}_penalty_sensitivity.png",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_file}")


# ---------------------------------------------------------------------------
# Main driver (not executed automatically)
# ---------------------------------------------------------------------------

def run_parameter_sensitivity(config: Dict, seeds: List[int]) -> None:
    """
    Full pipeline for one dataset configuration.

    Steps:
        1. Generate mock datasets (potentially expensive).
        2. For each penalty and dataset, run FCIT.
        3. Compute metrics and aggregate.
        4. Plot curves.

    TODO:
        - Parallelize heavy loops (multiprocessing or joblib).
        - Cache intermediate results to disk for resuming.
    """
    results: List[MockDatasetResult] = []
    for dataset_idx, seed in enumerate(seeds):
        df, true_edges = generate_mock_data(config, seed)
        for penalty in PENALTY_GRID:
            discovered_edges = run_fcit_with_penalty(df, penalty)
            precision, recall, f1 = compute_metrics(discovered_edges, true_edges)
            results.append(MockDatasetResult(penalty, precision, recall, f1))
        print(
            f"[{config['name']}] Finished mock dataset "
            f"{dataset_idx + 1}/{len(seeds)}"
        )

    summary = aggregate_results(results)
    plot_metric_curves(summary, config)


if __name__ == "__main__":
    print(
        "This script is a template for parameter sensitivity analysis.\n"
        "Populate the mock data generator and call run_parameter_sensitivity "
        "with appropriate seeds before running the full study."
    )

