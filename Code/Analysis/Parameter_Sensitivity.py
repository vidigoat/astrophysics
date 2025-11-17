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
import torch
from torch import nn
from pytetrad.tools import TetradSearch as ts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALFALFA_EDGE_MODELS = [
    ("logMH", "BARYONIC_MASS", 0.9),
    ("logMH", "ELPETRO_MASS", 0.6),
    ("logMH", "ZDIST", 0.3),
    ("logMH", "W50", 0.8),
    ("BARYONIC_MASS", "ELPETRO_MASS", 0.8),
    ("ELPETRO_MASS", "ELPETRO_ABSMAG_R", -0.7),
    ("ELPETRO_TH50_R", "ELPETRO_MASS", 0.3),
    ("SERSIC_N", "ELPETRO_TH50_R", 0.5),
    ("ELPETRO_BA", "ELPETRO_TH50_R", 0.4),
    ("COLOR_U_R", "ELPETRO_MTOL", 0.7),
    ("ELPETRO_MTOL", "BARYONIC_MASS", 0.35),
    ("ELPETRO_B300", "COLOR_U_R", -0.5),
    ("ELPETRO_METS", "ELPETRO_MASS", 0.4),
]

NSA_EDGE_MODELS = [
    ("COLOR_U_R", "ELPETRO_B300", -0.6),
    ("COLOR_U_R", "ELPETRO_ABSMAG_R", -0.7),
    ("COLOR_U_R", "ELPETRO_MTOL", 0.6),
    ("ELPETRO_B300", "ELPETRO_ABSMAG_R", -0.4),
    ("ELPETRO_ABSMAG_R", "ELPETRO_MASS", 0.8),
    ("ELPETRO_MTOL", "ELPETRO_MASS", 0.5),
    ("ELPETRO_METS", "ELPETRO_MASS", 0.6),
    ("ELPETRO_BA", "ELPETRO_TH50_R", 0.3),
    ("SERSIC_N", "ELPETRO_TH50_R", 0.5),
    ("ELPETRO_TH50_R", "ELPETRO_MASS", 0.3),
    ("ZDIST", "ELPETRO_MASS", -0.2),
]

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
    "edge_models": ALFALFA_EDGE_MODELS,
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
    "edge_models": NSA_EDGE_MODELS,
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
# Mock data generation with nonlinear MLP SEM
# ---------------------------------------------------------------------------


class MLPSEM(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], activation=nn.ReLU()):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def topological_order(variables: List[str], edge_models: List[Tuple[str, str, float]]) -> List[str]:
    parents = {var: [] for var in variables}
    for parent, child, _ in edge_models:
        parents[child].append(parent)

    remaining = set(variables)
    order = []
    while remaining:
        progressed = False
        for var in list(remaining):
            if all(parent in order for parent in parents[var]):
                order.append(var)
                remaining.remove(var)
                progressed = True
        if not progressed:
            raise ValueError("Cycle detected in edge models")
    return order


def simulate_mlp_sem(config: Dict, seed: int) -> Tuple[pd.DataFrame, set]:
    rng = np.random.default_rng(seed)
    variables = config["variables"]
    edge_models = config["edge_models"]
    n = config["sample_size"]

    parents = {var: [] for var in variables}
    for parent, child, weight in edge_models:
        parents[child].append((parent, weight))

    order = topological_order(variables, edge_models)
    data = np.zeros((n, len(variables)), dtype=np.float32)
    var_index = {var: idx for idx, var in enumerate(variables)}

    base_noise = rng.beta(2, 5, size=(n, len(variables))).astype(np.float32)
    base_noise = (base_noise - base_noise.mean(axis=0)) / (base_noise.std(axis=0) + 1e-6)

    for var in order:
        idx = var_index[var]
        parent_info = parents[var]
        if not parent_info:
            data[:, idx] = base_noise[:, idx]
            continue

        parent_indices = [var_index[p] for p, _ in parent_info]
        hidden_layers = [50, 50, 50, 50]
        mlp = MLPSEM(len(parent_indices), hidden_layers)
        torch.manual_seed(seed + idx)
        mlp.apply(lambda m: isinstance(m, nn.Linear) and nn.init.xavier_uniform_(m.weight))

        with torch.no_grad():
            parent_vals = torch.from_numpy(data[:, parent_indices])
            mlp_out = mlp(parent_vals).squeeze().numpy()

        signal = (mlp_out - mlp_out.mean()) / (mlp_out.std() + 1e-6)
        avg_weight = np.mean([abs(w) for _, w in parent_info])
        signal *= avg_weight
        data[:, idx] = signal + base_noise[:, idx]

    df = pd.DataFrame(data, columns=variables)
    true_edges = {frozenset({parent, child}) for parent, child, _ in edge_models}
    return df, true_edges


def generate_mock_data(config: Dict, seed: int) -> Tuple[pd.DataFrame, set]:
    return simulate_mlp_sem(config, seed)


# ---------------------------------------------------------------------------
# FCIT execution helper
# ---------------------------------------------------------------------------

def parse_edges(graph_str: str) -> set:
    relation_tokens = ["-->", "<--", "o-o", "o->", "<-o"]
    edges = set()
    for line in graph_str.splitlines():
        line = line.strip()
        if not line or line.startswith("Graph"):
            continue
        if not any(token in line for token in relation_tokens):
            continue
        parts = line.replace(":", " ").split()
        if parts and parts[0].rstrip(".").isdigit():
            parts = parts[1:]
        if len(parts) < 3:
            continue
        source = parts[0]
        relation = parts[1]
        target = parts[2]
        if relation in relation_tokens:
            edges.add(frozenset({source, target}))
    return edges


def run_fcit_with_penalty(df: pd.DataFrame, penalty: int) -> set:
    search = ts.TetradSearch(df)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=ALPHA)
    search.use_basis_function_bic(
        truncation_limit=TRUNCATION_LIMIT,
        penalty_discount=penalty,
    )
    search.run_fcit()
    graph_str = str(search.get_java())
    return parse_edges(graph_str)


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


def generate_random_seeds(config: Dict, master_seed: int = 1234) -> List[int]:
    rng = np.random.default_rng(master_seed)
    return rng.integers(
        low=0,
        high=2**31 - 1,
        size=config["n_datasets"],
        dtype=np.int64,
    ).tolist()


if __name__ == "__main__":
    print(
        "Parameter sensitivity script ready.\n"
        "Recommended workflow:\n"
        " 1. Generate seeds via generate_random_seeds(config).\n"
        " 2. Optionally reduce n_datasets for smoke tests.\n"
        " 3. Call run_parameter_sensitivity(config, seeds).\n"
        "WARNING: Full study (500 + 100 datasets) is computationally expensive."
    )

