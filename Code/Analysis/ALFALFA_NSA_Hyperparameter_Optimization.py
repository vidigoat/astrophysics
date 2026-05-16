"""
Hyperparameter sensitivity analysis for ALFALFA×NSA dataset.

Generates mock datasets with known causal structure and evaluates FCIT performance
across different hyperparameter combinations.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import beta  # type: ignore[reportMissingImports]
from pytetrad.tools import TetradSearch as ts  # type: ignore[reportMissingImports]
import networkx as nx  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt

# Initialize JVM for pytetrad
try:
    import jpype
    if not jpype.isJVMStarted():
        jpype.startJVM()
except Exception:
    pass  # JVM might already be started or will be started by pytetrad

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

N_MOCK_DATASETS = 100
FIXED_SAMPLE_SIZE = 20569
N_PROPERTIES = 13
MIN_EDGES = 12
MAX_EDGES = 16
N_HIDDEN_LAYERS = 4
N_NEURONS_PER_LAYER = 50
NOISE_DIST = beta(2, 5)

# Optimal: t=7, p=35.  Wide sweep matching Desmond & Ramsey (2025) Fig. 3 range.
TRUNCATION_LIMITS = [7]
PENALTY_DISCOUNTS = [5, 10, 15, 20, 28, 35, 42, 50, 75, 100, 150, 200]
ALPHAS = [0.01]

# Handle __file__ for notebooks/interactive environments
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Running in notebook/interactive mode
    REPO_ROOT = os.getcwd()
    # Try to find repo root by looking for common markers
    if not os.path.exists(os.path.join(REPO_ROOT, "Code", "Analysis")):
        # Go up directories to find repo root
        for _ in range(3):
            REPO_ROOT = os.path.dirname(REPO_ROOT)
            if os.path.exists(os.path.join(REPO_ROOT, "Code", "Analysis")):
                break

RESULTS_DIR = os.path.join(REPO_ROOT, "Results")
MOCK_DATA_DIR = os.path.join(REPO_ROOT, "Data", "MockDatasets")
PLOTS_DIR = os.path.join(REPO_ROOT, "Plots", "HyperparameterSensitivity")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MOCK_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ALFALFA × NSA properties (13 properties matching real dataset)
PROPERTY_NAMES = [
    'BARYONIC_MASS',
    'COLOR_U_R',
    'ELPETRO_B300',
    'SERSIC_N',
    'ELPETRO_METS',
    'ELPETRO_MTOL',
    'ELPETRO_BA',
    'ELPETRO_TH50_R',
    'ZDIST',
    'logMH',
    'ELPETRO_MASS',
    'ELPETRO_ABSMAG_R',
    'W50'
]

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def generate_mlp(n_inputs, n_outputs, n_hidden_layers=4, n_neurons=50, seed=None):
    """Generate a random MLP with specified architecture."""
    # Use local RNG to avoid affecting global state
    rng = np.random.RandomState(seed)
    
    layers = []
    # Input layer - use larger weights to create stronger signal
    # Scale by sqrt(2/n_inputs) for better gradient flow (He initialization)
    layers.append(rng.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs))
    
    # Hidden layers - use He initialization
    for _ in range(n_hidden_layers - 1):
        layers.append(rng.randn(n_neurons, n_neurons) * np.sqrt(2.0 / n_neurons))
    
    # Output layer - use larger weights
    layers.append(rng.randn(n_neurons, n_outputs) * np.sqrt(2.0 / n_neurons))
    
    return layers

def forward_pass(x, layers):
    """Forward pass through MLP."""
    for layer in layers[:-1]:
        x = relu(x @ layer)
    x = x @ layers[-1]
    return x

def generate_random_dag(n_nodes, min_edges, max_edges, seed=None):
    """Generate a random DAG with specified number of edges."""
    # Use local RNG to avoid affecting global state
    rng = np.random.RandomState(seed)
    
    n_edges = rng.randint(min_edges, max_edges + 1)
    
    # Create DAG using topological ordering
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    edges_added = 0
    
    # Generate all possible valid edges first (topological order ensures DAG)
    valid_edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            valid_edges.append((i, j))
    
    # Shuffle valid edges
    rng.shuffle(valid_edges)
    
    # Add edges until we reach target
    for u, v in valid_edges:
        if edges_added >= n_edges:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            if nx.is_directed_acyclic_graph(G):
                edges_added += 1
            else:
                G.remove_edge(u, v)
    
    # If we didn't get enough edges, try reverse direction
    if edges_added < n_edges:
        for u, v in valid_edges:
            if edges_added >= n_edges:
                break
            if not G.has_edge(v, u) and not G.has_edge(u, v):
                G.add_edge(v, u)
                if nx.is_directed_acyclic_graph(G):
                    edges_added += 1
                else:
                    G.remove_edge(v, u)
    
    return G

def generate_mock_data(dag, n_samples, property_names, seed=None):
    """Generate mock data from a DAG using CPN-like approach."""
    # Use local RNG to avoid affecting global state
    rng = np.random.RandomState(seed)
    
    n_nodes = len(dag.nodes())
    topological_order = list(nx.topological_sort(dag))
    
    # Generate MLPs for each node
    mlps = {}
    for node in topological_order:
        parents = list(dag.predecessors(node))
        n_parents = len(parents)
        
        if n_parents == 0:
            # Root node: just noise
            mlps[node] = None
        else:
            # Generate MLP: parents -> node
            mlp_seed = seed + node if seed is not None else None
            mlps[node] = generate_mlp(n_parents, 1, N_HIDDEN_LAYERS, N_NEURONS_PER_LAYER, mlp_seed)
    
    # Generate data
    data = np.zeros((n_samples, n_nodes))
    
    for node in topological_order:
        parents = list(dag.predecessors(node))
        
        if len(parents) == 0:
            # Root node: sample from noise distribution
            # Use rng to generate uniform, then transform to beta for reproducibility
            uniform = rng.random(size=n_samples)
            noise = beta.ppf(uniform, 2, 5)  # Inverse CDF transform
            # Scale and shift to reasonable range
            data[:, node] = (noise - 0.5) * 10  # Scale to roughly [-5, 5]
        else:
            # Get parent values
            parent_values = data[:, parents]
            
            # Forward pass through MLP
            mlp_output = forward_pass(parent_values, mlps[node])
            mlp_output = mlp_output.flatten()
            
            # Scale MLP output to ensure signal is strong enough
            # Normalize to have similar scale as parent values
            if np.std(mlp_output) > 0:
                mlp_output = mlp_output / np.std(mlp_output) * np.std(parent_values[:, 0]) if len(parents) > 0 else mlp_output
            
            # Add noise - use smaller noise relative to signal
            uniform = rng.random(size=n_samples)
            noise = beta.ppf(uniform, 2, 5)  # Inverse CDF transform
            noise_scaled = (noise - 0.5) * 0.5  # Reduced noise scale (was 2.0, now 0.5)
            
            data[:, node] = mlp_output + noise_scaled
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=property_names[:n_nodes])
    return df, dag

def extract_edges_from_graph(graph_str):
    """Extract edges from FCIT graph string."""
    import re
    edges = set()
    lines = [line.strip() for line in graph_str.split("\n") if line.strip() and not line.startswith("Graph")]
    
    # Patterns for different edge types
    # Directed: A --> B or A o-> B
    # Undirected: A o-o B
    directed_pattern = re.compile(r'([^\s]+)\s*(?:-->|o->)\s*([^\s]+)')
    undirected_pattern = re.compile(r'([^\s]+)\s*o-o\s*([^\s]+)')
    
    for line in lines:
        # Try directed edges first
        match = directed_pattern.search(line)
        if match:
            src = match.group(1).strip()
            tgt = match.group(2).strip()
            # Clean variable names (remove package prefix if present)
            if "." in src:
                src = src.split(".", 1)[1].strip()
            if "." in tgt:
                tgt = tgt.split(".", 1)[1].strip()
            # Remove any trailing punctuation or labels
            src = re.sub(r'[^\w].*$', '', src)
            tgt = re.sub(r'[^\w].*$', '', tgt)
            if src and tgt:
                edges.add((src, tgt))
            continue
        
        # Try undirected edges
        match = undirected_pattern.search(line)
        if match:
            node1 = match.group(1).strip()
            node2 = match.group(2).strip()
            # Clean variable names
            if "." in node1:
                node1 = node1.split(".", 1)[1].strip()
            if "." in node2:
                node2 = node2.split(".", 1)[1].strip()
            # Remove any trailing punctuation
            node1 = re.sub(r'[^\w].*$', '', node1)
            node2 = re.sub(r'[^\w].*$', '', node2)
            if node1 and node2:
                # Add both directions for undirected edge
                edges.add((node1, node2))
                edges.add((node2, node1))
    
    return edges

def dag_to_edge_set(dag, property_names):
    """Convert DAG to set of edges (as tuples of property names)."""
    edges = set()
    for u, v in dag.edges():
        edges.add((property_names[u], property_names[v]))
    return edges

def calculate_skeleton_edges(edges):
    """Convert directed edges to undirected skeleton."""
    skeleton = set()
    for u, v in edges:
        # Add both directions as undirected edge
        pair = tuple(sorted([u, v]))
        skeleton.add(pair)
    return skeleton

def calculate_shd(true_edges, predicted_edges):
    """Calculate Structural Hamming Distance (SHD)."""
    true_skeleton = calculate_skeleton_edges(true_edges)
    pred_skeleton = calculate_skeleton_edges(predicted_edges)
    
    # SHD = edges in true but not pred + edges in pred but not true
    shd = len(true_skeleton - pred_skeleton) + len(pred_skeleton - true_skeleton)
    return shd

def calculate_orientation_accuracy(true_edges, predicted_edges):
    """Calculate accuracy of edge orientations."""
    true_set = set(true_edges)
    pred_set = set(predicted_edges)
    
    # Count correctly oriented edges
    correct_orientations = len(true_set & pred_set)
    total_true = len(true_set)
    
    accuracy = correct_orientations / total_true if total_true > 0 else 0.0
    return accuracy

def calculate_edge_type_breakdown(predicted_edges):
    """Count edge types in predicted graph."""
    directed = set()
    undirected = set()
    processed = set()  # Track which edges we've already counted
    
    for edge in predicted_edges:
        if edge in processed:
            continue
            
        reverse = (edge[1], edge[0])
        if reverse in predicted_edges:
            # Edge appears in both directions = undirected
            undirected.add(tuple(sorted(edge)))
            processed.add(edge)
            processed.add(reverse)
        else:
            # Edge appears in only one direction = directed
            directed.add(edge)
            processed.add(edge)
    
    return len(directed), len(undirected)

def calculate_metrics(true_edges, predicted_edges):
    """Calculate comprehensive metrics: precision, recall, F1, SHD, orientation accuracy."""
    true_set = set(true_edges)
    pred_set = set(predicted_edges)
    
    # Convert to skeleton (undirected) for comparison
    # This handles cases where direction is wrong but edge exists
    true_skeleton = set()
    for u, v in true_set:
        true_skeleton.add(tuple(sorted([u, v])))
    
    pred_skeleton = set()
    for u, v in pred_set:
        pred_skeleton.add(tuple(sorted([u, v])))
    
    # True positives: edges in both skeletons (correct edge, regardless of direction)
    tp = len(true_skeleton & pred_skeleton)
    
    # False positives: edges in pred but not in true
    fp = len(pred_skeleton - true_skeleton)
    
    # False negatives: edges in true but not in pred
    fn = len(true_skeleton - pred_skeleton)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance
    shd = calculate_shd(true_edges, predicted_edges)
    
    # Orientation accuracy
    orientation_acc = calculate_orientation_accuracy(true_edges, predicted_edges)
    
    # Edge type breakdown
    n_directed, n_undirected = calculate_edge_type_breakdown(predicted_edges)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'orientation_accuracy': orientation_acc,
        'n_directed_edges': n_directed,
        'n_undirected_edges': n_undirected
    }

def main():
    mock_datasets_path = os.path.join(MOCK_DATA_DIR, 'alfalfa_nsa_mock_datasets.pkl')
    
    if os.path.exists(mock_datasets_path):
        with open(mock_datasets_path, 'rb') as f:
            mock_datasets = pickle.load(f)
    else:
        mock_datasets = []
        for i in range(N_MOCK_DATASETS):
            n_samples = FIXED_SAMPLE_SIZE
            dag = generate_random_dag(N_PROPERTIES, MIN_EDGES, MAX_EDGES, seed=42+i)
            df, true_dag = generate_mock_data(dag, n_samples, PROPERTY_NAMES, seed=42+i*1000)
            true_edges = dag_to_edge_set(true_dag, PROPERTY_NAMES)
            mock_datasets.append({
                'dataset_id': i,
                'data': df,
                'dag': true_dag,
                'true_edges': true_edges,
                'n_samples': n_samples
            })
        with open(mock_datasets_path, 'wb') as f:
            pickle.dump(mock_datasets, f)
    
    total_runs = len(TRUNCATION_LIMITS) * len(PENALTY_DISCOUNTS) * len(ALPHAS) * N_MOCK_DATASETS
    results = []
    run_num = 0
    successful_runs = 0
    failed_runs = 0
    
    print(f"\n{'='*70}")
    print(f"Starting ALFALFA×NSA hyperparameter sensitivity analysis")
    print(f"Total runs: {total_runs}")
    print(f"Truncation limits: {TRUNCATION_LIMITS}")
    print(f"Penalty discounts: {PENALTY_DISCOUNTS}")
    print(f"Mock datasets: {N_MOCK_DATASETS}")
    print(f"{'='*70}\n")
    
    for trunc_limit in TRUNCATION_LIMITS:
        for penalty_discount in PENALTY_DISCOUNTS:
            for alpha in ALPHAS:
                print(f"\nProcessing: t={trunc_limit}, p={penalty_discount}, α={alpha}")
                dataset_count = 0
                
                for mock_data in mock_datasets[:N_MOCK_DATASETS]:
                    run_num += 1
                    dataset_count += 1
                    df = mock_data['data']
                    true_edges = mock_data['true_edges']
                    
                    try:
                        # Check and restart JVM if needed
                        try:
                            import jpype
                            if not jpype.isJVMStarted():
                                jpype.startJVM()
                        except Exception:
                            pass
                        
                        search = ts.TetradSearch(df)
                        search.set_verbose(False)
                        search.use_basis_function_lrt(truncation_limit=trunc_limit, alpha=alpha)
                        search.use_basis_function_bic(truncation_limit=trunc_limit, penalty_discount=penalty_discount)
                        search.run_fcit()
                        
                        graph_str = str(search.get_java())
                        predicted_edges = extract_edges_from_graph(graph_str)
                        metrics = calculate_metrics(true_edges, predicted_edges)
                        
                        results.append({
                            'truncation_limit': trunc_limit,
                            'penalty_discount': penalty_discount,
                            'alpha': alpha,
                            'dataset_id': mock_data['dataset_id'],
                            'n_samples': mock_data['n_samples'],
                            'n_true_edges': len(true_edges),
                            'n_pred_edges': len(predicted_edges),
                            **metrics
                        })
                        successful_runs += 1
                        
                        if dataset_count % 10 == 0 or dataset_count == N_MOCK_DATASETS:
                            print(f"  Completed {dataset_count}/{N_MOCK_DATASETS} datasets (Run {run_num}/{total_runs}, Success: {successful_runs}, Failed: {failed_runs})")
                    except Exception as e:
                        failed_runs += 1
                        error_msg = str(e)
                        # Check if JVM crashed
                        if 'JVMNotRunning' in error_msg or 'JVM' in error_msg:
                            if dataset_count <= 3 or dataset_count % 10 == 0:
                                print(f"  JVM error at dataset {dataset_count}: {error_msg[:100]}")
                                print(f"  Note: JVM may have crashed. Consider restarting the script.")
                        else:
                            if dataset_count <= 3 or dataset_count % 10 == 0:
                                print(f"  Error at dataset {dataset_count}: {type(e).__name__}: {error_msg[:100]}")
                        continue
                
                print(f"Finished: t={trunc_limit}, p={penalty_discount}, α={alpha} ({dataset_count} datasets)")
    
    results_df = pd.DataFrame(results)
    
    # Calculate statistics by hyperparameter combination
    summary = []
    for trunc_limit in TRUNCATION_LIMITS:
        for penalty_discount in PENALTY_DISCOUNTS:
            for alpha in ALPHAS:
                mask = (results_df['truncation_limit'] == trunc_limit) & \
                       (results_df['penalty_discount'] == penalty_discount) & \
                       (results_df['alpha'] == alpha)
                
                subset = results_df[mask]
                if len(subset) > 0:
                    summary.append({
                        'truncation_limit': trunc_limit,
                        'penalty_discount': penalty_discount,
                        'alpha': alpha,
                        'mean_precision': subset['precision'].mean(),
                        'mean_recall': subset['recall'].mean(),
                        'mean_f1': subset['f1'].mean(),
                        'mean_shd': subset['shd'].mean(),
                        'mean_orientation_acc': subset['orientation_accuracy'].mean(),
                        'std_precision': subset['precision'].std(),
                        'std_recall': subset['recall'].std(),
                        'std_f1': subset['f1'].std(),
                        'std_shd': subset['shd'].std(),
                        'p16_precision': subset['precision'].quantile(0.16),
                        'p84_precision': subset['precision'].quantile(0.84),
                        'p16_recall': subset['recall'].quantile(0.16),
                        'p84_recall': subset['recall'].quantile(0.84),
                        'p16_f1': subset['f1'].quantile(0.16),
                        'p84_f1': subset['f1'].quantile(0.84),
                        'n_datasets': len(subset)
                    })
    
    results_df['size_category'] = pd.cut(
        results_df['n_samples'],
        bins=[0, 50000, 200000, float('inf')],
        labels=['Small (10k-50k)', 'Medium (50k-200k)', 'Large (200k-500k)']
    )
    results_df['complexity_category'] = pd.cut(
        results_df['n_true_edges'],
        bins=[0, 10, 15, float('inf')],
        labels=['Sparse (5-10)', 'Medium (10-15)', 'Dense (15-20)']
    )
    
    stratified_summary = []
    for size_cat in results_df['size_category'].unique():
        for complexity_cat in results_df['complexity_category'].unique():
            for trunc_limit in TRUNCATION_LIMITS:
                for penalty_discount in PENALTY_DISCOUNTS:
                    for alpha in ALPHAS:
                        mask = (results_df['size_category'] == size_cat) & \
                               (results_df['complexity_category'] == complexity_cat) & \
                               (results_df['truncation_limit'] == trunc_limit) & \
                               (results_df['penalty_discount'] == penalty_discount) & \
                               (results_df['alpha'] == alpha)
                        subset = results_df[mask]
                        if len(subset) > 0:
                            stratified_summary.append({
                                'size_category': size_cat,
                                'complexity_category': complexity_cat,
                                'truncation_limit': trunc_limit,
                                'penalty_discount': penalty_discount,
                                'alpha': alpha,
                                'mean_f1': subset['f1'].mean(),
                                'mean_precision': subset['precision'].mean(),
                                'mean_recall': subset['recall'].mean(),
                                'mean_shd': subset['shd'].mean(),
                                'n_datasets': len(subset)
                            })
    
    summary_df = pd.DataFrame(summary)
    stratified_df = pd.DataFrame(stratified_summary)
    
    if len(summary_df) == 0:
        raise ValueError("No results to save. Check if FCIT runs completed successfully.")
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'alfalfa_nsa_mock_data_hyperparameter_sensitivity.csv'), index=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'alfalfa_nsa_mock_data_hyperparameter_summary.csv'), index=False)
    if len(stratified_df) > 0:
        stratified_df.to_csv(os.path.join(RESULTS_DIR, 'alfalfa_nsa_mock_data_stratified_analysis.csv'), index=False)
    
    best = summary_df.loc[summary_df['mean_f1'].idxmax()]
    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"Total runs: {run_num}/{total_runs}")
    print(f"Successful: {successful_runs}, Failed: {failed_runs}")
    print(f"Best hyperparameters: t={best['truncation_limit']}, p={best['penalty_discount']}, "
          f"F1={best['mean_f1']:.3f}±{best['std_f1']:.3f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

