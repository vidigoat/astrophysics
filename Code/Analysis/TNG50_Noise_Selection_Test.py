"""
Test effect of observational noise and selection on TNG50 graph metrics.
Compares: Original TNG50 vs TNG50+Noise vs TNG50+Noise+Selection
"""
import os
import pickle
import numpy as np
import pandas as pd
from pytetrad.tools import TetradSearch as ts

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "tng50_final.pkl")
RESULTS_DIR = os.path.join(REPO_ROOT, "Results")
PLOTS_DIR = os.path.join(REPO_ROOT, "Plots", "RobustnessTests")
os.makedirs(PLOTS_DIR, exist_ok=True)

PVAL_THRESHOLD = 0.01
TRUNCATION_LIMIT = 7
PENALTY_DISCOUNT = 15
RANDOM_SEED = 42

def clean_token(token: str) -> str:
    token = token.strip()
    if "." in token:
        left, right = token.split(".", 1)
        if left.strip().isdigit():
            return right.strip()
    return token

def extract_edges_from_graph(graph_str):
    """Extract edges from graph string."""
    lines = [l.strip() for l in graph_str.split("\n") if l.strip() and not l.startswith("Graph")]
    edges = []
    for line in lines:
        if "-->" in line:
            parts = line.split("-->")
            edge = f"{clean_token(parts[0])} --> {clean_token(parts[1])}"
            edges.append(edge)
        elif "o->" in line:
            parts = line.split("o->")
            edge = f"{clean_token(parts[0])} o-> {clean_token(parts[1])}"
            edges.append(edge)
        elif "o-o" in line:
            parts = line.split("o-o")
            nodes = sorted([clean_token(parts[0]), clean_token(parts[1])])
            edge = f"{nodes[0]} o-o {nodes[1]}"
            edges.append(edge)
    return edges

def compute_graph_metrics(edges, n_variables):
    """Compute graph metrics."""
    total_edges = len(edges)
    max_possible = n_variables * (n_variables - 1) / 2
    edge_density = total_edges / max_possible if max_possible > 0 else 0
    
    directed_count = sum(1 for e in edges if "-->" in e)
    partial_count = sum(1 for e in edges if "o->" in e)
    
    directed_fraction = directed_count / total_edges if total_edges > 0 else 0
    orientation_fraction = (directed_count + partial_count) / total_edges if total_edges > 0 else 0
    
    return {
        'edge_density': edge_density,
        'directed_fraction': directed_fraction,
        'orientation_fraction': orientation_fraction,
        'total_edges': total_edges
    }

def add_observational_noise(data_dict, random_seed=42):
    """Add Gaussian noise to simulate observational errors."""
    np.random.seed(random_seed)
    noisy_data = {}
    
    # Define noise levels (standard deviations)
    noise_levels = {
        'DM_MASS': 0.1,  # Mass measurements: ~0.1 dex
        'STELLAR_MASS': 0.1,
        'GAS_MASS': 0.15,  # Gas mass less precise
        'BH_MASS': 0.2,  # BH mass less precise
        'BARYONIC_MASS': 0.1,
        'HALFMASS_RAD': 0.05,  # Size measurements more precise
        'VEL_DISP': 0.05,  # Velocity measurements precise
        'VMAX': 0.05,
        'GAS_METALLICITY': 0.2,  # Metallicity less precise
        'STAR_METALLICITY': 0.2,
        'PHOTOMETRIC_U': 0.05,  # Photometry precise
        'PHOTOMETRIC_R': 0.05,
        'SFR': 0.15,  # SFR less precise
        'COLOUR': 0.05  # Color difference precise
    }
    
    for key, values in data_dict.items():
        noise_std = noise_levels.get(key, 0.1)  # Default 0.1
        noise = np.random.normal(0, noise_std, size=len(values))
        noisy_data[key] = values + noise
    
    return noisy_data

def apply_alfalfa_selection(data_dict):
    """Apply ALFALFA-like selection cuts (blue, star-forming, gas-rich)."""
    # Blue galaxies: U-R < 0.5
    blue_mask = data_dict['COLOUR'] < 0.5
    
    # Star-forming: SFR > 10^-2 M☉/yr (log space: log10(SFR) > -2)
    sfr_mask = data_dict['SFR'] > -2.0
    
    # Gas-rich: GAS_MASS > 8.0 (log space)
    gas_mask = data_dict['GAS_MASS'] > 8.0
    
    # Combined selection
    selection_mask = blue_mask & sfr_mask & gas_mask
    
    # Apply selection
    selected_data = {}
    for key, values in data_dict.items():
        selected_data[key] = values[selection_mask]
    
    return selected_data, np.sum(selection_mask)

print("Loading TNG50 data...")
with open(DATA_PATH, "rb") as f:
    original_data = pickle.load(f)

var_names = list(original_data.keys())
n_variables = len(var_names)
n_original = len(original_data[var_names[0]])

print(f"Original TNG50: {n_original:,} galaxies, {n_variables} variables")

# Create noisy version
print("\nCreating TNG50 + Noise version...")
noisy_data = add_observational_noise(original_data, random_seed=RANDOM_SEED)
n_noisy = len(noisy_data[var_names[0]])

# Create noisy + selection version
print("Creating TNG50 + Noise + Selection version...")
selected_data, n_selected = apply_alfalfa_selection(noisy_data)
print(f"  Selection kept {n_selected:,} galaxies ({100*n_selected/n_noisy:.1f}%)")

results = []

# Test 1: Original TNG50
print("\n" + "="*60)
print("Test 1: Original TNG50")
print("="*60)
data = np.column_stack([original_data[var] for var in var_names])
df = pd.DataFrame(data, columns=var_names)

search = ts.TetradSearch(df)
search.set_verbose(False)
search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=PVAL_THRESHOLD)
search.use_basis_function_bic(truncation_limit=TRUNCATION_LIMIT, penalty_discount=PENALTY_DISCOUNT)
search.run_fcit()

graph_str = str(search.get_java())
edges = extract_edges_from_graph(graph_str)
metrics = compute_graph_metrics(edges, n_variables)
metrics['version'] = 'Original'
metrics['n_galaxies'] = n_original
results.append(metrics)
print(f"  Edge density: {metrics['edge_density']:.3f}")
print(f"  Total edges: {metrics['total_edges']}")

# Test 2: TNG50 + Noise
print("\n" + "="*60)
print("Test 2: TNG50 + Noise")
print("="*60)
data = np.column_stack([noisy_data[var] for var in var_names])
df = pd.DataFrame(data, columns=var_names)

search = ts.TetradSearch(df)
search.set_verbose(False)
search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=PVAL_THRESHOLD)
search.use_basis_function_bic(truncation_limit=TRUNCATION_LIMIT, penalty_discount=PENALTY_DISCOUNT)
search.run_fcit()

graph_str = str(search.get_java())
edges = extract_edges_from_graph(graph_str)
metrics = compute_graph_metrics(edges, n_variables)
metrics['version'] = 'Noise'
metrics['n_galaxies'] = n_noisy
results.append(metrics)
print(f"  Edge density: {metrics['edge_density']:.3f}")
print(f"  Total edges: {metrics['total_edges']}")

# Test 3: TNG50 + Noise + Selection
print("\n" + "="*60)
print("Test 3: TNG50 + Noise + Selection")
print("="*60)
data = np.column_stack([selected_data[var] for var in var_names])
df = pd.DataFrame(data, columns=var_names)

search = ts.TetradSearch(df)
search.set_verbose(False)
search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=PVAL_THRESHOLD)
search.use_basis_function_bic(truncation_limit=TRUNCATION_LIMIT, penalty_discount=PENALTY_DISCOUNT)
search.run_fcit()

graph_str = str(search.get_java())
edges = extract_edges_from_graph(graph_str)
metrics = compute_graph_metrics(edges, n_variables)
metrics['version'] = 'Noise+Selection'
metrics['n_galaxies'] = n_selected
results.append(metrics)
print(f"  Edge density: {metrics['edge_density']:.3f}")
print(f"  Total edges: {metrics['total_edges']}")

# Save results
results_df = pd.DataFrame(results)
output_path = os.path.join(RESULTS_DIR, "tng50_noise_selection_test.csv")
results_df.to_csv(output_path, index=False)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nEdge Density Comparison:")
for _, row in results_df.iterrows():
    print(f"  {row['version']:20s}: {row['edge_density']:.3f} (N={row['n_galaxies']:,})")

print(f"\nResults saved to: {output_path}")


