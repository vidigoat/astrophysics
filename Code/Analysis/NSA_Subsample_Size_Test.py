"""
Test if TNG50's higher edge density is due to sample size.
Subsamples NSA to match TNG50 size (10,992 galaxies) and compares edge densities.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pytetrad.tools import TetradSearch as ts
import matplotlib.pyplot as plt

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
NSA_DATA_PATH = os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl")
TNG50_RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "tng50_fcit_t7_p15.txt")
RESULTS_DIR = os.path.join(REPO_ROOT, "Results")
PLOTS_DIR = os.path.join(REPO_ROOT, "Plots", "RobustnessTests")
os.makedirs(PLOTS_DIR, exist_ok=True)

PVAL_THRESHOLD = 0.01
TRUNC_LIMIT = 14
PENALTY_DISCOUNT = 50
N_SUBSAMPLES = 10
TNG50_SIZE = 10992  # From TNG50 results file
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

def compute_edge_density(edges, n_variables):
    """Compute edge density."""
    total_edges = len(edges)
    max_possible = n_variables * (n_variables - 1) / 2
    return total_edges / max_possible if max_possible > 0 else 0

def get_tng50_edge_density():
    """Get TNG50 edge density from results file."""
    with open(TNG50_RESULTS_PATH, 'r') as f:
        content = f.read()
    
    # Extract edges
    if "Graph Edges:" in content:
        edges_section = content.split("Graph Edges:")[1]
        edges = []
        for line in edges_section.split('\n'):
            line = line.strip()
            if not line or line.startswith('='):
                continue
            # Remove leading number
            line = line.split('.', 1)[-1].strip() if '.' in line else line
            if any(arrow in line for arrow in ['-->', 'o-o', 'o->']):
                edges.append(line)
        
        # Count variables
        if "Variables:" in content:
            var_section = content.split("Variables:")[1].split("Graph")[0]
            n_vars = len([l for l in var_section.split('\n') if l.strip() and not l.startswith('=')])
        else:
            n_vars = 14  # Default for TNG50
        
        return compute_edge_density(edges, n_vars), len(edges)
    return None, None

print("Loading NSA data...")
with open(NSA_DATA_PATH, "rb") as f:
    nsa_data = pickle.load(f)

var_names = list(nsa_data.keys())
n_variables = len(var_names)
n_total = len(nsa_data[var_names[0]])

print(f"NSA total: {n_total:,} galaxies")
print(f"TNG50 size: {TNG50_SIZE:,} galaxies")
print(f"Subsampling NSA {N_SUBSAMPLES} times to match TNG50 size...")

# Get TNG50 edge density
tng50_density, tng50_edges = get_tng50_edge_density()
print(f"\nTNG50 edge density: {tng50_density:.3f} ({tng50_edges} edges)")

# Run subsamples
subsample_densities = []
np.random.seed(RANDOM_SEED)

for i in range(N_SUBSAMPLES):
    print(f"\nSubsample {i+1}/{N_SUBSAMPLES}...")
    
    # Random subsample
    sample_idx = np.random.choice(n_total, size=TNG50_SIZE, replace=False)
    
    # Create subsampled dataframe
    subsampled_data = {key: values[sample_idx] for key, values in nsa_data.items()}
    data = np.column_stack([subsampled_data[var] for var in var_names])
    df = pd.DataFrame(data, columns=var_names)
    
    # Run FCIT
    try:
        search = ts.TetradSearch(df)
        search.set_verbose(False)
        search.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
        search.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
        search.run_fcit()
        
        graph_str = str(search.get_java())
        edges = extract_edges_from_graph(graph_str)
        density = compute_edge_density(edges, n_variables)
        subsample_densities.append(density)
        print(f"  Edge density: {density:.3f} ({len(edges)} edges)")
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Statistics
mean_nsa_density = np.mean(subsample_densities)
std_nsa_density = np.std(subsample_densities, ddof=1)
max_nsa_density = np.max(subsample_densities)
min_nsa_density = np.min(subsample_densities)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nNSA subsamples (N={len(subsample_densities)}):")
print(f"  Mean edge density: {mean_nsa_density:.3f} ± {std_nsa_density:.3f}")
print(f"  Range: [{min_nsa_density:.3f}, {max_nsa_density:.3f}]")
print(f"\nTNG50:")
print(f"  Edge density: {tng50_density:.3f}")
print(f"\nDifference: {tng50_density - mean_nsa_density:.3f}")
print(f"TNG50 is {tng50_density/max_nsa_density:.2f}x higher than max NSA subsample")

# Count how many NSA subsamples exceed TNG50 (should be 0)
n_above_tng50 = sum(1 for d in subsample_densities if d >= tng50_density)
print(f"\nNSA subsamples with density >= TNG50: {n_above_tng50}/{len(subsample_densities)}")

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(subsample_densities, bins=min(10, len(subsample_densities)), 
         alpha=0.7, color='#3498db', edgecolor='black', linewidth=1.2)
plt.axvline(tng50_density, color='#e74c3c', linewidth=2.5, linestyle='--', 
            label=f'TNG50 density ({tng50_density:.3f})')
plt.axvline(mean_nsa_density, color='#2ecc71', linewidth=2, linestyle=':', 
            label=f'NSA mean ({mean_nsa_density:.3f})')

plt.xlabel('Edge Density', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('NSA Subsample Size Test\n(Subsampled to match TNG50 size)', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=10, framealpha=0.95, edgecolor='black')
plt.grid(alpha=0.3, axis='y', linestyle=':')
plt.tight_layout()

output_path = os.path.join(PLOTS_DIR, "nsa_subsample_size_test.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")  # Publication quality 300 DPI
plt.close()

print(f"\nPlot saved to: {output_path}")

# Save results
results_df = pd.DataFrame({
    'subsample_id': range(1, len(subsample_densities) + 1),
    'edge_density': subsample_densities
})
results_df.loc[len(results_df)] = {'subsample_id': 'TNG50', 'edge_density': tng50_density}
results_df.loc[len(results_df)] = {'subsample_id': 'NSA_mean', 'edge_density': mean_nsa_density}

csv_path = os.path.join(RESULTS_DIR, "nsa_subsample_size_test.csv")
results_df.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")

