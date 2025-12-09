"""
Run PC and FCI algorithms on all three datasets (TNG50, NSA, ALFALFA×NSA).
This script standardizes alpha=0.01 across all datasets for consistent comparison.
"""
import os
import pickle
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "Results")
VIZ_DIR = os.path.join(REPO_ROOT, "Plots", "CausalStructure")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Setup graphviz path
graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

ALPHA = 0.05  # Standardized alpha for all datasets

def extract_edges_from_pc_graph(cg, var_names):
    """Extract edges from PC CausalGraph object."""
    edges = []
    graph = cg.G.graph
    
    for i in range(len(var_names)):
        for j in range(i+1, len(var_names)):
            if graph[j, i] == 1 and graph[i, j] == -1:
                edges.append(f"{var_names[i]} --> {var_names[j]}")
            elif graph[i, j] == 1 and graph[j, i] == -1:
                edges.append(f"{var_names[j]} --> {var_names[i]}")
            elif graph[i, j] == graph[j, i] == -1:
                edges.append(f"{var_names[i]} o-o {var_names[j]}")
            elif graph[i, j] == graph[j, i] == 1:
                edges.append(f"{var_names[i]} <-> {var_names[j]}")
    
    return edges

def extract_edges_from_fci_pag(g, var_names):
    """Extract edges from FCI PAG."""
    edges = []
    graph = g.graph
    
    for i in range(len(var_names)):
        for j in range(i+1, len(var_names)):
            i_to_j = graph[j, i]
            j_to_i = graph[i, j]
            
            if i_to_j == -1 and j_to_i == 1:
                edges.append(f"{var_names[i]} --> {var_names[j]}")
            elif i_to_j == 1 and j_to_i == -1:
                edges.append(f"{var_names[j]} --> {var_names[i]}")
            elif i_to_j == -1 and j_to_i == -1:
                edges.append(f"{var_names[i]} o-o {var_names[j]}")
            elif i_to_j == 2 and j_to_i == 1:
                edges.append(f"{var_names[i]} o-> {var_names[j]}")
            elif i_to_j == 1 and j_to_i == 2:
                edges.append(f"{var_names[j]} o-> {var_names[i]}")
            elif i_to_j == 1 and j_to_i == 1:
                edges.append(f"{var_names[i]} <-> {var_names[j]}")
            elif i_to_j == 2 and j_to_i == 2:
                edges.append(f"{var_names[i]} o-o {var_names[j]}")
    
    return edges

def save_graph_results(dataset_name, algorithm_name, var_names, edges, output_dir, alpha):
    """Save graph results in FCIT-like format."""
    dataset_clean = dataset_name.lower().replace('×', '_').replace(' ', '_')
    txt_path = os.path.join(output_dir, f"{dataset_clean}_{algorithm_name.lower()}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{algorithm_name} Causal Graph – {dataset_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Variables: {len(var_names)}\n")
        f.write(f"Parameters:\n")
        f.write(f"  alpha = {alpha}\n")
        if dataset_name == 'NSA':
            f.write(f"  subsample_size = 20000\n")
        f.write("\nVariables:\n")
        for idx, var in enumerate(var_names, 1):
            f.write(f"  {idx:2d}. {var}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write("Graph Edges:\n\n")
        for idx, edge in enumerate(sorted(edges), 1):
            f.write(f"{idx}. {edge}\n")
    
    print(f"  Saved: {txt_path}")
    return txt_path

def save_graph_viz(graph_obj, dataset_name, algorithm_name, var_names):
    """Save graph visualization."""
    try:
        pyd = GraphUtils.to_pydot(graph_obj, labels=var_names)
        dataset_clean = dataset_name.lower().replace('×', '_').replace(' ', '_')
        png_path = os.path.join(VIZ_DIR, f"{dataset_clean}_{algorithm_name.lower()}.png")
        pyd.write_png(png_path)
        print(f"  Graph saved: {png_path}")
        return True
    except Exception as e:
        print(f"  Could not save graph: {str(e)[:60]}")
        return False

def run_algorithms_on_dataset(dataset_name, data_path, var_names):
    """Run PC and FCI on a dataset."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Load data
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    data = np.column_stack([data_dict[var] for var in var_names])
    n_samples, n_vars = data.shape
    print(f"Data shape: {n_samples:,} samples × {n_vars} variables")
    
    # NSA is too large - subsample to avoid memory issues
    if dataset_name == 'NSA' and n_samples > 20000:
        print(f"  Subsampling NSA from {n_samples:,} to 20,000 samples for PC/FCI...")
        np.random.seed(42)
        sample_idx = np.random.choice(n_samples, size=20000, replace=False)
        data = data[sample_idx]
        n_samples = len(data)
        print(f"  Using {n_samples:,} samples")
    
    # Run PC
    print(f"\nRunning PC algorithm (alpha={ALPHA})...")
    pc_start = time.time()
    try:
        # Try fisherz first, fallback to chisq if singular
        try:
            cg = pc(data, alpha=ALPHA, indep_test='fisherz', stable=True, verbose=False, show_progress=False)
        except Exception as e:
            if 'singular' in str(e).lower():
                print(f"  FisherZ failed, trying chisq...")
                cg = pc(data, alpha=ALPHA, indep_test='chisq', stable=True, verbose=False, show_progress=False)
            else:
                raise
        pc_edges = extract_edges_from_pc_graph(cg, var_names)
        save_graph_results(dataset_name, "PC", var_names, pc_edges, RESULTS_DIR, ALPHA)
        save_graph_viz(cg.G, dataset_name, "PC", var_names)
        pc_time = time.time() - pc_start
        print(f"  PC found {len(pc_edges)} edges (took {pc_time:.1f}s)")
    except Exception as e:
        print(f"  Error running PC: {str(e)[:80]}")
        pc_edges = []
    
    # Run FCI
    print(f"\nRunning FCI algorithm (alpha={ALPHA})...")
    fci_start = time.time()
    try:
        # Try fisherz first, fallback to chisq if singular
        try:
            g, fci_edges_list = fci(data, independence_test_method='fisherz', alpha=ALPHA, depth=-1, verbose=False, show_progress=False)
        except Exception as e:
            if 'singular' in str(e).lower():
                print(f"  FisherZ failed, trying chisq...")
                g, fci_edges_list = fci(data, independence_test_method='chisq', alpha=ALPHA, depth=-1, verbose=False, show_progress=False)
            else:
                raise
        fci_edges = extract_edges_from_fci_pag(g, var_names)
        save_graph_results(dataset_name, "FCI", var_names, fci_edges, RESULTS_DIR, ALPHA)
        save_graph_viz(g, dataset_name, "FCI", var_names)
        fci_time = time.time() - fci_start
        print(f"  FCI found {len(fci_edges)} edges (took {fci_time:.1f}s)")
    except Exception as e:
        print(f"  Error running FCI: {str(e)[:80]}")
        fci_edges = []
    
    return pc_edges, fci_edges

# All three datasets
DATASETS = [
    {
        'name': 'TNG50',
        'data_path': os.path.join(REPO_ROOT, 'Data', 'tng50_final.pkl'),
        'var_names': ['DM_MASS', 'STELLAR_MASS', 'GAS_MASS', 'BH_MASS', 'BARYONIC_MASS',
                      'HALFMASS_RAD', 'VEL_DISP', 'VMAX', 'GAS_METALLICITY', 'STAR_METALLICITY',
                      'PHOTOMETRIC_U', 'PHOTOMETRIC_R', 'SFR', 'COLOUR']
    },
    {
        'name': 'NSA',
        'data_path': os.path.join(REPO_ROOT, 'Data', 'nsa_final_10props.pkl'),
        'var_names': ['COLOR_U_R', 'ELPETRO_B300', 'SERSIC_N', 'ELPETRO_METS', 'ELPETRO_MTOL',
                      'ELPETRO_BA', 'ELPETRO_TH50_R', 'ZDIST', 'ELPETRO_MASS', 'ELPETRO_ABSMAG_R',
                      'BARYONIC_MASS']
    },
    {
        'name': 'ALFALFA×NSA',
        'data_path': os.path.join(REPO_ROOT, 'Data', 'alfalfa_nsa_final_13props.pkl'),
        'var_names': ['BARYONIC_MASS', 'COLOR_U_R', 'ELPETRO_B300', 'SERSIC_N', 'ELPETRO_METS',
                      'ELPETRO_MTOL', 'ELPETRO_BA', 'ELPETRO_TH50_R', 'ZDIST', 'logMH',
                      'ELPETRO_MASS', 'ELPETRO_ABSMAG_R', 'W50']
    }
]

print("="*70)
print("Running PC and FCI on ALL datasets with alpha=0.01")
print("="*70)
print(f"Expected runtime: 1-2 hours (PC/FCI can be slow)")
print("="*70)

total_start = time.time()

# Run algorithms on each dataset
all_results = {}
for dataset_config in DATASETS:
    pc_edges, fci_edges = run_algorithms_on_dataset(
        dataset_config['name'],
        dataset_config['data_path'],
        dataset_config['var_names']
    )
    all_results[dataset_config['name']] = {
        'PC': pc_edges,
        'FCI': fci_edges
    }

total_time = time.time() - total_start

print("\n" + "="*70)
print("PC and FCI completed for all datasets!")
print("="*70)
print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print("\nSummary:")
for dataset_name, results in all_results.items():
    print(f"  {dataset_name}:")
    print(f"    PC: {len(results['PC'])} edges")
    print(f"    FCI: {len(results['FCI'])} edges")
print("="*70)

