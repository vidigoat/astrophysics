"""
Compute graph metrics uncertainty from bootstrap runs.
Calculates edge density, directed fraction, and orientation fraction with confidence intervals.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pytetrad.tools import TetradSearch as ts
from scipy import stats

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "Results")
PLOTS_DIR = os.path.join(REPO_ROOT, "Plots", "RobustnessTests")
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    """Compute graph metrics from edge list."""
    total_edges = len(edges)
    max_possible = n_variables * (n_variables - 1) / 2
    edge_density = total_edges / max_possible if max_possible > 0 else 0
    
    directed_count = sum(1 for e in edges if "-->" in e)
    partial_count = sum(1 for e in edges if "o->" in e)
    undirected_count = sum(1 for e in edges if "o-o" in e)
    
    orientation_fraction = (directed_count + partial_count) / total_edges if total_edges > 0 else 0
    
    return {
        'edge_density': edge_density,
        'orientation_fraction': orientation_fraction,
        'total_edges': total_edges,
        'directed_edges': directed_count,
        'partial_edges': partial_count,
        'undirected_edges': undirected_count
    }

def run_bootstrap_metrics(dataset_name, data_path, n_bootstrap, trunc_limit, penalty_discount, 
                          sample_fraction=0.8, random_seed=42):
    """Run bootstrap and compute metrics for each run."""
    print(f"\nRunning {n_bootstrap} bootstraps for {dataset_name}...")
    
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    var_names = list(data_dict.keys())
    n_variables = len(var_names)
    full_data = np.column_stack([data_dict[var] for var in var_names])
    df_full = pd.DataFrame(full_data, columns=var_names)
    n_total = len(df_full)
    n_sample = int(n_total * sample_fraction)
    
    metrics_list = []
    np.random.seed(random_seed)
    
    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}...")
        
        sample_idx = np.random.choice(n_total, size=n_sample, replace=False)
        df_sample = df_full.iloc[sample_idx]
        
        try:
            search = ts.TetradSearch(df_sample)
            search.set_verbose(False)
            search.use_basis_function_lrt(truncation_limit=trunc_limit, alpha=0.01)
            search.use_basis_function_bic(truncation_limit=trunc_limit, penalty_discount=penalty_discount)
            search.run_fcit()
            
            graph_str = str(search.get_java())
            edges = extract_edges_from_graph(graph_str)
            metrics = compute_graph_metrics(edges, n_variables)
            metrics['bootstrap_id'] = i + 1
            metrics_list.append(metrics)
        except Exception as e:
            print(f"  Warning: Bootstrap {i+1} failed: {e}")
            continue
    
    return pd.DataFrame(metrics_list), n_variables

def compute_confidence_intervals(df, metric_name):
    """Compute mean and 95% CI for a metric."""
    values = df[metric_name].values
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    n = len(values)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean_val, scale=std_val/np.sqrt(n))
    return mean_val, ci_95[0], ci_95[1]

# Dataset configurations
DATASETS = [
    {
        'name': 'TNG50',
        'data_path': os.path.join(REPO_ROOT, 'Data', 'tng50_final.pkl'),
        'n_bootstrap': 50,
        'trunc_limit': 7,
        'penalty_discount': 15
    },
    {
        'name': 'NSA',
        'data_path': os.path.join(REPO_ROOT, 'Data', 'nsa_final_10props.pkl'),
        'n_bootstrap': 10,  # Already done, but we'll recompute metrics
        'trunc_limit': 14,
        'penalty_discount': 50
    },
    {
        'name': 'ALFALFA×NSA',
        'data_path': os.path.join(REPO_ROOT, 'Data', 'alfalfa_nsa_final_13props.pkl'),
        'n_bootstrap': 50,
        'trunc_limit': 7,
        'penalty_discount': 35
    }
]

all_results = []

for dataset_config in DATASETS:
    df_metrics, n_vars = run_bootstrap_metrics(
        dataset_name=dataset_config['name'],
        data_path=dataset_config['data_path'],
        n_bootstrap=dataset_config['n_bootstrap'],
        trunc_limit=dataset_config['trunc_limit'],
        penalty_discount=dataset_config['penalty_discount']
    )
    
    # Compute statistics
    mean_density, ci_low_density, ci_high_density = compute_confidence_intervals(df_metrics, 'edge_density')
    mean_orient, ci_low_orient, ci_high_orient = compute_confidence_intervals(df_metrics, 'orientation_fraction')
    
    ci_width_density = ci_high_density - ci_low_density
    ci_width_orient = ci_high_orient - ci_low_orient
    
    all_results.append({
        'Dataset': dataset_config['name'],
        'Edge_Density_Mean': mean_density,
        'Edge_Density_CI_Low': ci_low_density,
        'Edge_Density_CI_High': ci_high_density,
        'Edge_Density_CI_Width': ci_width_density,
        'Orientation_Fraction_Mean': mean_orient,
        'Orientation_Fraction_CI_Low': ci_low_orient,
        'Orientation_Fraction_CI_High': ci_high_orient,
        'N_Bootstrap': len(df_metrics),
        'N_Variables': n_vars
    })
    
    # Save individual bootstrap metrics
    output_path = os.path.join(RESULTS_DIR, f"{dataset_config['name'].lower().replace('×', '_').replace(' ', '_')}_bootstrap_metrics.csv")
    df_metrics.to_csv(output_path, index=False)
    print(f"  Saved metrics to {output_path}")

# Create summary table
summary_df = pd.DataFrame(all_results)
summary_path = os.path.join(RESULTS_DIR, "graph_metrics_uncertainty_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n" + "="*80)
print("GRAPH METRICS UNCERTAINTY SUMMARY")
print("="*80)
print("\nEdge Density (mean ± 95% CI):")
for _, row in summary_df.iterrows():
    ci_low = row['Edge_Density_CI_Low']
    ci_high = row['Edge_Density_CI_High']
    print(f"  {row['Dataset']:15s}: {row['Edge_Density_Mean']:.3f} ± {(ci_high-ci_low)/2:.3f}")

print("\nOrientation Fraction (mean ± 95% CI):")
for _, row in summary_df.iterrows():
    ci_low = row['Orientation_Fraction_CI_Low']
    ci_high = row['Orientation_Fraction_CI_High']
    print(f"  {row['Dataset']:15s}: {row['Orientation_Fraction_Mean']:.3f} ± {(ci_high-ci_low)/2:.3f}")

print(f"\nSummary saved to: {summary_path}")

