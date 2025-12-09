"""
Comprehensive comparison of FCIT, FCI, and PC algorithms across all datasets.
Implements 4-step analysis:
1. Collect outputs from all algorithms
2. Compute graph metrics (edge density, orientation fraction, stable edges, V-structures)
3. Compare datasets using metrics
4. Compute cross-algorithm agreement (Jaccard similarity)
"""

import os
import re
import numpy as np
import pandas as pd

# Setup paths
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    REPO_ROOT = os.getcwd()
    if not os.path.exists(os.path.join(REPO_ROOT, "Code", "Analysis")):
        for _ in range(3):
            REPO_ROOT = os.path.dirname(REPO_ROOT)
            if os.path.exists(os.path.join(REPO_ROOT, "Code", "Analysis")):
                break

RESULTS_DIR = os.path.join(REPO_ROOT, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configurations
DATASETS = {
    'TNG50': {
        'var_names': ['DM_MASS', 'STELLAR_MASS', 'GAS_MASS', 'BH_MASS', 'BARYONIC_MASS',
                      'HALFMASS_RAD', 'VEL_DISP', 'VMAX', 'GAS_METALLICITY', 'STAR_METALLICITY',
                      'PHOTOMETRIC_U', 'PHOTOMETRIC_R', 'SFR', 'COLOUR'],
        'fcit_file': 'tng50_fcit_t7_p15.txt',
        'pc_file': 'tng50_pc.txt',
        'fci_file': 'tng50_fci.txt',
        'bootstrap_file': 'tng50_bootstrap_validation.csv'
    },
    'NSA': {
        'var_names': ['COLOR_U_R', 'ELPETRO_B300', 'SERSIC_N', 'ELPETRO_METS', 'ELPETRO_MTOL',
                      'ELPETRO_BA', 'ELPETRO_TH50_R', 'ZDIST', 'ELPETRO_MASS', 'ELPETRO_ABSMAG_R',
                      'BARYONIC_MASS'],
        'fcit_file': 'nsa_fcit_t14_p50.txt',
        'pc_file': 'nsa_pc.txt',
        'fci_file': 'nsa_fci.txt',
        'bootstrap_file': 'nsa_bootstrap_validation.csv'
    },
    'ALFALFA×NSA': {
        'var_names': ['BARYONIC_MASS', 'COLOR_U_R', 'ELPETRO_B300', 'SERSIC_N', 'ELPETRO_METS',
                      'ELPETRO_MTOL', 'ELPETRO_BA', 'ELPETRO_TH50_R', 'ZDIST', 'logMH',
                      'ELPETRO_MASS', 'ELPETRO_ABSMAG_R', 'W50'],
        'fcit_file': 'alfalfa_nsa_fcit_t7_p35.txt',
        'pc_file': 'alfalfa_nsa_pc.txt',
        'fci_file': 'alfalfa_nsa_fci.txt',
        'bootstrap_file': 'bootstrap_validation.csv'
    }
}

def parse_fcit_edges(txt_path):
    """Parse edges from FCIT result file."""
    if not os.path.exists(txt_path):
        return []
    
    edges = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "Graph Edges:" in content:
        edges_section = content.split("Graph Edges:")[1]
        for line in edges_section.split('\n'):
            line = line.strip()
            if not line or line.startswith('='):
                continue
            line = re.sub(r'^\d+\.\s*', '', line)
            if any(arrow in line for arrow in ['-->', 'o-o', 'o->', '<->']):
                edges.append(line.strip())
    
    return edges

def edges_to_skeleton(edges):
    """Convert edges to undirected skeleton (set of tuples)."""
    skeleton = set()
    for edge in edges:
        if '-->' in edge:
            parts = edge.split('-->')
            u, v = parts[0].strip(), parts[1].strip()
            skeleton.add(tuple(sorted([u, v])))
        elif 'o->' in edge:
            parts = edge.split('o->')
            u, v = parts[0].strip(), parts[1].strip()
            skeleton.add(tuple(sorted([u, v])))
        elif 'o-o' in edge:
            parts = edge.split('o-o')
            u, v = parts[0].strip(), parts[1].strip()
            skeleton.add(tuple(sorted([u, v])))
        elif '<->' in edge:
            parts = edge.split('<->')
            u, v = parts[0].strip(), parts[1].strip()
            skeleton.add(tuple(sorted([u, v])))
    return skeleton

def compute_edge_density(edges, n_vars):
    """Compute edge density."""
    skeleton = edges_to_skeleton(edges)
    max_possible = n_vars * (n_vars - 1) / 2
    return len(skeleton) / max_possible if max_possible > 0 else 0.0

def compute_orientation_fraction(edges):
    """Compute fraction of oriented edges (directed + partially directed)."""
    if not edges:
        return 0.0
    
    oriented = sum(1 for e in edges if '-->' in e or 'o->' in e)
    return oriented / len(edges)

def compute_v_structures(edges, var_names):
    """Count V-structures (colliders): A --> C <-- B."""
    v_count = 0
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    
    # Build adjacency list for directed edges
    directed_edges = {}
    for edge in edges:
        if '-->' in edge:
            parts = edge.split('-->')
            u, v = parts[0].strip(), parts[1].strip()
            if u in var_to_idx and v in var_to_idx:
                if v not in directed_edges:
                    directed_edges[v] = []
                directed_edges[v].append(u)
    
    # Count colliders (nodes with >= 2 incoming edges)
    for node, parents in directed_edges.items():
        if len(parents) >= 2:
            # Count pairs of parents
            v_count += len(parents) * (len(parents) - 1) // 2
    
    return v_count

def get_stable_edges(bootstrap_file, threshold=0.7):
    """Get edges that appear in >= threshold fraction of bootstraps."""
    if not os.path.exists(os.path.join(RESULTS_DIR, bootstrap_file)):
        return []
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, bootstrap_file))
    if 'percentage' in df.columns:
        stable = df[df['percentage'] >= threshold * 100]['edge'].tolist()
    elif 'count' in df.columns:
        # Need to know total bootstrap count - estimate from max count
        max_count = df['count'].max()
        if max_count > 0:
            total_bootstrap = int(max_count / threshold) if max_count < 100 else 100
            stable = df[df['count'] >= threshold * total_bootstrap]['edge'].tolist()
        else:
            stable = []
    else:
        stable = []
    
    return stable

def compute_jaccard_similarity(edges1, edges2):
    """Compute Jaccard similarity between two edge sets (skeleton-based)."""
    skeleton1 = edges_to_skeleton(edges1)
    skeleton2 = edges_to_skeleton(edges2)
    
    intersection = len(skeleton1 & skeleton2)
    union = len(skeleton1 | skeleton2)
    
    return intersection / union if union > 0 else 0.0

def main():
    """Main comparison function."""
    print("="*70)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("FCIT vs FCI vs PC")
    print("="*70)
    
    all_results = []
    algorithm_edges = {}  # {dataset: {algorithm: edges}}
    
    # STEP 1: Collect outputs from all algorithms
    print("\n" + "="*70)
    print("STEP 1: Collecting algorithm outputs")
    print("="*70)
    
    for dataset_name, config in DATASETS.items():
        print(f"\nDataset: {dataset_name}")
        algorithm_edges[dataset_name] = {}
        
        # Load FCIT edges
        fcit_path = os.path.join(RESULTS_DIR, config['fcit_file'])
        fcit_edges = parse_fcit_edges(fcit_path)
        algorithm_edges[dataset_name]['FCIT'] = fcit_edges
        print(f"  FCIT: {len(fcit_edges)} edges")
        
        # Load PC edges
        pc_path = os.path.join(RESULTS_DIR, config['pc_file'])
        pc_edges = parse_fcit_edges(pc_path)
        algorithm_edges[dataset_name]['PC'] = pc_edges
        print(f"  PC: {len(pc_edges)} edges")
        
        # Load FCI edges
        fci_path = os.path.join(RESULTS_DIR, config['fci_file'])
        fci_edges = parse_fcit_edges(fci_path)
        algorithm_edges[dataset_name]['FCI'] = fci_edges
        print(f"  FCI: {len(fci_edges)} edges")
    
    # STEP 2: Compute graph metrics
    print("\n" + "="*70)
    print("STEP 2: Computing graph metrics")
    print("="*70)
    
    metrics_results = []
    
    for dataset_name, config in DATASETS.items():
        n_vars = len(config['var_names'])
        
        for algo_name in ['FCIT', 'FCI', 'PC']:
            edges = algorithm_edges[dataset_name].get(algo_name, [])
            
            edge_density = compute_edge_density(edges, n_vars)
            orientation_frac = compute_orientation_fraction(edges) if algo_name != 'PC' else None
            v_structures = compute_v_structures(edges, config['var_names'])
            
            # Bootstrap validation only available for FCIT (only algorithm that was bootstrapped)
            if algo_name == 'FCIT':
                stable_edges = get_stable_edges(config['bootstrap_file'], threshold=0.8)
                n_stable = len([e for e in stable_edges if e in edges]) if stable_edges else None
            else:
                n_stable = None  # No bootstrap validation run for PC/FCI
            
            metrics_results.append({
                'dataset': dataset_name,
                'algorithm': algo_name,
                'n_variables': n_vars,
                'n_edges': len(edges),
                'edge_density': edge_density,
                'orientation_fraction': orientation_frac,
                'n_v_structures': v_structures,
                'n_stable_edges': n_stable
            })
            
            print(f"\n{dataset_name} - {algo_name}:")
            print(f"  Edges: {len(edges)}")
            print(f"  Edge Density: {edge_density:.4f}")
            if orientation_frac is not None:
                print(f"  Orientation Fraction: {orientation_frac:.4f}")
            print(f"  V-structures: {v_structures}")
            if n_stable is not None:
                print(f"  Stable edges (>=80%): {n_stable}")
    
    # Save metrics
    df_metrics = pd.DataFrame(metrics_results)
    metrics_path = os.path.join(RESULTS_DIR, 'algorithm_comparison_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # STEP 3: Dataset comparison (check TNG50 > NSA > ALFALFA×NSA)
    print("\n" + "="*70)
    print("STEP 3: Dataset comparison (TNG50 > NSA > ALFALFA×NSA)")
    print("="*70)
    
    comparison_results = []
    
    for algo_name in ['FCIT', 'FCI', 'PC']:
        tng50_metric = df_metrics[(df_metrics['dataset'] == 'TNG50') & (df_metrics['algorithm'] == algo_name)]['edge_density'].values[0]
        nsa_metric = df_metrics[(df_metrics['dataset'] == 'NSA') & (df_metrics['algorithm'] == algo_name)]['edge_density'].values[0]
        alfalfa_metric = df_metrics[(df_metrics['dataset'] == 'ALFALFA×NSA') & (df_metrics['algorithm'] == algo_name)]['edge_density'].values[0]
        
        ranking_correct = (tng50_metric > nsa_metric > alfalfa_metric)
        
        comparison_results.append({
            'algorithm': algo_name,
            'TNG50_density': tng50_metric,
            'NSA_density': nsa_metric,
            'ALFALFA×NSA_density': alfalfa_metric,
            'ranking_correct': ranking_correct
        })
        
        print(f"\n{algo_name}:")
        print(f"  TNG50: {tng50_metric:.4f}")
        print(f"  NSA: {nsa_metric:.4f}")
        print(f"  ALFALFA×NSA: {alfalfa_metric:.4f}")
        print(f"  Ranking correct: {'YES' if ranking_correct else 'NO'}")
    
    df_comparison = pd.DataFrame(comparison_results)
    comparison_path = os.path.join(RESULTS_DIR, 'dataset_comparison.csv')
    df_comparison.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # STEP 4: Cross-algorithm agreement (Jaccard similarity)
    print("\n" + "="*70)
    print("STEP 4: Cross-algorithm agreement (Jaccard similarity)")
    print("="*70)
    
    jaccard_results = []
    
    for dataset_name in DATASETS.keys():
        fcit_edges = algorithm_edges[dataset_name]['FCIT']
        fci_edges = algorithm_edges[dataset_name]['FCI']
        pc_edges = algorithm_edges[dataset_name]['PC']
        
        j_fcit_fci = compute_jaccard_similarity(fcit_edges, fci_edges)
        j_fcit_pc = compute_jaccard_similarity(fcit_edges, pc_edges)
        j_fci_pc = compute_jaccard_similarity(fci_edges, pc_edges)
        
        jaccard_results.append({
            'dataset': dataset_name,
            'FCIT_FCI': j_fcit_fci,
            'FCIT_PC': j_fcit_pc,
            'FCI_PC': j_fci_pc
        })
        
        print(f"\n{dataset_name}:")
        print(f"  FCIT <-> FCI: {j_fcit_fci:.4f}")
        print(f"  FCIT <-> PC: {j_fcit_pc:.4f}")
        print(f"  FCI <-> PC: {j_fci_pc:.4f}")
    
    df_jaccard = pd.DataFrame(jaccard_results)
    jaccard_path = os.path.join(RESULTS_DIR, 'algorithm_jaccard_similarity.csv')
    df_jaccard.to_csv(jaccard_path, index=False)
    print(f"\nJaccard similarities saved to: {jaccard_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  1. {metrics_path}")
    print(f"  2. {comparison_path}")
    print(f"  3. {jaccard_path}")

if __name__ == '__main__':
    main()

