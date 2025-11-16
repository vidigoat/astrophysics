import pandas as pd
import pytetrad.tools.TetradSearch as ts
import numpy as np
import time
import graphviz as gviz
import pickle
import os

# Add Graphviz to PATH if GRAPHVIZ_BIN environment variable is set
graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if graphviz_bin:
    os.environ["PATH"] += os.pathsep + graphviz_bin

# FCIT parameters (from paper recommendations)
pval_threshold = 0.01
trunc_lim = 14
penalty_discount = 50

print("="*70)
print("FCIT CAUSAL ANALYSIS - FINAL 13 PROPERTIES")
print("="*70)
print(f"Parameters (from paper):")
print(f"  p-value threshold:   {pval_threshold}")
print(f"  Truncation limit:    {trunc_lim}")
print(f"  Penalty discount:    {penalty_discount}")
print("="*70)

# Resolve repository paths
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(repo_root, 'Data', 'alfalfa_nsa_final_13props.pkl')
results_dir = os.path.join(repo_root, 'Results')
viz_dir = os.path.join(repo_root, 'Plots', 'CausalStructure')

# Ensure output directories exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

# Load the data
filename = data_path
print(f"\nLoading: {filename}")

with open(filename, 'rb') as f:
    data_dict = pickle.load(f)

# Extract variable names and data
var_names = list(data_dict.keys())
data = np.column_stack([data_dict[var] for var in var_names])

print(f"\nVariables ({len(var_names)}):")
for i, var in enumerate(var_names, 1):
    print(f"  {i:2d}. {var}")

print(f"\nData shape: {data.shape}")
print(f"Total galaxies: {data.shape[0]}")

# Convert to DataFrame
data = pd.DataFrame(data, columns=var_names)

# Run FCIT
print("\n" + "="*70)
print("Starting FCIT algorithm...")
print("="*70)
start = time.time()

search = ts.TetradSearch(data)
search.set_verbose(True)

# Configure FCIT (using basis function methods)
search.use_basis_function_lrt(truncation_limit=trunc_lim, alpha=pval_threshold)
search.use_basis_function_bic(truncation_limit=trunc_lim, penalty_discount=penalty_discount)

# Run FCIT
search.run_fcit()

graph = search.get_java()
print("\n" + "="*70)
print("CAUSAL GRAPH:")
print("="*70)
print(graph)

end = time.time()
runtime = round(end - start, 3)
print("\n" + "="*70)
print(f"FCIT completed in {runtime} seconds ({runtime/60:.2f} minutes)")
print("="*70)

# Count edges
graph_str = str(graph)
edge_lines = [line for line in graph_str.split('\n') if line.strip() and not line.startswith('Graph')]
num_edges = len([line for line in edge_lines if any(arrow in line for arrow in ['-->', '<--', 'o-o', 'o->'])])

print(f"\nGraph Statistics:")
print(f"  Nodes: {len(var_names)}")
print(f"  Edges: {num_edges}")

# Analyze edge directions
directed = len([line for line in edge_lines if '-->' in line or '<--' in line])
partially = len([line for line in edge_lines if 'o->' in line or '<-o' in line])
undirected = len([line for line in edge_lines if 'o-o' in line and 'o->' not in line])

print(f"\nEdge Types:")
print(f"  Fully directed:      {directed}")
print(f"  Partially directed:  {partially}")
print(f"  Undirected:          {undirected}")

# Save graph visualization
print("\n" + "="*70)
print("Generating graph visualizations...")
print("="*70)

gdot = search.get_dot()
graph_viz = gviz.Source(gdot)

base_name = f"fcit_final_13props_t{trunc_lim}_p{penalty_discount}"

# Save as PNG
png_path = graph_viz.render(
    filename=base_name,
    directory=viz_dir,
    format="png",
    cleanup=True,
)

print(f"PNG saved: {png_path}")

# Save graph structure as text
graph_text_file = os.path.join(results_dir, f"fcit_final_13props_t{trunc_lim}_p{penalty_discount}.txt")
with open(graph_text_file, 'w', encoding='utf-8') as f:
    f.write("FCIT Causal Graph - Final 13 Properties\n")
    f.write("="*70 + "\n")
    f.write(f"Dataset: ALFALFA x NSA\n")
    f.write(f"Galaxies: {data.shape[0]}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  Truncation limit:  {trunc_lim}\n")
    f.write(f"  Penalty discount:  {penalty_discount}\n")
    f.write(f"  p-value threshold: {pval_threshold}\n")
    f.write(f"  Runtime:           {runtime} seconds\n\n")
    f.write(f"Properties:\n")
    for i, var in enumerate(var_names, 1):
        f.write(f"  {i:2d}. {var}\n")
    f.write("\n" + "="*70 + "\n\n")
    f.write(str(graph))

print(f"Graph text saved: {graph_text_file}")

print("\n" + "="*70)
print("Outputs")
print("="*70)
print(f"Results saved in:")
print(f"  - Visualization PNG: {png_path}")
print(f"  - Graph details: {graph_text_file}")