"""
FCIT causal discovery analysis for ALFALFA × NSA dataset.
Runs FCIT algorithm and saves causal graph visualization and results.
"""

import pandas as pd
import pytetrad.tools.TetradSearch as ts
import numpy as np
import time
import graphviz as gviz
import pickle
import os

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

PVAL_THRESHOLD = 0.01
TRUNC_LIM = 14
PENALTY_DISCOUNT = 50

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(repo_root, 'Data', 'alfalfa_nsa_final_13props.pkl')
results_dir = os.path.join(repo_root, 'Results')
viz_dir = os.path.join(repo_root, 'Plots', 'CausalStructure')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

var_names = list(data_dict.keys())
data = np.column_stack([data_dict[var] for var in var_names])
data = pd.DataFrame(data, columns=var_names)

start = time.time()
search = ts.TetradSearch(data)
search.set_verbose(False)
search.use_basis_function_lrt(truncation_limit=TRUNC_LIM, alpha=PVAL_THRESHOLD)
search.use_basis_function_bic(truncation_limit=TRUNC_LIM, penalty_discount=PENALTY_DISCOUNT)
search.run_fcit()
runtime = round(time.time() - start, 3)

graph = search.get_java()
graph_str = str(graph)
edge_lines = [line for line in graph_str.split('\n') if line.strip() and not line.startswith('Graph')]
num_edges = len([line for line in edge_lines if any(arrow in line for arrow in ['-->', '<--', 'o-o', 'o->'])])

gdot = search.get_dot()
graph_viz = gviz.Source(gdot)
base_name = f"fcit_final_13props_t{TRUNC_LIM}_p{PENALTY_DISCOUNT}"

png_path = graph_viz.render(
    filename=base_name,
    directory=viz_dir,
    format="png",
    cleanup=True,
)
if png_path and os.path.exists(png_path):
    final_png = os.path.join(viz_dir, f"{base_name}.png")
    if png_path != final_png:
        import shutil
        shutil.move(png_path, final_png)

graph_text_file = os.path.join(results_dir, f"fcit_final_13props_t{TRUNC_LIM}_p{PENALTY_DISCOUNT}.txt")
with open(graph_text_file, 'w', encoding='utf-8') as f:
    f.write("FCIT Causal Graph - Final 13 Properties\n")
    f.write("="*70 + "\n")
    f.write(f"Dataset: ALFALFA x NSA\n")
    f.write(f"Galaxies: {data.shape[0]}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  Truncation limit:  {TRUNC_LIM}\n")
    f.write(f"  Penalty discount:  {PENALTY_DISCOUNT}\n")
    f.write(f"  p-value threshold: {PVAL_THRESHOLD}\n")
    f.write(f"  Runtime:           {runtime} seconds\n\n")
    f.write(f"Properties:\n")
    for i, var in enumerate(var_names, 1):
        f.write(f"  {i:2d}. {var}\n")
    f.write("\n" + "="*70 + "\n\n")
    f.write(str(graph))
