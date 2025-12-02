"""
FCIT causal discovery analysis for Illustris TNG50 simulation dataset.
Runs FCIT algorithm and saves causal graph visualization and results.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from pytetrad.tools import TetradSearch as ts
import graphviz as gviz

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

PVAL_THRESHOLD = 0.01
TRUNCATION_LIMIT = 7  # Optimized from mock data analysis
PENALTY_DISCOUNT = 15  # Optimized from mock data analysis

def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(repo_root, "Data", "tng50_final.pkl")
    results_dir = os.path.join(repo_root, "Results")
    viz_dir = os.path.join(repo_root, "Plots", "CausalStructure")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    with open(data_path, "rb") as fp:
        data_dict = pickle.load(fp)

    variables = list(data_dict.keys())
    data = np.column_stack([data_dict[var] for var in variables])
    df = pd.DataFrame(data, columns=variables)


    start = time.time()
    search = ts.TetradSearch(df)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(truncation_limit=TRUNCATION_LIMIT, penalty_discount=PENALTY_DISCOUNT)
    search.run_fcit()
    runtime = time.time() - start

    graph = search.get_java()
    graph_str = str(graph)
    edge_lines = [line for line in graph_str.split("\n") if line.strip() and not line.startswith("Graph")]

    base_name = f"tng50_fcit_t{TRUNCATION_LIMIT}_p{PENALTY_DISCOUNT}"
    txt_path = os.path.join(results_dir, f"{base_name}.txt")

    dot_graph = search.get_dot()
    graph_viz = gviz.Source(dot_graph)
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

    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("FCIT Causal Graph – Illustris TNG50 Simulation\n")
        fp.write("=" * 70 + "\n")
        fp.write(f"Subhalos: {df.shape[0]:,}\n")
        fp.write("Parameters:\n")
        fp.write(f"  truncation_limit = {TRUNCATION_LIMIT}\n")
        fp.write(f"  penalty_discount = {PENALTY_DISCOUNT}\n")
        fp.write(f"  alpha            = {PVAL_THRESHOLD}\n")
        fp.write(f"Runtime: {runtime:.2f} s ({runtime/60:.2f} min)\n\n")
        fp.write("Variables:\n")
        for idx, name in enumerate(variables, start=1):
            fp.write(f"  {idx:2d}. {name}\n")
        fp.write("\n" + "=" * 70 + "\n\n")
        fp.write(graph_str)


if __name__ == "__main__":
    main()

