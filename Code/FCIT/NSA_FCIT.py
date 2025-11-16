import os
import time
import pickle
import numpy as np
import pandas as pd
from pytetrad.tools import TetradSearch as ts
import graphviz as gviz

# Add Graphviz to PATH if GRAPHVIZ_BIN environment variable is set
graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

PVAL_THRESHOLD = 0.01
TRUNCATION_LIMIT = 14
PENALTY_DISCOUNT = 50


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(repo_root, "Data", "nsa_final_10props.pkl")
    results_dir = os.path.join(repo_root, "Results")
    viz_dir = os.path.join(repo_root, "Plots", "CausalStructure")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("=" * 70)
    print("FCIT ANALYSIS – NSA DATASET (10 PROPERTIES)")
    print("=" * 70)
    print(f"Loading data from: {data_path}")

    with open(data_path, "rb") as fp:
        data_dict = pickle.load(fp)

    variables = list(data_dict.keys())
    data = np.column_stack([data_dict[var] for var in variables])
    df = pd.DataFrame(data, columns=variables)

    print(f"\nVariables ({len(variables)}):")
    for idx, name in enumerate(variables, start=1):
        print(f"  {idx:2d}. {name}")

    print(f"\nData shape: {df.shape}")
    print(f"Galaxies : {df.shape[0]:,}")

    search = ts.TetradSearch(df)
    search.set_verbose(True)
    search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(truncation_limit=TRUNCATION_LIMIT, penalty_discount=PENALTY_DISCOUNT)

    print("\n" + "=" * 70)
    print("Running FCIT...")
    print("=" * 70)
    start = time.time()
    search.run_fcit()
    runtime = time.time() - start

    graph = search.get_java()
    graph_str = str(graph)

    print("\n" + "=" * 70)
    print("CAUSAL GRAPH:")
    print("=" * 70)
    print(graph_str)
    print(f"\nRuntime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")

    edge_lines = [line for line in graph_str.split("\n") if line.strip() and not line.startswith("Graph")]
    num_edges = sum(any(token in line for token in ["-->", "<--", "o-o", "o->", "<-o"]) for line in edge_lines)
    fully_directed = sum("-->" in line or "<--" in line for line in edge_lines)
    partially_directed = sum("o->" in line or "<-o" in line for line in edge_lines)
    undirected = sum("o-o" in line and "o->" not in line and "<-o" not in line for line in edge_lines)

    print("\nGraph statistics:")
    print(f"  Nodes: {len(variables)}")
    print(f"  Edges: {num_edges}")
    print(f"  Fully directed edges:      {fully_directed}")
    print(f"  Partially directed edges:  {partially_directed}")
    print(f"  Undirected edges:          {undirected}")

    base_name = f"nsa_fcit_t{TRUNCATION_LIMIT}_p{PENALTY_DISCOUNT}"
    txt_path = os.path.join(results_dir, f"{base_name}.txt")

    dot_graph = search.get_dot()
    graph_viz = gviz.Source(dot_graph)
    png_path = graph_viz.render(
        filename=base_name,
        directory=viz_dir,
        format="png",
        cleanup=True,
    )

    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("FCIT Causal Graph – NSA Dataset (10 properties)\n")
        fp.write("=" * 70 + "\n")
        fp.write(f"Galaxies: {df.shape[0]:,}\n")
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

    print("\nOutputs saved:")
    print(f"  Visualization PNG: {png_path}")
    print(f"  Graph details:     {txt_path}")


if __name__ == "__main__":
    main()

