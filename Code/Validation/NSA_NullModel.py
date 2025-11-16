"""Null model validation for NSA-only dataset (cleaned)."""
import os
import time
import pickle

import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

# Add Graphviz to PATH if GRAPHVIZ_BIN environment variable is set
graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

PVAL_THRESHOLD = 0.01
TRUNC_LIMIT = 14
PENALTY_DISCOUNT = 50
N_NULL_RUNS = 5
RANDOM_SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "nsa_null_model_test.pkl")

print("=" * 70)
print("NSA NULL MODEL TEST")
print("=" * 70)

with open(DATA_PATH, "rb") as fp:
    data_dict = pickle.load(fp)

var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df_real = pd.DataFrame(full_data, columns=var_names)

n_total = len(df_real)
print(f"\nReal dataset: {n_total:,} galaxies, {len(var_names)} properties\n")

np.random.seed(RANDOM_SEED)

print("=" * 70)
print("CONTROL: Running FCIT on REAL data")
print("=" * 70)

search_real = ts.TetradSearch(df_real)
search_real.set_verbose(False)
search_real.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
search_real.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
search_real.run_fcit()

graph_real = str(search_real.get_java())
lines_real = [line for line in graph_real.split("\n") if line.strip() and not line.startswith("Graph")]
n_edges_real = len([l for l in lines_real if any(a in l for a in ["-->", "<--", "o-o", "o->"])])

print(f"\nReal data edges: {n_edges_real}")

print("\n" + "=" * 70)
print(f"NULL MODEL: Running FCIT on SHUFFLED data ({N_NULL_RUNS} runs)")
print("=" * 70)
print("\nEach run shuffles columns independently to destroy causal structure.\n")

null_edge_counts = []

for i in range(N_NULL_RUNS):
    print(f"[{i + 1}/{N_NULL_RUNS}] Shuffling and running FCIT...", end=" ", flush=True)

    df_null = df_real.copy()
    for col in df_null.columns:
        df_null[col] = np.random.permutation(df_null[col].values)

    start = time.time()

    search_null = ts.TetradSearch(df_null)
    search_null.set_verbose(False)
    search_null.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
    search_null.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
    search_null.run_fcit()

    graph_null = str(search_null.get_java())
    lines_null = [line for line in graph_null.split("\n") if line.strip() and not line.startswith("Graph")]
    n_edges_null = len([l for l in lines_null if any(a in l for a in ["-->", "<--", "o-o", "o->"])])

    runtime = time.time() - start
    null_edge_counts.append(n_edges_null)

    print(f"{n_edges_null} edges, {runtime:.2f}s")

print("\n" + "=" * 70)
print("NSA NULL MODEL RESULTS")
print("=" * 70)

mean_null = np.mean(null_edge_counts)
std_null = np.std(null_edge_counts)

print(f"\nReal data:      {n_edges_real} edges")
print(f"Null model:     {mean_null:.2f} +/- {std_null:.2f} edges")
print(f"Difference:     {n_edges_real - mean_null:.1f} edges")

if std_null > 0:
    z_score = (n_edges_real - mean_null) / std_null
    print(f"Z-score          : {z_score:.2f}")
else:
    print("Z-score          : undefined (null edges constant)")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "wb") as fp_pickle:
    pickle.dump(
        {
            "real_edges": n_edges_real,
            "null_counts": null_edge_counts,
            "null_mean": mean_null,
            "null_std": std_null,
            "runs": N_NULL_RUNS,
        },
        fp_pickle,
    )

print(f"\nSaved: {RESULTS_PATH}")
print("\nNSA null model validation complete.\n")
