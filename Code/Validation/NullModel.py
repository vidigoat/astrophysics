"""Null model validation for ALFALFA × NSA dataset (cleaned)."""
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
N_NULL_RUNS = 10
RANDOM_SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "alfalfa_nsa_final_13props.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "null_model_test.pkl")

print("=" * 70)
print("STEP 3: NULL MODEL TEST (ALFALFA × NSA)")
print("=" * 70)

with open(DATA_PATH, "rb") as f:
    data_dict = pickle.load(f)

var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df_real = pd.DataFrame(full_data, columns=var_names)

print(f"\nReal dataset: {len(df_real):,} galaxies, {len(var_names)} properties")

print("\n" + "=" * 70)
print("CONTROL: Running FCIT on REAL data")
print("=" * 70)

search_real = ts.TetradSearch(df_real)
search_real.set_verbose(False)
search_real.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
search_real.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
search_real.run_fcit()

graph_real = str(search_real.get_java())
lines_real = [l for l in graph_real.split("\n") if l.strip() and not l.startswith("Graph")]
n_edges_real = len([l for l in lines_real if any(a in l for a in ["-->", "<--", "o-o", "o->"])])

print(f"\nReal data edges: {n_edges_real}")

print("\n" + "=" * 70)
print(f"NULL MODEL: Running FCIT on SHUFFLED data ({N_NULL_RUNS} runs)")
print("=" * 70)
print("\nShuffling destroys causal relationships; surviving edges indicate artefacts.\n")

null_edge_counts = []
np.random.seed(RANDOM_SEED)

for i in range(N_NULL_RUNS):
    print(f"[{i + 1}/{N_NULL_RUNS}] Shuffling and running FCIT...", end=" ")

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
    lines_null = [l for l in graph_null.split("\n") if l.strip() and not l.startswith("Graph")]
    n_edges_null = len([l for l in lines_null if any(a in l for a in ["-->", "<--", "o-o", "o->"])])

    null_edge_counts.append(n_edges_null)
    runtime = time.time() - start

    print(f"{n_edges_null} edges, {runtime:.2f}s")

print("\n" + "=" * 70)
print("NULL MODEL RESULTS")
print("=" * 70)

mean_null = np.mean(null_edge_counts)
std_null = np.std(null_edge_counts)

print(f"\nReal data:      {n_edges_real} edges")
print(f"Null model:     {mean_null:.2f} +/- {std_null:.2f} edges")
print(f"Difference:     {n_edges_real - mean_null:.1f} edges")

if std_null > 0:
    z_score = (n_edges_real - mean_null) / std_null
    print(f"Z-score:        {z_score:.2f}")
else:
    print("Z-score:        undefined (null edges constant)")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
results = {
    "real_edges": n_edges_real,
    "null_mean": mean_null,
    "null_std": std_null,
    "null_counts": null_edge_counts,
}

with open(RESULTS_PATH, "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved: {RESULTS_PATH}")
print("\nNull model validation complete.")