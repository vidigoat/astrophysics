"""
Null model validation for ALFALFA × NSA dataset.
Tests whether discovered edges are genuine by shuffling data to destroy causal structure.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
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

with open(DATA_PATH, "rb") as f:
    data_dict = pickle.load(f)

var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df_real = pd.DataFrame(full_data, columns=var_names)

search_real = ts.TetradSearch(df_real)
search_real.set_verbose(False)
search_real.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
search_real.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
search_real.run_fcit()

graph_real = str(search_real.get_java())
lines_real = [l for l in graph_real.split("\n") if l.strip() and not l.startswith("Graph")]
n_edges_real = len([l for l in lines_real if any(a in l for a in ["-->", "<--", "o-o", "o->"])])

null_edge_counts = []
np.random.seed(RANDOM_SEED)

for i in range(N_NULL_RUNS):
    df_null = df_real.copy()
    for col in df_null.columns:
        df_null[col] = np.random.permutation(df_null[col].values)

    search_null = ts.TetradSearch(df_null)
    search_null.set_verbose(False)
    search_null.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
    search_null.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
    search_null.run_fcit()

    graph_null = str(search_null.get_java())
    lines_null = [l for l in graph_null.split("\n") if l.strip() and not l.startswith("Graph")]
    n_edges_null = len([l for l in lines_null if any(a in l for a in ["-->", "<--", "o-o", "o->"])])
    null_edge_counts.append(n_edges_null)

mean_null = np.mean(null_edge_counts)
std_null = np.std(null_edge_counts)

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
results = {
    "real_edges": n_edges_real,
    "null_mean": mean_null,
    "null_std": std_null,
    "null_counts": null_edge_counts,
}

with open(RESULTS_PATH, "wb") as f:
    pickle.dump(results, f)
