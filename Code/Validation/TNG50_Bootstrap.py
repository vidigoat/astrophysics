"""
Bootstrap validation for Illustris TNG50 simulation dataset.
Tests robustness of discovered causal edges across random subsamples.
"""

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from pytetrad.tools import TetradSearch as ts

graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if not graphviz_bin:
    graphviz_bin = r'C:\Users\sanji\Downloads\Graphviz-14.0.2-win64\bin'
if graphviz_bin and os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

PVAL_THRESHOLD = 0.01
TRUNCATION_LIMIT = 7  # Optimized hyperparameters
PENALTY_DISCOUNT = 15  # Optimized hyperparameters
N_BOOTSTRAP = 50
SAMPLE_FRACTION = 0.8
RANDOM_SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "tng50_final.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "tng50_bootstrap_validation.csv")

with open(DATA_PATH, "rb") as f:
    data_dict = pickle.load(f)

var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df_full = pd.DataFrame(full_data, columns=var_names)
n_total = len(df_full)
n_sample = int(n_total * SAMPLE_FRACTION)

def clean_token(token: str) -> str:
    token = token.strip()
    if "." in token:
        left, right = token.split(".", 1)
        if left.strip().isdigit():
            return right.strip()
    return token

edge_counter = Counter()
np.random.seed(RANDOM_SEED)

for i in range(N_BOOTSTRAP):
    sample_idx = np.random.choice(n_total, size=n_sample, replace=False)
    df_sample = df_full.iloc[sample_idx]

    search = ts.TetradSearch(df_sample)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=TRUNCATION_LIMIT, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(truncation_limit=TRUNCATION_LIMIT, penalty_discount=PENALTY_DISCOUNT)
    search.run_fcit()

    graph_str = str(search.get_java())
    lines = [l.strip() for l in graph_str.split("\n") if l.strip() and not l.startswith("Graph")]

    for line in lines:
        if "-->" in line:
            parts = line.split("-->")
            edge = f"{clean_token(parts[0])} --> {clean_token(parts[1])}"
            edge_counter[edge] += 1
        elif "o->" in line:
            parts = line.split("o->")
            edge = f"{clean_token(parts[0])} o-> {clean_token(parts[1])}"
            edge_counter[edge] += 1
        elif "o-o" in line:
            parts = line.split("o-o")
            nodes = sorted([clean_token(parts[0]), clean_token(parts[1])])
            edge = f"{nodes[0]} o-o {nodes[1]}"
            edge_counter[edge] += 1

sorted_edges = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
results_df = pd.DataFrame(
    [{"edge": edge, "count": count, "percentage": count / N_BOOTSTRAP * 100} for edge, count in sorted_edges]
)
results_df.to_csv(RESULTS_PATH, index=False)

