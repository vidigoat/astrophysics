"""Bootstrap validation for NSA-only dataset (cleaned)."""
import os
import time
import pickle
from collections import Counter

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
N_BOOTSTRAP = 10
SAMPLE_FRACTION = 0.8
RANDOM_SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "nsa_bootstrap_validation.csv")

print("=" * 70)
print("NSA BOOTSTRAP VALIDATION")
print("=" * 70)

with open(DATA_PATH, "rb") as fp:
    data_dict = pickle.load(fp)

var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df_full = pd.DataFrame(full_data, columns=var_names)
n_total = len(df_full)

print(f"\nDataset: {n_total:,} galaxies, {len(var_names)} properties")

sample_size = int(n_total * SAMPLE_FRACTION)

print(f"Bootstrap runs: {N_BOOTSTRAP}")
print(f"Sample size per run: {sample_size:,} galaxies (80% without replacement)\n")

def clean_token(token: str) -> str:
    token = token.strip()
    if "." in token:
        left, right = token.split(".", 1)
        if left.strip().isdigit():
            return right.strip()
    return token

edge_counter = Counter()
np.random.seed(RANDOM_SEED)
start_time = time.time()

for i in range(N_BOOTSTRAP):
    sample_idx = np.random.choice(n_total, size=sample_size, replace=False)
    df_sample = df_full.iloc[sample_idx]

    print(f"[{i + 1}/{N_BOOTSTRAP}]", end=" ", flush=True)

    search = ts.TetradSearch(df_sample)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
    search.run_fcit()

    graph_str = str(search.get_java())
    lines = [line.strip() for line in graph_str.split("\n") if line.strip() and not line.startswith("Graph")]

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

    if (i + 1) % 5 == 0:
        print()

elapsed = time.time() - start_time

print(f"\n\n{'=' * 70}")
print("NSA BOOTSTRAP RESULTS")
print(f"{'=' * 70}\n")
print(f"Total time: {elapsed / 60:.1f} minutes\n")

sorted_edges = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)

print(f"{'Edge':<55} {'Count':<7} {'%':<7} Confidence")
print("-" * 90)

robust_edges = []
for edge, count in sorted_edges:
    pct = (count / N_BOOTSTRAP) * 100
    if pct >= 95:
        conf = "VERY STRONG"
    elif pct >= 80:
        conf = "STRONG"
    elif pct >= 50:
        conf = "MODERATE"
    else:
        conf = "WEAK"

    print(f"{edge:<55} {count:<7} {pct:>5.1f}%  {conf}")
    if pct >= 80:
        robust_edges.append((edge, pct))

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
results_df = pd.DataFrame(
    [{"edge": edge, "count": count, "percentage": count / N_BOOTSTRAP * 100} for edge, count in sorted_edges]
)
results_df.to_csv(RESULTS_PATH, index=False)

print(f"\n{'=' * 70}")
print("BOOTSTRAP EDGES >= 80%")
print(f"{'=' * 70}")
for edge, pct in robust_edges:
    print(f"  - {edge} ({pct:.1f}%)")

print(f"\nSaved: {RESULTS_PATH}")
print("\nNSA bootstrap validation complete.\n")
