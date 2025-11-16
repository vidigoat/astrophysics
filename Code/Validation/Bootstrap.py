"""Bootstrap validation for ALFALFA × NSA dataset (cleaned)."""
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
N_BOOTSTRAP = 50
SAMPLE_FRACTION = 0.8
RANDOM_SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "Data", "alfalfa_nsa_final_13props.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "Results", "bootstrap_validation.csv")

print("=" * 70)
print("BOOTSTRAP VALIDATION (ALFALFA × NSA)")
print("=" * 70)

with open(DATA_PATH, "rb") as f:
    data_dict = pickle.load(f)

var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df_full = pd.DataFrame(full_data, columns=var_names)
n_total = len(df_full)

print(f"\nDataset: {n_total:,} galaxies, {len(var_names)} properties")

n_sample = int(n_total * SAMPLE_FRACTION)

print(f"Bootstrap: {N_BOOTSTRAP} runs, {n_sample:,} galaxies each")
print(f"Parameters: trunc={TRUNC_LIMIT}, penalty={PENALTY_DISCOUNT}, alpha={PVAL_THRESHOLD}\n")

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
    sample_idx = np.random.choice(n_total, size=n_sample, replace=False)
    df_sample = df_full.iloc[sample_idx]

    print(f"[{i + 1}/{N_BOOTSTRAP}]", end=" ", flush=True)

    search = ts.TetradSearch(df_sample)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(truncation_limit=TRUNC_LIMIT, penalty_discount=PENALTY_DISCOUNT)
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

    if (i + 1) % 10 == 0:
        print()

total_time = time.time() - start_time

print(f"\n\n{'=' * 70}")
print("BOOTSTRAP RESULTS")
print(f"{'=' * 70}\n")
print(f"Total time: {total_time / 60:.1f} minutes\n")

sorted_edges = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)

print(f"{'Edge':<55} {'Count':<10} {'%':<8} Confidence")
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

    print(f"{edge:<55} {count:<10} {pct:>5.1f}%  {conf}")

    if pct >= 80:
        robust_edges.append((edge, pct))

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
results_df = pd.DataFrame(
    [{"edge": edge, "count": count, "percentage": count / N_BOOTSTRAP * 100} for edge, count in sorted_edges]
)
results_df.to_csv(RESULTS_PATH, index=False)

print(f"\n{'=' * 70}")
print("(>=80% occurrence)")
print(f"{'=' * 70}\n")
for edge, pct in robust_edges:
    print(f"  - {edge} ({pct:.1f}%)")

print(f"\nSaved: {RESULTS_PATH}")
print(f"\nBootstrap validation complete ({len(robust_edges)} edges >= 80%).")
