"""
Hyperparameter optimisation for the NSA (10-property) sample.

Mock datasets with a known causal structure are generated with a Causal
Perceptron Network: a random 12-16 edge DAG over 10 nodes, where each variable
is a random multi-layer-perceptron function of its parents plus independent
Beta(2,5) noise, sampled at the NSA real sample size. FCIT is then refit over a
grid of truncation limits and penalty discounts, and the combination that
maximises the mean skeleton F1 across the mock datasets is selected.

Grid and sizes can be overridden through environment variables (NSA_SAMPLE,
NSA_NMOCK, NSA_TRUNC, NSA_PEN) for quick checks; the defaults run the full grid.

Output: Results/nsa_hyperparameter_tuning.csv
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import networkx as nx
import pytetrad.tools.TetradSearch as ts

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "Results")
os.makedirs(RESULTS, exist_ok=True)

ALPHA = 0.01
N_PROPERTIES = 10
MIN_EDGES, MAX_EDGES = 12, 16
N_HIDDEN_LAYERS, N_NEURONS = 4, 50

SAMPLE = int(os.environ.get("NSA_SAMPLE", "484539"))
N_MOCK = int(os.environ.get("NSA_NMOCK", "100"))
# A run may process only a contiguous slice of the mock datasets [MOCK_START,
# MOCK_END); this lets several workers cover disjoint slices of the same fixed
# set of mocks in parallel, each writing a partial result that is merged later.
MOCK_START = int(os.environ.get("NSA_MOCK_START", "0"))
MOCK_END = int(os.environ.get("NSA_MOCK_END", str(N_MOCK)))
TRUNCS = [int(x) for x in os.environ.get("NSA_TRUNC", "7,14").split(",")]
PENS = [int(x) for x in os.environ.get("NSA_PEN", "30,40,50,60").split(",")]
TAG = os.environ.get("NSA_TAG", "")
PART_CSV = os.path.join(
    RESULTS, f"nsa_tuning_part_{MOCK_START}_{MOCK_END}{('_' + TAG) if TAG else ''}.csv")

PROPERTY_NAMES = [
    "ZDIST", "ELPETRO_ABSMAG_R", "ELPETRO_B300", "ELPETRO_MASS", "SERSIC_N",
    "ELPETRO_BA", "ELPETRO_TH50_R", "COLOR_U_R", "ELPETRO_METS", "ELPETRO_MTOL",
]


def relu(x):
    return np.maximum(0, x)


def generate_mlp(n_inputs, n_outputs, seed):
    rng = np.random.RandomState(seed)
    layers = [rng.randn(n_inputs, N_NEURONS) * np.sqrt(2.0 / n_inputs)]
    for _ in range(N_HIDDEN_LAYERS - 1):
        layers.append(rng.randn(N_NEURONS, N_NEURONS) * np.sqrt(2.0 / N_NEURONS))
    layers.append(rng.randn(N_NEURONS, n_outputs) * np.sqrt(2.0 / N_NEURONS))
    return layers


def forward_pass(x, layers):
    for layer in layers[:-1]:
        x = relu(x @ layer)
    return x @ layers[-1]


def generate_random_dag(n_nodes, seed):
    rng = np.random.RandomState(seed)
    n_edges = rng.randint(MIN_EDGES, MAX_EDGES + 1)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    valid = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng.shuffle(valid)
    added = 0
    for u, v in valid:
        if added >= n_edges:
            break
        G.add_edge(u, v)
        if nx.is_directed_acyclic_graph(G):
            added += 1
        else:
            G.remove_edge(u, v)
    return G


def generate_mock_data(dag, n_samples, seed):
    rng = np.random.RandomState(seed)
    order = list(nx.topological_sort(dag))
    mlps = {}
    for node in order:
        parents = list(dag.predecessors(node))
        mlps[node] = None if not parents else generate_mlp(len(parents), 1, seed + node)
    data = np.zeros((n_samples, len(dag.nodes())))
    for node in order:
        parents = list(dag.predecessors(node))
        if not parents:
            data[:, node] = (rng.beta(2, 5, size=n_samples) - 0.5) * 10
        else:
            out = forward_pass(data[:, parents], mlps[node]).flatten()
            if np.std(out) > 0:
                out = out / np.std(out) * np.std(data[:, parents][:, 0])
            data[:, node] = out + (rng.beta(2, 5, size=n_samples) - 0.5) * 0.5
    return pd.DataFrame(data, columns=PROPERTY_NAMES[:len(dag.nodes())])


def true_skeleton(dag):
    return {tuple(sorted((PROPERTY_NAMES[u], PROPERTY_NAMES[v]))) for u, v in dag.edges()}


_MARKS = ("<->", "-->", "<--", "o->", "<-o", "o-o")


def pred_skeleton(graph_str):
    sk, inblk = set(), False
    for raw in graph_str.split("\n"):
        s = raw.strip()
        if s.startswith("Graph Edges"):
            inblk = True
            continue
        if not inblk:
            continue
        if s.startswith("Graph "):
            break
        for m in _MARKS:
            if m in s:
                body = s.split(".", 1)[1].strip() if "." in s.split(m)[0] else s
                a, b = body.split(m, 1)
                sk.add(tuple(sorted((a.strip(), b.strip()))))
                break
    return sk


def f1_score(true_sk, pred_sk):
    tp = len(true_sk & pred_sk)
    fp = len(pred_sk - true_sk)
    fn = len(true_sk - pred_sk)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main():
    print(f"NSA tuning: sample={SAMPLE}, mocks=[{MOCK_START},{MOCK_END}), "
          f"truncs={TRUNCS}, pens={PENS} -> {os.path.basename(PART_CSV)}", flush=True)
    t0 = time.time()

    rows = []
    for t in TRUNCS:
        for p in PENS:
            scores, c0 = [], time.time()
            # mocks are generated per-index from fixed seeds (identical across cells
            # and across workers), one at a time, so only one mock is held in memory
            for j in range(MOCK_START, MOCK_END):
                dag = generate_random_dag(N_PROPERTIES, seed=42 + j)
                df = generate_mock_data(dag, SAMPLE, seed=42 + j * 1000)
                tsk = true_skeleton(dag)
                try:
                    s = ts.TetradSearch(df)
                    s.set_verbose(False)
                    s.use_basis_function_lrt(truncation_limit=t, alpha=ALPHA)
                    s.use_basis_function_bic(truncation_limit=t, penalty_discount=p)
                    s.run_fcit()
                    scores.append(f1_score(tsk, pred_skeleton(str(s.get_java()))))
                except Exception as e:
                    print(f"  t={t} p={p} mock{j} ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
                del df
            sum_f1 = float(np.sum(scores)) if scores else 0.0
            n = len(scores)
            rows.append({"truncation_limit": t, "penalty_discount": p,
                         "sum_f1": round(sum_f1, 6), "n": n,
                         "mean_f1": round(sum_f1 / n, 4) if n else 0.0})
            print(f"t={t:2d} p={p:3d}: mean_f1={(sum_f1/n if n else 0):.3f}  "
                  f"({n} mocks, {time.time()-c0:.0f}s)", flush=True)
            # save after every cell so progress is preserved if the run is interrupted
            pd.DataFrame(rows).to_csv(PART_CSV, index=False)

    print(f"\nshard [{MOCK_START},{MOCK_END}) done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
