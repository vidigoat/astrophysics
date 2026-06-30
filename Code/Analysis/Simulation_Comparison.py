"""
Causal-structure comparison: observations vs simulations, and simulation vs
simulation.

Runs FCIT many times on each matched sample, forms the consensus PAG (edge kept
if it appears in >= 50% of runs; oriented only if one mark dominates >= 60% of
its appearances), and writes:

  Results/consensus_matchA_{NSA,TNG50,SIMBA}.csv   (common observable variables)
  Results/consensus_matchB_{TNG50,SIMBA}.csv       (full intrinsic variables)
  Results/comparison_A.csv   per-pair presence+mark in NSA / TNG50 / SIMBA
  Results/comparison_B.csv   per-pair presence+mark in TNG50 / SIMBA + verdict

Comparison A asks whether the simulations reproduce the causal structure FCIT
finds among the *observable* properties of real galaxies. Comparison B asks
which causal links are robust across two independent galaxy-formation codes
(shared) versus driven by a particular subgrid model (divergent).
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
RESULTS = os.path.join(REPO, "Results")
ALPHA = 0.01
PRESENCE_THR = 0.50
ORIENT_THR = 0.60
N_RUNS = 50

# (name, file, truncation, penalty)
CFG = [
    ("matchA_NSA",   "matchA_NSA.pkl",   7, 20),
    ("matchA_TNG50", "matchA_TNG50.pkl", 7, 20),
    ("matchA_SIMBA", "matchA_SIMBA.pkl", 7, 20),
    ("matchB_TNG50", "matchB_TNG50.pkl", 7, 8),
    ("matchB_SIMBA", "matchB_SIMBA.pkl", 7, 8),
]
_MARKS = ("<->", "-->", "<--", "o->", "<-o", "o-o")
_FLIP = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}


def load(fn):
    import pickle
    d = pickle.load(open(os.path.join(DATA, fn), "rb"))
    return pd.DataFrame({k: np.asarray(v, float) for k, v in d.items()})


def parse_edges(graph_str):
    edges, inblock = [], False
    for raw in graph_str.split("\n"):
        s = raw.strip()
        if s.startswith("Graph Edges"):
            inblock = True
            continue
        if not inblock:
            continue
        if s.startswith("Graph "):
            break
        for m in _MARKS:
            if m in s:
                body = s.split(".", 1)[1].strip() if "." in s.split(m)[0] else s
                left, right = body.split(m, 1)
                edges.append((left.strip(), m, right.strip()))
                break
    return edges


def canon(a, mark, b):
    if a <= b:
        return (a, b), mark
    return (b, a), _FLIP[mark]


def consensus(name, fn, t, p, n_runs=N_RUNS):
    df = load(fn)
    present = Counter()
    marks = {}
    for _ in range(n_runs):
        s = ts.TetradSearch(df)
        s.set_verbose(False)
        s.use_basis_function_lrt(truncation_limit=t, alpha=ALPHA)
        s.use_basis_function_bic(truncation_limit=t, penalty_discount=p)
        s.run_fcit()
        for a, mk, b in parse_edges(str(s.get_java())):
            key, cm = canon(a, mk, b)
            present[key] += 1
            marks.setdefault(key, Counter())[cm] += 1
    rows = []
    for key, cnt in present.items():
        if cnt / n_runs < PRESENCE_THR:
            continue
        top, topn = marks[key].most_common(1)[0]
        cm = top if topn / cnt >= ORIENT_THR else "o-o"
        rows.append({"var_a": key[0], "var_b": key[1],
                     "presence": round(cnt / n_runs, 3),
                     "consensus_mark": cm})
    out = pd.DataFrame(rows).sort_values("presence", ascending=False)
    out.to_csv(os.path.join(RESULTS, f"consensus_{name}.csv"), index=False)
    n_or = (~out["consensus_mark"].isin(["o-o", "<->"])).sum()
    print(f"{name}: {len(out)} edges ({n_or} oriented)  [t={t}, p={p}]")
    return out


def _mark_map(df):
    return {(r.var_a, r.var_b): r.consensus_mark for r in df.itertuples()}


def compare(names, label):
    maps = {n: _mark_map(pd.read_csv(os.path.join(RESULTS, f"consensus_{n}.csv")))
            for n in names}
    allpairs = sorted(set().union(*[set(m) for m in maps.values()]))
    rows = []
    for pr in allpairs:
        row = {"var_a": pr[0], "var_b": pr[1]}
        for n in names:
            row[n.split("_")[-1]] = maps[n].get(pr, "-")
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(names) == 2:  # comparison B: agree / differ verdict
        a, b = [n.split("_")[-1] for n in names]
        def verdict(r):
            ma, mb = r[a], r[b]
            if ma == "-" or mb == "-":
                return "only_" + (a if mb == "-" else b)
            if ma == mb:
                return "identical"
            oriented = lambda m: m not in ("o-o", "<->", "-")
            if oriented(ma) and oriented(mb):
                return "both_oriented_diff" if ma != mb else "identical"
            return "skeleton_shared_orient_diff"
        df["verdict"] = df.apply(verdict, axis=1)
    df.to_csv(os.path.join(RESULTS, f"comparison_{label}.csv"), index=False)
    print(f"\n=== Comparison {label} ===")
    print(df.to_string(index=False))
    return df


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for name, fn, t, p in CFG:
        consensus(name, fn, t, p)
    compare(["matchA_NSA", "matchA_TNG50", "matchA_SIMBA"], "A")
    compare(["matchB_TNG50", "matchB_SIMBA"], "B")


if __name__ == "__main__":
    main()
