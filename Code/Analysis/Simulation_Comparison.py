"""
Causal-structure comparison across observations and three independent
cosmological simulations (TNG50, EAGLE, SIMBA).

Runs FCIT many times on each matched sample, forms the consensus PAG, and writes:
  Results/consensus_matchA_{NSA,TNG50,EAGLE,SIMBA}.csv  (common observables)
  Results/consensus_matchB_{TNG50,EAGLE,SIMBA}.csv       (full intrinsic set)
  Results/comparison_A.csv   per-pair mark in NSA / TNG50 / EAGLE / SIMBA
  Results/comparison_B.csv   per-pair marks across the three codes + a class:
     invariant_oriented   -- present, same direction in all three (robust physics)
     invariant_undirected -- present, undirected in all three (robust, dir. unresolved)
     invariant_skeleton   -- present in all three, directions DIFFER (model-dependent dir.)
     majority             -- present in two of three
     code_specific        -- present in only one

Comparison A asks whether the simulations reproduce the causal structure FCIT
finds among observable galaxy properties. Comparison B is the model-discrimination
test: edges in the invariant backbone are candidate robust physics; the rest
localise where the recovered structure depends on the subgrid model.
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

CFG = [
    ("matchA_NSA",   "matchA_NSA.pkl",   7, 20),
    ("matchA_TNG50", "matchA_TNG50.pkl", 7, 20),
    ("matchA_EAGLE", "matchA_EAGLE.pkl", 7, 20),
    ("matchA_SIMBA", "matchA_SIMBA.pkl", 7, 20),
    ("matchB_TNG50", "matchB_TNG50.pkl", 7, 8),
    ("matchB_EAGLE", "matchB_EAGLE.pkl", 7, 8),
    ("matchB_SIMBA", "matchB_SIMBA.pkl", 7, 8),
]
_MARKS = ("<->", "-->", "<--", "o->", "<-o", "o-o")
_FLIP = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}
_ORIENTED = ("-->", "<--", "o->", "<-o")


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
    return ((a, b), mark) if a <= b else ((b, a), _FLIP[mark])


def _direction(m):
    """Determined causal direction of a canonical 'a MARK b' edge."""
    if m in ("-->", "o->"):
        return "fwd"   # arrowhead at b
    if m in ("<--", "<-o"):
        return "rev"   # arrowhead at a
    return "und"        # o-o, <->


def consensus(name, fn, t, p, n_runs=N_RUNS):
    df = load(fn)
    present, marks = Counter(), {}
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
                     "presence": round(cnt / n_runs, 3), "consensus_mark": cm})
    out = pd.DataFrame(rows).sort_values("presence", ascending=False)
    out.to_csv(os.path.join(RESULTS, f"consensus_{name}.csv"), index=False)
    n_or = out["consensus_mark"].isin(_ORIENTED).sum()
    print(f"{name}: {len(out)} edges ({n_or} oriented)  [t={t}, p={p}]")
    return out


def _mark_map(name):
    df = pd.read_csv(os.path.join(RESULTS, f"consensus_{name}.csv"))
    return {(r.var_a, r.var_b): r.consensus_mark for r in df.itertuples()}


def compare_A():
    codes = ["NSA", "TNG50", "EAGLE", "SIMBA"]
    maps = {c: _mark_map(f"matchA_{c}") for c in codes}
    allpairs = sorted(set().union(*[set(m) for m in maps.values()]))
    rows = [{"var_a": a, "var_b": b, **{c: maps[c].get((a, b), "-") for c in codes}}
            for (a, b) in allpairs]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "comparison_A.csv"), index=False)
    print("\nComparison A:", len(rows), "edges in union")


def compare_B():
    codes = ["TNG50", "EAGLE", "SIMBA"]
    maps = {c: _mark_map(f"matchB_{c}") for c in codes}
    allpairs = sorted(set().union(*[set(m) for m in maps.values()]))
    rows = []
    for (a, b) in allpairs:
        mk = {c: maps[c].get((a, b), "-") for c in codes}
        present = [c for c in codes if mk[c] != "-"]
        if len(present) == 3:
            # classify by the determined causal DIRECTION (an arrowhead at b is
            # the same direction whether the tail is a circle or a bar), so o->
            # and --> count as agreement, since the tail mark is not reproducible.
            dirs = [_direction(mk[c]) for c in codes]
            if all(d == "fwd" for d in dirs) or all(d == "rev" for d in dirs):
                cls = "invariant_oriented"     # robust backbone: same direction in all 3
            elif all(d == "und" for d in dirs):
                cls = "invariant_undirected"
            else:
                cls = "invariant_conflict"     # present in all 3, directions disagree
        elif len(present) == 2:
            cls = "majority"
        else:
            cls = "code_specific"
        rows.append({"var_a": a, "var_b": b, **mk, "edge_class": cls})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "comparison_B.csv"), index=False)
    print("\n=== Comparison B (3-code) class tally ===")
    print(df["edge_class"].value_counts().to_string())
    print("\nINVARIANT BACKBONE (present in all three codes):")
    inv = df[df["edge_class"].str.startswith("invariant")]
    for r in inv.itertuples():
        print(f"  {r.var_a:13s} -- {r.var_b:13s}  TNG50:{r.TNG50:4s} EAGLE:{r.EAGLE:4s} SIMBA:{r.SIMBA:4s}  [{r.edge_class}]")
    return df


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for name, fn, t, p in CFG:
        consensus(name, fn, t, p)
    compare_A()
    compare_B()


if __name__ == "__main__":
    main()
