"""
Consensus / orientation-stability analysis for FCIT.

FCIT's skeleton is stable run-to-run, but the BOSS/GRaSP permutation search
leaves edge ORIENTATIONS non-deterministic. This script runs FCIT N times on
each real data set at the adopted hyperparameters and reports, for every pair
of variables:
  - presence  : fraction of runs in which the edge appears
  - the orientation distribution (arrow-at-B / arrow-at-A / undirected)

The "consensus graph" keeps edges present in >= PRESENCE_THR of runs, and
assigns each a single orientation only if one direction dominates in
>= ORIENT_THR of the runs in which the edge appears; otherwise it is reported
as undirected (o-o). This is the reproducible structure that should be plotted
and interpreted in the paper.

Output: Results/consensus_<dataset>.csv and a printed summary.
"""

from __future__ import annotations
import os, pickle
from collections import Counter
import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

ALPHA = 0.01
PRESENCE_THR = 0.50   # edge kept if it appears in >= 50% of runs
ORIENT_THR = 0.60     # direction assigned if it dominates in >= 60% of appearances

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data")
RESULTS = os.path.join(REPO, "Results")

# (name, file, truncation, penalty, n_runs)
# NSA is ~60s/run on 484k rows, so it is run last with fewer repeats; the fast
# samples (ALFALFA, TNG50) get many repeats.
CFG = [
    ("ALFALFA_NSA", "alfalfa_nsa_final_13props.pkl", 7, 35, 50),
    ("TNG50",       "tng50_final.pkl",                7, 15, 50),
    ("NSA",         "nsa_final_10props.pkl",         14, 50, 10),
]

_MARKS = ("<->", "-->", "<--", "o->", "<-o", "o-o")


def load(fname):
    with open(os.path.join(DATA, fname), "rb") as f:
        d = pickle.load(f)
    v = list(d.keys())
    return pd.DataFrame(np.column_stack([d[k] for k in v]), columns=v)


def parse_edges(graph_str):
    """Return list of (a, mark, b) for the Graph Edges block."""
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
                # strip leading "<n>. "
                body = s.split(".", 1)[1].strip() if "." in s.split(m)[0] else s
                left, right = body.split(m, 1)
                edges.append((left.strip(), m, right.strip()))
                break
    return edges


def canon(a, mark, b):
    """Return (pairkey, arrow_at) where pairkey is sorted (x,y) and arrow_at in
    {'second','first','none'} describing where the arrowhead points in sorted order."""
    if a <= b:
        x, y, mk = a, b, mark
    else:
        # flip endpoints and mark
        flip = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}
        x, y, mk = b, a, flip[mark]
    if mk in ("-->", "o->"):
        arrow = "second"     # x ... -> y  (arrow at y)
    elif mk in ("<--", "<-o"):
        arrow = "first"      # x <- ... y  (arrow at x)
    else:
        arrow = "none"       # o-o or <->
    return (x, y), arrow


def run_dataset(name, fname, t, p, n_runs):
    df = load(fname)              # load ONCE, reuse across runs
    pair_present = Counter()
    pair_arrow = {}  # pairkey -> Counter of arrow positions
    for i in range(n_runs):
        s = ts.TetradSearch(df)
        s.set_verbose(False)
        s.use_basis_function_lrt(truncation_limit=t, alpha=ALPHA)
        s.use_basis_function_bic(truncation_limit=t, penalty_discount=p)
        s.run_fcit()
        for a, mk, b in parse_edges(str(s.get_java())):
            key, arrow = canon(a, mk, b)
            pair_present[key] += 1
            pair_arrow.setdefault(key, Counter())[arrow] += 1

    rows = []
    for key, cnt in pair_present.items():
        x, y = key
        arrows = pair_arrow[key]
        n2 = arrows.get("second", 0)   # arrow at y
        n1 = arrows.get("first", 0)    # arrow at x
        nn = arrows.get("none", 0)
        # decide consensus orientation among the runs where the edge appears
        if n2 / cnt >= ORIENT_THR:
            cons = f"{x} --> {y}"
        elif n1 / cnt >= ORIENT_THR:
            cons = f"{y} --> {x}"
        else:
            cons = f"{x} o-o {y}"
        rows.append({
            "var_a": x, "var_b": y,
            "presence": round(cnt / n_runs, 3),
            "arrow_at_b_frac": round(n2 / cnt, 3),
            "arrow_at_a_frac": round(n1 / cnt, 3),
            "undirected_frac": round(nn / cnt, 3),
            "consensus": cons,
        })
    df_out = pd.DataFrame(rows).sort_values("presence", ascending=False)
    return df_out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for name, fname, t, p, n_runs in CFG:
        print(f"\n{'='*72}\n{name}  (t={t}, p={p}, {n_runs} runs)\n{'='*72}", flush=True)
        out = run_dataset(name, fname, t, p, n_runs)
        out.to_csv(os.path.join(RESULTS, f"consensus_{name}.csv"), index=False)
        stable = out[out["presence"] >= PRESENCE_THR]
        oriented = stable[~stable["consensus"].str.contains("o-o")]
        print(f"edges present >= {PRESENCE_THR:.0%}: {len(stable)}   "
              f"of which oriented (>= {ORIENT_THR:.0%}): {len(oriented)}")
        print(stable[["var_a", "var_b", "presence", "consensus"]].to_string(index=False))


if __name__ == "__main__":
    main()
