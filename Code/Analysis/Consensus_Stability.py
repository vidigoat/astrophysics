"""
Consensus / orientation-stability analysis for FCIT.

FCIT's skeleton is stable run-to-run, but the BOSS/GRaSP permutation search
leaves edge ORIENTATIONS non-deterministic. This script runs FCIT N times on
each real data set at the adopted hyperparameters and reports, for every pair
of variables:
  - presence  : fraction of runs in which the edge appears
  - the orientation distribution (arrow-at-B / arrow-at-A / undirected)

The "consensus graph" keeps edges present in >= PRESENCE_THR of runs, and
assigns each the modal full PAG mark (-->, o->, o-o, <->) only if that mark
dominates in >= ORIENT_THR of the runs in which the edge appears; otherwise it
is reported as undirected (o-o). Crucially the FULL mark is preserved, so a
fully directed edge (-->, no latent confounder under faithfulness) is kept
distinct from a partially oriented one (o->, latent confounder still possible).
This is the reproducible structure that should be plotted and interpreted in the
paper.

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


_FLIP = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}


def canon(a, mark, b):
    """Return (pairkey, canonical_mark) where pairkey is the sorted pair (x, y)
    with x <= y, and canonical_mark is the *full* PAG mark rewritten for that
    ordering. We deliberately keep the complete mark (-->, o->, o-o, <->) rather
    than collapsing it to an arrow position, so that a partially-oriented edge
    (o->, latent confounder still possible) is never reported as a fully directed
    edge (-->, confounder excluded under faithfulness). The o->/--> split is then
    aggregated across runs like everything else, so the reported split is itself a
    consensus rather than an artefact of any single run."""
    if a <= b:
        return (a, b), mark
    return (b, a), _FLIP[mark]


def run_dataset(name, fname, t, p, n_runs):
    df = load(fname)              # load ONCE, reuse across runs
    pair_present = Counter()
    pair_marks = {}  # pairkey -> Counter of canonical full marks
    for i in range(n_runs):
        s = ts.TetradSearch(df)
        s.set_verbose(False)
        s.use_basis_function_lrt(truncation_limit=t, alpha=ALPHA)
        s.use_basis_function_bic(truncation_limit=t, penalty_discount=p)
        s.run_fcit()
        for a, mk, b in parse_edges(str(s.get_java())):
            key, cmark = canon(a, mk, b)
            pair_present[key] += 1
            pair_marks.setdefault(key, Counter())[cmark] += 1

    rows = []
    for key, cnt in pair_present.items():
        x, y = key
        marks = pair_marks[key]
        top_mark, top_n = marks.most_common(1)[0]
        # arrow-position tallies (kept for reference / backward compatibility)
        n_at_y = sum(v for m, v in marks.items() if m in ("-->", "o->"))
        n_at_x = sum(v for m, v in marks.items() if m in ("<--", "<-o"))
        n_none = sum(v for m, v in marks.items() if m in ("o-o", "<->"))
        # consensus full mark: the modal mark if it dominates >= ORIENT_THR of the
        # runs in which the edge appears, otherwise report it as undirected.
        cons_mark = top_mark if top_n / cnt >= ORIENT_THR else "o-o"
        rows.append({
            "var_a": x, "var_b": y,
            "presence": round(cnt / n_runs, 3),
            "mark_frac": round(top_n / cnt, 3),
            "arrow_at_b_frac": round(n_at_y / cnt, 3),
            "arrow_at_a_frac": round(n_at_x / cnt, 3),
            "undirected_frac": round(n_none / cnt, 3),
            "consensus_mark": cons_mark,
            "consensus": f"{x} {cons_mark} {y}",
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
