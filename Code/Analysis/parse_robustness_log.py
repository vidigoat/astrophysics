"""
Parse the verbose FCIT run log produced by Hyperparameter_Robustness_Real_Data.py
and extract the actual PAG edge marks for every (dataset, t, p) cell.

The original CSV only stores 'match'/'present'/'missing' against a hardcoded
'expected_mark' that assumes fully-directed edges (-->). But FCIT outputs PAGs
where the canonical mark for a directed-like edge is 'o->' (circle on the source
side, arrowhead on the destination side). That means the original 'present'
category lumps together:
  - 'o->' (still arrowhead at expected target — consistent with expected direction)
  - '<-o' (arrowhead at the SOURCE  — REVERSED, inconsistent)
  - '<->' (bidirected — latent confounder, weakly consistent)
  - 'o-o' (no orientation either way — uninformative)

For the referee response we want to distinguish these. Here we reparse the log
and classify every key-edge observation into one of:
  COMPATIBLE   – arrowhead at expected target (-->, o->, <->)
  REVERSED     – arrowhead at expected source (<--, <-o)
  UNDIRECTED   – circles both sides (o-o)
  MISSING      – edge absent

Writes:
  Results/hyperparameter_robustness_marks.csv  (one row per (dataset,t,p,edge)
                                                with the raw observed mark)
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(REPO_ROOT, "Results", "robustness_run.log")
MARKS_CSV = os.path.join(REPO_ROOT, "Results", "hyperparameter_robustness_marks.csv")

# Must mirror Hyperparameter_Robustness_Real_Data.py.
DATASETS = [
    ("ALFALFA_NSA", 7, 35, 13, [
        ("ELPETRO_MASS",  "ELPETRO_ABSMAG_R"),
        ("logMH",         "ELPETRO_MASS"),
        ("W50",           "logMH"),
    ]),
    ("NSA", 14, 50, 10, [
        ("ELPETRO_MASS",     "ELPETRO_ABSMAG_R"),
        ("ELPETRO_ABSMAG_R", "ELPETRO_TH50_R"),
        ("ELPETRO_MTOL",     "ELPETRO_ABSMAG_R"),
    ]),
    ("TNG50", 7, 15, None, [
        ("STELLAR_MASS", "BARYONIC_MASS"),
        ("STELLAR_MASS", "HALFMASS_RAD"),
        ("GAS_MASS",     "BH_MASS"),
        ("DM_MASS",      "GAS_METALLICITY"),
    ]),
]

# Identify dataset by the set of node names in the 'Graph Nodes:' line.
DATASET_KEYS = {
    "ALFALFA_NSA": frozenset({
        "BARYONIC_MASS", "COLOR_U_R", "ELPETRO_B300", "SERSIC_N", "ELPETRO_METS",
        "ELPETRO_MTOL", "ELPETRO_BA", "ELPETRO_TH50_R", "ZDIST", "logMH",
        "ELPETRO_MASS", "ELPETRO_ABSMAG_R", "W50",
    }),
    "NSA": frozenset({
        "COLOR_U_R", "ELPETRO_B300", "SERSIC_N", "ELPETRO_METS", "ELPETRO_MTOL",
        "ELPETRO_BA", "ELPETRO_TH50_R", "ZDIST", "ELPETRO_MASS", "ELPETRO_ABSMAG_R",
    }),
    "TNG50": frozenset({
        "DM_MASS", "STELLAR_MASS", "GAS_MASS", "BH_MASS", "BARYONIC_MASS",
        "HALFMASS_RAD", "VEL_DISP", "VMAX", "GAS_METALLICITY", "STAR_METALLICITY",
        "PHOTOMETRIC_U", "PHOTOMETRIC_R", "SFR", "COLOUR",
    }),
}


_MARK_TOKENS = ("<->", "-->", "<--", "o->", "<-o", "o-o")


def grid_for(t0: int, p0: int) -> List[Tuple[int, int]]:
    truncs = [t0 - 1, t0, t0 + 1]
    pens = sorted({round(p0 * 0.8), p0, round(p0 * 1.2)})
    return [(t, p) for t in truncs for p in pens]


def identify_dataset(nodes: set[str]) -> str | None:
    """Best guess by node overlap. Returns dataset name or None."""
    best, best_score = None, 0
    for name, keys in DATASET_KEYS.items():
        score = len(nodes & keys)
        # Require the candidate's full key set to be inside `nodes` (subset
        # match) — otherwise NSA would also match ALFALFA_NSA since NSA's
        # 10 vars are a subset of ALFALFA_NSA's 13.
        if not keys.issubset(nodes):
            continue
        # Tie-break: prefer the more specific (larger key set) match.
        if len(keys) > best_score:
            best, best_score = name, len(keys)
    return best


def parse_log_into_graphs(log_text: str) -> List[Tuple[set[str], List[Tuple[str, str, str]]]]:
    """Return a list of (nodes, edges) tuples in the order they appear."""
    graphs: List[Tuple[set[str], List[Tuple[str, str, str]]]] = []
    # Split on 'Graph Nodes:' headers — each occurrence begins one graph.
    chunks = log_text.split("Graph Nodes:")[1:]
    for chunk in chunks:
        # Take everything up to the next blank line after 'Graph Edges:'.
        if "Graph Edges:" not in chunk:
            continue
        nodes_part, rest = chunk.split("Graph Edges:", 1)
        nodes = {n.strip() for n in nodes_part.replace("\n", "").split(";") if n.strip()}
        edges: List[Tuple[str, str, str]] = []
        for raw in rest.split("\n"):
            line = raw.strip()
            if not line:
                # First blank line after edges ends the block — but only if we
                # already have at least one edge OR after we exit the header lines.
                if edges:
                    break
                continue
            # Skip lines that aren't numbered edges.
            if not re.match(r"^\d+\.\s", line):
                # If we encounter a non-edge after edges were collected, stop.
                if edges:
                    break
                continue
            body = re.sub(r"^\d+\.\s+", "", line).strip()
            for tok in _MARK_TOKENS:
                if tok in body:
                    left, right = body.split(tok, 1)
                    a, b = left.strip(), right.strip()
                    edges.append((a, tok, b))
                    break
        graphs.append((nodes, edges))
    return graphs


def classify_mark(observed: str | None, expected_src: str, expected_dst: str,
                  graph_edges: List[Tuple[str, str, str]]) -> Tuple[str, str]:
    """Given the EXPECTED direction (src -> dst), look up the edge in the graph
    and return (raw_mark, classification).

    raw_mark is the mark expressed FROM src's perspective:
        '-->'  src -> dst       compatible
        'o->'  src o-> dst       compatible (PAG ambiguous on src side)
        '<->'  src <-> dst       weakly compatible (latent confounder)
        '<--'  src <- dst        reversed
        '<-o'  src <-o dst       reversed (PAG ambiguous on dst side)
        'o-o'  src o-o dst       undirected
        None   not present       missing
    """
    raw = None
    for a, mark, b in graph_edges:
        if a == expected_src and b == expected_dst:
            raw = mark
            break
        if a == expected_dst and b == expected_src:
            flip = {
                "-->": "<--", "<--": "-->",
                "o->": "<-o", "<-o": "o->",
                "o-o": "o-o", "<->": "<->",
            }.get(mark, mark)
            raw = flip
            break

    if raw is None:
        return ("", "MISSING")
    if raw in ("-->", "o->", "<->"):
        cls = "COMPATIBLE"
    elif raw in ("<--", "<-o"):
        cls = "REVERSED"
    elif raw == "o-o":
        cls = "UNDIRECTED"
    else:
        cls = "OTHER"
    return (raw, cls)


def main() -> None:
    if not os.path.exists(LOG_PATH):
        sys.exit(f"Log not found: {LOG_PATH}")
    with open(LOG_PATH) as f:
        log_text = f.read()

    graphs = parse_log_into_graphs(log_text)
    print(f"Parsed {len(graphs)} graphs from log")

    # Walk graphs in the order datasets/grid_for produce them.
    rows: List[dict] = []
    g_idx = 0
    for ds_name, t0, p0, _, key_edges in DATASETS:
        cells = grid_for(t0, p0)
        for (t, p) in cells:
            if g_idx >= len(graphs):
                print(f"[warn] ran out of graphs at {ds_name} t={t} p={p}")
                break
            nodes, edges = graphs[g_idx]
            guessed = identify_dataset(nodes)
            if guessed != ds_name:
                print(f"[warn] graph #{g_idx} dataset mismatch: "
                      f"expected {ds_name}, log shows {guessed} ({len(nodes)} nodes)")
            for (src, dst) in key_edges:
                raw, cls = classify_mark(None, src, dst, edges)
                rows.append({
                    "dataset": ds_name,
                    "trunc": t,
                    "penalty": p,
                    "is_baseline": (t == t0 and p == p0),
                    "src": src,
                    "dst": dst,
                    "raw_mark": raw,
                    "class": cls,
                    "n_edges": len(edges),
                })
            g_idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(MARKS_CSV, index=False)
    print(f"Wrote {len(df)} rows to {MARKS_CSV}")
    print()
    # Per-dataset summary
    for ds_name, t0, p0, _, key_edges in DATASETS:
        sub = df[df["dataset"] == ds_name]
        print(f"=== {ds_name} (baseline t={t0}, p={p0}) ===")
        for (src, dst) in key_edges:
            e = sub[(sub["src"] == src) & (sub["dst"] == dst)]
            counts = e["class"].value_counts().to_dict()
            base = e[e["is_baseline"]]
            base_mark = base["raw_mark"].iloc[0] if not base.empty else "?"
            print(f"  {src:>20} -> {dst:<22}  baseline={base_mark!r:>8}  "
                  f"COMPATIBLE={counts.get('COMPATIBLE',0)}/9  "
                  f"REVERSED={counts.get('REVERSED',0)}/9  "
                  f"UNDIRECTED={counts.get('UNDIRECTED',0)}/9  "
                  f"MISSING={counts.get('MISSING',0)}/9")
        print()


if __name__ == "__main__":
    main()
