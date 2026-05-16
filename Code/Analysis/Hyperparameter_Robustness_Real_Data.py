"""
Hyperparameter robustness test on real data.

For each dataset, runs FCIT on the full real data over a tight 3x3 grid
centered on the paper's optimum (truncation_limit +/- 1, penalty_discount +/- 20%)
and reports:
  - total edges per cell
  - directed-edge fraction per cell
  - presence/orientation of a fixed list of physically-motivated edges

Output: a per-dataset summary table and a one-line verdict on stability.

No subsampling. 9 FCIT runs per dataset, 27 total. Use many cores if available.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

ALPHA = 0.01
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
RESULTS_DIR = os.path.join(REPO_ROOT, "Results")


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# Key physical edges: each entry is (source, target, expected_mark)
# where expected_mark is one of:
#   "-->"  directed
#   "<->"  bidirected (latent confounder)
#   "o->"  partially directed (acceptable as "directed-like")
# Presence is checked symmetrically (A--B in either direction counts as present)
# but orientation must match expected_mark to count as "stable".

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    pickle_file: str
    baseline_trunc: int
    baseline_penalty: int
    key_edges: Tuple[Tuple[str, str, str], ...]


DATASETS: Tuple[DatasetConfig, ...] = (
    DatasetConfig(
        name="ALFALFA_NSA",
        pickle_file="alfalfa_nsa_final_13props.pkl",
        baseline_trunc=7,
        baseline_penalty=35,
        key_edges=(
            ("ELPETRO_MASS",   "ELPETRO_ABSMAG_R", "-->"),  # stellar_mass -> r_luminosity
            ("logMH",          "ELPETRO_MASS",     "-->"),  # HI_mass     -> stellar_mass
            ("W50",            "logMH",            "<->"),  # HI_linewidth <-> HI_mass
        ),
    ),
    DatasetConfig(
        name="NSA",
        pickle_file="nsa_final_10props.pkl",
        baseline_trunc=14,
        baseline_penalty=50,
        key_edges=(
            ("ELPETRO_MASS",      "ELPETRO_ABSMAG_R", "-->"),  # stellar_mass    -> r_luminosity
            ("ELPETRO_ABSMAG_R",  "ELPETRO_TH50_R",   "-->"),  # r_luminosity    -> half_light_radius
            ("ELPETRO_MTOL",      "ELPETRO_ABSMAG_R", "-->"),  # mass_to_light   -> r_luminosity
        ),
    ),
    DatasetConfig(
        name="TNG50",
        pickle_file="tng50_final.pkl",
        baseline_trunc=7,
        baseline_penalty=15,
        key_edges=(
            ("STELLAR_MASS", "BARYONIC_MASS",   "-->"),
            ("STELLAR_MASS", "HALFMASS_RAD",    "-->"),
            ("GAS_MASS",     "BH_MASS",         "-->"),
            ("DM_MASS",      "GAS_METALLICITY", "-->"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# FCIT runner
# ---------------------------------------------------------------------------

def load_dataset(pickle_file: str) -> pd.DataFrame:
    """Load a dataset pickle (dict of arrays) into a DataFrame."""
    with open(os.path.join(DATA_DIR, pickle_file), "rb") as f:
        data_dict = pickle.load(f)
    var_names = list(data_dict.keys())
    arr = np.column_stack([data_dict[v] for v in var_names])
    return pd.DataFrame(arr, columns=var_names)


def run_fcit(df: pd.DataFrame, truncation_limit: int, penalty_discount: int) -> str:
    """Run FCIT once and return Tetrad's graph string."""
    search = ts.TetradSearch(df)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=truncation_limit, alpha=ALPHA)
    search.use_basis_function_bic(truncation_limit=truncation_limit,
                                  penalty_discount=penalty_discount)
    search.run_fcit()
    return str(search.get_java())


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------

# Edge marks recognised in Tetrad's output. Order matters: longer tokens first.
_MARK_TOKENS = ("<->", "-->", "<--", "o->", "<-o", "o-o")


def _clean(name: str) -> str:
    """Strip Tetrad's numeric prefix (e.g. '1. FOO' -> 'FOO')."""
    name = name.strip()
    if "." in name:
        left, right = name.split(".", 1)
        if left.strip().isdigit():
            return right.strip()
    return name


def parse_edges(graph_str: str) -> List[Tuple[str, str, str]]:
    """Parse Tetrad graph string into a list of (a, mark, b) tuples."""
    edges: List[Tuple[str, str, str]] = []
    in_edges_block = False
    for raw in graph_str.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Graph Edges"):
            in_edges_block = True
            continue
        if not in_edges_block:
            continue
        # Once we leave the edges block (blank line above breaks the loop early),
        # any further "Graph ..." header ends it.
        if line.startswith("Graph "):
            break
        for tok in _MARK_TOKENS:
            if tok in line:
                left, right = line.split(tok, 1)
                a = _clean(left)
                b = _clean(right)
                # Normalise '<--' to '-->' with swapped endpoints
                if tok == "<--":
                    edges.append((b, "-->", a))
                elif tok == "<-o":
                    edges.append((b, "o->", a))
                else:
                    edges.append((a, tok, b))
                break
    return edges


def edge_directed_fraction(edges: List[Tuple[str, str, str]]) -> float:
    """Fraction of edges that are fully directed (-->)."""
    if not edges:
        return 0.0
    n_dir = sum(1 for _, m, _ in edges if m == "-->")
    return n_dir / len(edges)


def find_edge(edges: List[Tuple[str, str, str]], src: str, dst: str) -> str | None:
    """Return the orientation mark for the edge between src and dst, or None.

    Orientation is canonicalised so the returned mark describes src->dst:
      "-->", "o->", "o-o", "<->"  if src is the 'tail' side
      "<--", "<-o"                if direction is reversed
    """
    for a, mark, b in edges:
        if a == src and b == dst:
            return mark
        if a == dst and b == src:
            # Flip mark to express it from src's perspective
            flip = {"-->": "<--", "o->": "<-o", "o-o": "o-o", "<->": "<->"}.get(mark, mark)
            return flip
    return None


def classify_edge(observed: str | None, expected: str) -> str:
    """Compare observed orientation against expected. Returns one of:
       'match'   -- present with expected orientation
       'present' -- present but different orientation
       'missing' -- not present at all
    """
    if observed is None:
        return "missing"
    if observed == expected:
        return "match"
    return "present"


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def grid_for(cfg: DatasetConfig) -> List[Tuple[int, int]]:
    """3x3 grid: truncation +/- 1, penalty +/- 20% (rounded)."""
    t0, p0 = cfg.baseline_trunc, cfg.baseline_penalty
    truncs = [t0 - 1, t0, t0 + 1]
    pens = sorted({round(p0 * 0.8), p0, round(p0 * 1.2)})
    return [(t, p) for t in truncs for p in pens]


# ---------------------------------------------------------------------------
# Per-dataset run and summary
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    trunc: int
    penalty: int
    n_edges: int
    directed_fraction: float
    runtime_s: float
    key_edge_status: Dict[Tuple[str, str], str]  # (src,dst) -> match/present/missing


def run_dataset(cfg: DatasetConfig) -> List[CellResult]:
    print(f"\n[{cfg.name}] loading {cfg.pickle_file}")
    df = load_dataset(cfg.pickle_file)
    print(f"[{cfg.name}] n={len(df)}, vars={list(df.columns)}")
    print(f"[{cfg.name}] baseline t={cfg.baseline_trunc}, p={cfg.baseline_penalty}")

    results: List[CellResult] = []
    for t, p in grid_for(cfg):
        t0 = time.time()
        graph_str = run_fcit(df, truncation_limit=t, penalty_discount=p)
        edges = parse_edges(graph_str)
        statuses: Dict[Tuple[str, str], str] = {}
        for src, dst, expected in cfg.key_edges:
            observed = find_edge(edges, src, dst)
            statuses[(src, dst)] = classify_edge(observed, expected)
        cell = CellResult(
            trunc=t, penalty=p,
            n_edges=len(edges),
            directed_fraction=edge_directed_fraction(edges),
            runtime_s=time.time() - t0,
            key_edge_status=statuses,
        )
        results.append(cell)
        print(f"  t={t:>3} p={p:>3}  edges={cell.n_edges:>3}  "
              f"directed_frac={cell.directed_fraction:.2f}  "
              f"runtime={cell.runtime_s:6.1f}s")
    return results


def print_summary(cfg: DatasetConfig, cells: List[CellResult]) -> None:
    """Per-dataset readable summary."""
    print("\n" + "=" * 78)
    print(f"  {cfg.name}   (baseline t={cfg.baseline_trunc}, p={cfg.baseline_penalty})")
    print("=" * 78)

    # Grid table: edges + directed fraction
    print(f"\n  Grid summary (9 cells, full data, no subsampling):")
    print(f"  {'t':>4} {'p':>4} {'edges':>6} {'dir_frac':>10}")
    for c in cells:
        print(f"  {c.trunc:>4} {c.penalty:>4} {c.n_edges:>6} {c.directed_fraction:>10.2f}")

    edge_counts = [c.n_edges for c in cells]
    n_min, n_max = min(edge_counts), max(edge_counts)
    n_baseline = next(c.n_edges for c in cells
                      if c.trunc == cfg.baseline_trunc and c.penalty == cfg.baseline_penalty)
    rel_spread = (n_max - n_min) / n_baseline if n_baseline else float("inf")

    print(f"\n  Edge count: baseline={n_baseline}, min={n_min}, max={n_max}, "
          f"relative spread={rel_spread:.1%}")

    # Key edges across grid
    print(f"\n  Key physical edges (across all 9 cells):")
    all_stable = True
    for src, dst, expected in cfg.key_edges:
        statuses = [c.key_edge_status[(src, dst)] for c in cells]
        n_match = statuses.count("match")
        n_present = statuses.count("present")
        n_missing = statuses.count("missing")
        verdict = "STABLE" if n_match == 9 else (
            "ORIENT VARIES" if n_missing == 0 else "DROPS OUT"
        )
        if n_match != 9:
            all_stable = False
        print(f"    {src:>20} {expected:>4} {dst:<22}  "
              f"match={n_match}/9  present={n_present}/9  missing={n_missing}/9   {verdict}")

    # One-line verdict
    if all_stable and rel_spread < 0.15:
        verdict = "Main conclusions are stable across hyperparameter neighbourhood."
    elif all_stable:
        verdict = ("Key physical edges stable; edge density shows moderate variation "
                   f"({rel_spread:.0%}).")
    else:
        verdict = "Some sensitivity observed: at least one key edge varies across the grid."
    print(f"\n  Verdict: {verdict}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results: Dict[str, List[CellResult]] = {}
    for cfg in DATASETS:
        all_results[cfg.name] = run_dataset(cfg)

    for cfg in DATASETS:
        print_summary(cfg, all_results[cfg.name])

    # Also dump a flat CSV for the paper
    rows = []
    for cfg in DATASETS:
        for c in all_results[cfg.name]:
            row = {
                "dataset": cfg.name,
                "trunc": c.trunc,
                "penalty": c.penalty,
                "n_edges": c.n_edges,
                "directed_fraction": round(c.directed_fraction, 4),
                "runtime_s": round(c.runtime_s, 2),
            }
            for (src, dst), status in c.key_edge_status.items():
                row[f"{src}__{dst}"] = status
            rows.append(row)
    out_csv = os.path.join(RESULTS_DIR, "hyperparameter_robustness_real_data.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nFlat results written to: {out_csv}")


if __name__ == "__main__":
    main()
