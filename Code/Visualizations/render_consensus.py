"""
Render the consensus PAGs (from Code/Analysis/Consensus_Stability.py) to PNGs.

Across repeated runs the skeleton and the *position* of each determined
arrowhead are stable, but the finer distinction between a fully directed edge
(-->) and a partially oriented one (o->) -- i.e. whether the tail endpoint is
also determined -- is sensitive to the BOSS/GRaSP search order and is NOT
reproducible run-to-run. We therefore render orientation conservatively at the
arrowhead level: every edge whose arrowhead is determined is drawn as o-> (a
circle on the unoriented end, an arrowhead on the oriented end), and edges with
no determined arrowhead are drawn as o-o (circles on both ends). This never
claims a tail determination that does not survive re-running the search.

  o->  oriented   : circle tail, arrowhead  (the head end is not an ancestor of
                    the tail end; a latent confounder is not excluded)
  o-o  undirected : circle, circle          (neither endpoint determined)

Only edges present in >= PRESENCE_THR of runs are shown. Output goes to Plots/.
"""
import os
import pandas as pd
import graphviz as gviz

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "Results")
PLOTS = os.path.join(REPO, "Plots")
PRESENCE_THR = 0.50

DATASETS = {
    "ALFALFA_NSA": "alfalfa_nsa_fcit_t7_p35",
    "NSA":         "nsa_fcit_t14_p50",
    "TNG50":       "tng50_fcit_t7_p15",
}

# Graphviz endpoint attributes (conservative: a determined arrowhead is drawn as
# o->, with a circle on the unoriented end; we never draw a bare tail because the
# tail determination is not reproducible across runs). A circle endpoint uses the
# "odot" arrow style.
_EDGE_STYLE = {
    "-->": dict(dir="both", arrowtail="odot",   arrowhead="normal"),
    "o->": dict(dir="both", arrowtail="odot",   arrowhead="normal"),
    "<--": dict(dir="both", arrowtail="normal", arrowhead="odot"),
    "<-o": dict(dir="both", arrowtail="normal", arrowhead="odot"),
    "o-o": dict(dir="both", arrowtail="odot",   arrowhead="odot"),
    "<->": dict(dir="both", arrowtail="odot",   arrowhead="odot"),
}


def _mark_of(row):
    """Return the canonical full PAG mark for a consensus-CSV row, supporting both
    the new 'consensus_mark' column and the older 'consensus' string format."""
    if "consensus_mark" in row and isinstance(row["consensus_mark"], str):
        return row["consensus_mark"]
    c = str(row["consensus"])
    for m in ("<->", "-->", "<--", "o->", "<-o", "o-o"):
        if f" {m} " in c:
            return m
    return "o-o"


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for name, base in DATASETS.items():
        csv = os.path.join(RESULTS, f"consensus_{name}.csv")
        if not os.path.exists(csv):
            print(f"skip {name}: {csv} missing")
            continue
        df = pd.read_csv(csv)
        df = df[df["presence"] >= PRESENCE_THR]
        g = gviz.Digraph(engine="dot")
        g.attr(rankdir="TB", splines="true", overlap="false")
        if name == "TNG50":
            # The TNG50 graph has many nodes and is laid out very wide/flat by
            # default, which makes it look small next to the squarer ALFALFA/NSA
            # graphs. Nudge it toward a more square aspect so the three figures
            # are visually balanced at the same printed width.
            g.attr(ratio="0.75", ranksep="0.55", nodesep="0.35")
        g.attr("node", shape="ellipse", fontsize="11", fontname="Helvetica")
        g.attr("edge", arrowsize="0.8")
        n_or = n_undir = 0
        for _, r in df.iterrows():
            a, b = str(r["var_a"]), str(r["var_b"])
            mark = _mark_of(r)
            g.edge(a, b, **_EDGE_STYLE.get(mark, _EDGE_STYLE["o-o"]))
            if mark in ("o-o", "<->"):
                n_undir += 1
            else:
                n_or += 1
        g.render(filename=base, directory=PLOTS, format="png", cleanup=True)
        print(f"{name}: {len(df)} edges -> {n_or} oriented (arrowhead), "
              f"{n_undir} undirected -> Plots/{base}.png")


if __name__ == "__main__":
    main()
