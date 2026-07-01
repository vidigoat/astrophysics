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


# Plain-English names for each variable, so PAG nodes carry both a readable
# label and the technical catalogue name (cf. Desmond & Ramsey 2026, Fig. 4).
PLAIN = {
    # observational (ALFALFA--NSA, NSA)
    "ZDIST": "Redshift",
    "ELPETRO_ABSMAG_R": "r-band luminosity",
    "ELPETRO_B300": "Recent star formation",
    "ELPETRO_MASS": "Stellar mass",
    "SERSIC_N": "Sersic index",
    "ELPETRO_BA": "Axis ratio",
    "ELPETRO_TH50_R": "Half-light radius",
    "logMH": "HI mass",
    "W50": "HI line width",
    "BARYONIC_MASS": "Baryonic mass",
    "COLOR_U_R": "Colour (U-R)",
    "ELPETRO_METS": "Stellar metallicity",
    "ELPETRO_MTOL": "Mass-to-light ratio",
    # TNG50
    "DM_MASS": "Halo mass",
    "STELLAR_MASS": "Stellar mass",
    "GAS_MASS": "Gas mass",
    "BH_MASS": "Black-hole mass",
    "HALFMASS_RAD": "Half-mass radius",
    "VEL_DISP": "Velocity dispersion",
    "VMAX": "Max. circular velocity",
    "GAS_METALLICITY": "Gas metallicity",
    "STAR_METALLICITY": "Stellar metallicity",
    "PHOTOMETRIC_R": "r-band luminosity",
    "PHOTOMETRIC_U": "u-band luminosity",
    "COLOUR": "Colour (U-R)",
    "SFR": "Star formation",
}


def _node_label(var):
    """Graphviz HTML-like label: plain-English name on top, technical
    variable name beneath in a smaller font. Falls back to the raw name."""
    plain = PLAIN.get(var)
    if not plain:
        return var
    return f'<{plain}<BR/><FONT POINT-SIZE="10">{var}</FONT>>'


def main():
    os.makedirs(PLOTS, exist_ok=True)
    only = os.environ.get("ONLY")  # render just one dataset if set (e.g. ONLY=TNG50)
    for name, base in DATASETS.items():
        if only and name != only:
            continue
        csv = os.path.join(RESULTS, f"consensus_{name}.csv")
        if not os.path.exists(csv):
            print(f"skip {name}: {csv} missing")
            continue
        df = pd.read_csv(csv)
        df = df[df["presence"] >= PRESENCE_THR]
        g = gviz.Digraph(engine="dot")
        g.attr(rankdir="TB", splines="true", overlap="false", dpi="200")
        if name == "TNG50":
            # The TNG50 graph is the densest (15 nodes, 33 edges). Give the nodes
            # more room and a larger label, and make the arrowheads/edges thinner,
            # so the nodes read clearly and the arrows do not dominate the figure.
            g.attr(ratio="0.72", ranksep="0.7", nodesep="0.5")
            g.attr("node", shape="ellipse", fontname="Helvetica", fontsize="15",
                   margin="0.12,0.06", penwidth="1.0")
            g.attr("edge", arrowsize="0.5", penwidth="0.8")
        else:
            g.attr("node", shape="ellipse", fontsize="11", fontname="Helvetica")
            g.attr("edge", arrowsize="0.8")
        # Define every node first, with a dual (plain + technical) label.
        seen = set()
        for _, r in df.iterrows():
            for v in (str(r["var_a"]), str(r["var_b"])):
                if v not in seen:
                    seen.add(v)
                    g.node(v, label=_node_label(v))
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
