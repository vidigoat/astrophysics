"""
Render the simulation-comparison PAGs (from Code/Analysis/Simulation_Comparison.py).

  Comparison A: three small PAGs (NSA, TNG50, SIMBA) on the common observable
                variables -> Plots/cmpA_{nsa,tng50,simba}.png
  Comparison B: a single merged TNG50-vs-SIMBA graph in which every edge is
                coloured by whether the two codes agree -> Plots/cmpB_merged.png
                  black solid  = identical in both codes (robust)
                  orange bold  = present in both but oriented differently
                  blue dashed  = recovered only by TNG50
                  red dashed   = recovered only by SIMBA
"""
import os
import pandas as pd
import graphviz as gviz

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "Results")
PLOTS = os.path.join(REPO, "Plots")

LABELS = {
    "stellar_mass": "Stellar mass", "r_mag": "r-band luminosity",
    "colour": "Colour (U-R)", "metallicity": "Stellar metallicity",
    "size": "Size", "halo_mass": "Halo mass", "gas_mass": "Gas mass",
    "bh_mass": "Black-hole mass", "baryon_mass": "Baryonic mass",
    "veldisp": "Velocity dispersion", "Z_star": "Stellar metallicity",
    "Z_gas": "Gas metallicity", "u_mag": "u-band luminosity",
    "sfr": "Star formation",
}

# endpoint styles for a given full PAG mark (circle = odot)
_STYLE = {
    "-->": dict(dir="both", arrowtail="none",  arrowhead="normal"),
    "o->": dict(dir="both", arrowtail="odot",  arrowhead="normal"),
    "<--": dict(dir="both", arrowtail="normal", arrowhead="none"),
    "<-o": dict(dir="both", arrowtail="normal", arrowhead="odot"),
    "o-o": dict(dir="both", arrowtail="odot",  arrowhead="odot"),
    "<->": dict(dir="both", arrowtail="normal", arrowhead="normal"),
}


def _lab(v):
    return LABELS.get(v, v)


def render_simple(name, out):
    df = pd.read_csv(os.path.join(RESULTS, f"consensus_{name}.csv"))
    g = gviz.Digraph(engine="dot")
    g.attr(rankdir="TB", overlap="false", splines="true")
    g.attr("node", shape="ellipse", fontsize="13", fontname="Helvetica")
    g.attr("edge", arrowsize="0.85", penwidth="1.3")
    nodes = set()
    for _, r in df.iterrows():
        for v in (r["var_a"], r["var_b"]):
            if v not in nodes:
                nodes.add(v)
                g.node(v, label=_lab(v))
    for _, r in df.iterrows():
        g.edge(r["var_a"], r["var_b"], **_STYLE.get(r["consensus_mark"], _STYLE["o-o"]))
    g.render(filename=out, directory=PLOTS, format="png", cleanup=True)
    print(f"{name}: {len(df)} edges -> Plots/{out}.png")


def render_agreement_B(out):
    """Shared skeleton only (edges both codes recover): black = same
    orientation in both (robust), orange = codes disagree on direction."""
    df = pd.read_csv(os.path.join(RESULTS, "comparison_B.csv"))
    shared = df[(df["TNG50"] != "-") & (df["SIMBA"] != "-")]
    g = gviz.Digraph(engine="dot")
    g.attr(rankdir="TB", overlap="false", splines="true", ranksep="0.55", nodesep="0.4")
    g.attr("node", shape="ellipse", fontsize="12", fontname="Helvetica")
    g.attr("edge", arrowsize="0.85", penwidth="1.5")
    nodes = set()
    for _, r in shared.iterrows():
        for v in (r["var_a"], r["var_b"]):
            if v not in nodes:
                nodes.add(v)
                g.node(v, label=_lab(v))
    for _, r in shared.iterrows():
        a, b = r["var_a"], r["var_b"]
        if r["verdict"] == "identical":
            g.edge(a, b, color="black", **_STYLE.get(r["TNG50"], _STYLE["o-o"]))
        else:  # present in both but orientation differs between the codes
            g.edge(a, b, color="darkorange2", penwidth="2.2",
                   dir="both", arrowtail="normal", arrowhead="normal")
    g.render(filename=out, directory=PLOTS, format="png", cleanup=True)
    n_id = (shared["verdict"] == "identical").sum()
    print(f"agreement B: {len(shared)} shared edges, {n_id} identical (black), "
          f"{len(shared)-n_id} divergent (orange) -> Plots/{out}.png")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    render_simple("matchA_NSA", "cmpA_nsa")
    render_simple("matchA_TNG50", "cmpA_tng50")
    render_simple("matchA_SIMBA", "cmpA_simba")
    render_simple("matchB_TNG50", "cmpB_tng50")
    render_simple("matchB_SIMBA", "cmpB_simba")
    render_agreement_B("cmpB_agreement")


if __name__ == "__main__":
    main()
