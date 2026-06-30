"""
Render the simulation-comparison PAGs from Simulation_Comparison.py.

  Comparison A: small PAGs for NSA, TNG50, EAGLE, SIMBA on common observables
                -> Plots/cmpA_{nsa,tng50,eagle,simba}.png
  Comparison B: full intrinsic PAGs for each code
                -> Plots/cmpB_{tng50,eagle,simba}.png
  Invariant backbone: the edges present in ALL THREE codes, coloured by whether
                the codes agree on direction
                -> Plots/cmpB_backbone.png
                  black solid  = same orientation in all three (robust physics)
                  grey         = undirected in all three (robust, direction unresolved)
                  orange       = present in all three but directions DIFFER (model-dependent)
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
    seen = set()
    for _, r in df.iterrows():
        for v in (r["var_a"], r["var_b"]):
            if v not in seen:
                seen.add(v); g.node(v, label=_lab(v))
    for _, r in df.iterrows():
        g.edge(r["var_a"], r["var_b"], **_STYLE.get(r["consensus_mark"], _STYLE["o-o"]))
    g.render(filename=out, directory=PLOTS, format="png", cleanup=True)
    print(f"{name}: {len(df)} edges -> Plots/{out}.png")


def render_backbone(out):
    df = pd.read_csv(os.path.join(RESULTS, "comparison_B.csv"))
    inv = df[df["edge_class"].str.startswith("invariant")]
    g = gviz.Digraph(engine="dot")
    g.attr(rankdir="TB", overlap="false", splines="true", ranksep="0.55", nodesep="0.4")
    g.attr("node", shape="ellipse", fontsize="12", fontname="Helvetica")
    g.attr("edge", arrowsize="0.85", penwidth="1.6")
    seen = set()
    for _, r in inv.iterrows():
        for v in (r["var_a"], r["var_b"]):
            if v not in seen:
                seen.add(v); g.node(v, label=_lab(v))
    for _, r in inv.iterrows():
        cls = r["edge_class"]
        if cls == "invariant_oriented":
            g.edge(r["var_a"], r["var_b"], color="black", **_STYLE.get(r["TNG50"], _STYLE["o-o"]))
        elif cls == "invariant_undirected":
            g.edge(r["var_a"], r["var_b"], color="gray55", dir="none")
        else:  # invariant_skeleton: codes disagree on direction
            g.edge(r["var_a"], r["var_b"], color="darkorange2", penwidth="2.2",
                   dir="both", arrowtail="normal", arrowhead="normal")
    g.render(filename=out, directory=PLOTS, format="png", cleanup=True)
    n_o = (inv["edge_class"] == "invariant_oriented").sum()
    n_u = (inv["edge_class"] == "invariant_undirected").sum()
    n_s = (inv["edge_class"] == "invariant_conflict").sum()
    print(f"backbone: {len(inv)} edges in all 3 codes -> "
          f"{n_o} same-direction (black), {n_u} undirected (grey), {n_s} dir-conflict (orange)")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for c in ("NSA", "TNG50", "EAGLE", "SIMBA"):
        render_simple(f"matchA_{c}", f"cmpA_{c.lower()}")
    for c in ("TNG50", "EAGLE", "SIMBA"):
        render_simple(f"matchB_{c}", f"cmpB_{c.lower()}")
    render_backbone("cmpB_backbone")


if __name__ == "__main__":
    main()
