"""Render the FCIT graphs with graphviz, in the style of the earlier submission
(ellipse nodes with a plain-English name and the variable name beneath; odot for
undetermined endpoints, normal arrowhead for determined ones).

Outputs Plots/v3/pag_<code>.png for the three z=0 graphs and
Plots/v3/pag_temporal.png for the time-tiered EAGLE graph (epochs as columns).
"""

import os, io, contextlib, pickle
import numpy as np, pandas as pd
import graphviz as gviz
from pytetrad.tools import TetradSearch as ts

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "paper", "Plots", "v3")
os.makedirs(OUT, exist_ok=True)

PLAIN = {
    "DM_MASS": "Halo mass",
    "STELLAR_MASS": "Stellar mass",
    "GAS_MASS": "Gas mass",
    "BH_MASS": "Black-hole mass",
    "LOG_SIGMA": "Velocity dispersion",
    "GAS_METALLICITY": "Gas metallicity",
    "STAR_METALLICITY": "Stellar metallicity",
    "SSFR": "Specific SFR",
}
TECH = {
    "DM_MASS": "M_h",
    "STELLAR_MASS": "M_star",
    "GAS_MASS": "M_gas",
    "BH_MASS": "M_BH",
    "LOG_SIGMA": "sigma_star",
    "GAS_METALLICITY": "Z_gas",
    "STAR_METALLICITY": "Z_star",
    "SSFR": "sSFR",
}
GOLD = "#b8860b"
INK = "#222222"
HALO = "#d55e00"
BHC = "#b8860b"


def style_of(mark, color):
    d = dict(dir="both", color=color, arrowsize="0.9")
    d["arrowtail"] = "odot" if mark[0] == "o" else ("normal" if mark[0] == "<" else "none")
    d["arrowhead"] = "odot" if mark[-1] == "o" else ("normal" if mark[-1] == ">" else "none")
    return d


def label(v):
    return f'<{PLAIN[v]}<BR/><FONT POINT-SIZE="10">{TECH[v]}</FONT>>'


def render_z0():
    with contextlib.redirect_stdout(io.StringIO()):
        import fcit_z0 as F
    for code, E in F.graphs.items():
        g = gviz.Digraph(engine="dot")
        g.attr(
            rankdir="TB",
            splines="true",
            overlap="false",
            ranksep="0.5",
            nodesep="0.3",
            dpi="300",
            ratio="1.0",
            size="6,6",
            ordering="out",
        )
        g.attr(
            "node",
            shape="ellipse",
            fontsize="13",
            fontname="Helvetica",
            color=INK,
            fontcolor=INK,
            penwidth="1.2",
        )
        g.attr("edge", penwidth="1.2", color=INK)
        for v in F.V:
            extra = {}
            g.node(v, label=label(v))
        for a, m, b in sorted(E):
            gold = "BH_MASS" in (a, b) and ({a, b} & {"DM_MASS", "STELLAR_MASS", "LOG_SIGMA"})
            st = style_of(m, INK)
            if gold:
                st["penwidth"] = "2.4"
            g.edge(a, b, **st)
        g.render(filename=f"pag_{code}", directory=OUT, format="png", cleanup=True)
        print("->", os.path.join(OUT, f"pag_{code}.png"), len(E), "edges")


def render_temporal(run="RefL0100N1504", pen=5.0, trunc=7):
    d = pickle.load(open(os.path.join(HERE, "data", f"EAGLE_{run}_temporal.pkl"), "rb"))
    TIERS = [
        (0, ["M200_z2"]),
        (1, ["MSTAR_z2", "MBH_z2"]),
        (2, ["M200_z1"]),
        (3, ["MSTAR_z1", "MBH_z1"]),
        (4, ["M200_z0"]),
        (5, ["MSTAR_z0", "MBH_z0", "SFR_z0", "SIGMA_z0"]),
    ]
    V = [v for _, vs in TIERS for v in vs]
    ok = np.ones(len(d["RootID"]), bool)
    for v in V:
        ok &= np.isfinite(d[v])
    ok &= (d["MSTAR_z0"] > 9.0) & (d["M200_z0"] > 10.5) & d["CENTRAL_z1"] & d["CENTRAL_z2"]
    df = pd.DataFrame({v: d[v][ok] for v in V})
    df = (df - df.mean()) / df.std()
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    for t, vs in TIERS:
        for v in vs:
            s.add_to_tier(t, v)
    HALOS = [v for v in V if v.startswith("M200")]
    for b in V:
        if b not in HALOS:
            for h in HALOS:
                s.set_forbidden(b, h)
    s.use_basis_function_lrt(truncation_limit=trunc, alpha=0.01)
    s.use_basis_function_bic(truncation_limit=trunc, penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = []
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E.append((p[1], p[2], p[3]))
    name = {
        "M200": "Halo mass",
        "MBH": "Black-hole mass",
        "MSTAR": "Stellar mass",
        "SFR": "Star formation",
        "SIGMA": "Velocity dispersion",
    }
    tech = {"M200": "M_h", "MBH": "M_BH", "MSTAR": "M_star", "SFR": "SFR", "SIGMA": "sigma_star"}
    g = gviz.Digraph(engine="dot")
    g.attr(rankdir="LR", splines="true", ranksep="1.4", nodesep="0.45", dpi="300", newrank="true")
    g.attr(
        "node", shape="ellipse", fontsize="13", fontname="Helvetica", color=INK, fontcolor=INK, penwidth="1.2"
    )
    g.attr("edge", penwidth="1.1", color=INK, arrowsize="0.9")
    for z, title in (("z2", "z = 2"), ("z1", "z = 1"), ("z0", "z = 0")):
        with g.subgraph(name=f"cluster_{z}") as c:
            c.attr(
                label=title,
                fontsize="14",
                fontname="Helvetica-Bold",
                color="#bbbbbb",
                style="rounded",
                rank="same",
            )
            for v in V:
                if v.endswith(z):
                    k = v.split("_")[0]
                    extra = {}
                    c.node(
                        v,
                        label=f'<{name[k]}<BR/><FONT POINT-SIZE="10">{tech[k]}({title.replace(" ", "")})</FONT>>',
                        **extra,
                    )
    for a, m, b in sorted(E):
        into_bh0 = (b == "MBH_z0" and m.endswith(">")) or (a == "MBH_z0" and m.startswith("<"))
        from_bh = a.startswith("MBH") and m.endswith(">") and b.split("_")[0] in ("MSTAR", "SFR", "SIGMA")
        gold = into_bh0 or from_bh
        st = style_of(m, INK)
        if gold:
            st["penwidth"] = "2.4"
        g.edge(a, b, **st)
    g.render(filename="pag_temporal", directory=OUT, format="png", cleanup=True)
    print("->", os.path.join(OUT, "pag_temporal.png"), len(E), "edges, N =", len(df))


if __name__ == "__main__":
    render_z0()
    render_temporal()
