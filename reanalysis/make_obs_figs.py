import re, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_pag import circle_layout, spring_layout, best_circle_layout, draw

SHORT = {
    "ELPETRO_ABSMAG_R": "$M_r$",
    "ELPETRO_MASS": r"$M_\star$",
    "ELPETRO_TH50_R": r"$R_{50}$",
    "ELPETRO_BA": "$b/a$",
    "ELPETRO_B300": "$b_{300}$",
    "ELPETRO_METS": r"$Z_\star$",
    "COLOR_U_R": "$u-r$",
    "SERSIC_N": "$n$",
    "ZDIST": "$z$",
    "logMH": r"$M_{\rm HI}$",
    "W50": "$W_{50}$",
}

txt = open("results/obs_rerun.txt").read()
blocks = re.findall(r"=== (\w+) : WITHOUT derived quantities ===\n(.*?)(?==== |\Z)", txt, re.S)
for tag, body in blocks:
    E = []
    for ln in body.split("\n"):
        p = ln.split()
        if len(p) >= 3 and set(p[1]) <= set("<->o-") and p[0] in SHORT:
            E.append((SHORT.get(p[0], p[0]), p[1], SHORT.get(p[2], p[2])))
    nodes = sorted({v for a, _, b in E for v in (a, b)})
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    pos, nx_ = best_circle_layout(nodes, E)
    draw(ax, E, pos, nodefs=11)
    print("   crossings:", nx_)
    out = "../paper/Plots/%s_pag.png" % tag.lower()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("%-8s %2d edges, %d nodes -> %s" % (tag, len(E), len(nodes), out))
