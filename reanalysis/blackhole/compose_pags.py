"""Render the three z=0 FCIT graphs on ONE fixed node layout, so that the panels can be
compared edge by edge, and compose them into Plots/v3/fig_pags.png.

Edges that touch the black hole are drawn black and heavy; all others recede to grey.
The reader therefore sees, at a glance, which node the black hole is attached to in each code.
"""
import os, io, contextlib
import graphviz as gviz
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "paper", "Plots", "v3")

# one layout for every panel (neato, pinned positions in points)
POS = {                      # inches; identical in every panel
    "DM_MASS":          (2.05, 4.60),
    "STELLAR_MASS":     (0.40, 3.10),
    "LOG_SIGMA":        (3.70, 3.10),
    "BH_MASS":          (2.05, 1.80),
    "GAS_MASS":         (4.55, 1.80),
    "SSFR":             (0.40, 0.50),
    "STAR_METALLICITY": (2.25, 0.50),
    "GAS_METALLICITY":  (4.10, 0.50),
}
NAME = {                     # two-line labels keep the ellipses narrow
    "DM_MASS": "Halo mass", "STELLAR_MASS": "Stellar mass", "GAS_MASS": "Gas mass",
    "BH_MASS": "Black hole", "LOG_SIGMA": "Velocity&#92;ndispersion",
    "GAS_METALLICITY": "Gas&#92;nmetallicity", "STAR_METALLICITY": "Stellar&#92;nmetallicity",
    "SSFR": "Specific&#92;nSFR",
}
INK, GREY = "#000000", "#a0a0a0"


def marks(m, color, heavy):
    d = dict(dir="both", color=color, arrowsize="0.8", penwidth="2.6" if heavy else "1.0")
    d["arrowtail"] = "odot" if m[0] == "o" else ("normal" if m[0] == "<" else "none")
    d["arrowhead"] = "odot" if m[-1] == "o" else ("normal" if m[-1] == ">" else "none")
    return d


def render(code, E):
    g = gviz.Digraph(engine="neato")
    g.attr(splines="true", dpi="200", pad="0.3", outputorder="edgesfirst")
    g.attr("node", shape="ellipse", fontsize="15", fontname="Helvetica",
           color=INK, fontcolor=INK, penwidth="1.3", style="filled", fillcolor="white")
    for v, (x, y) in POS.items():
        g.node(v, label=NAME[v].replace("&#92;n", "\\n"), pos=f"{x},{y}!")
    for a, m, b in sorted(E):                       # grey first, black on top
        if "BH_MASS" not in (a, b):
            g.edge(a, b, **marks(m, GREY, False))
    for a, m, b in sorted(E):
        if "BH_MASS" in (a, b):
            g.edge(a, b, **marks(m, INK, True))
    p = g.render(filename=f"_pag_{code}", directory=OUT, format="png", cleanup=True)
    return p


def compose(paths, titles, out, pad=34, title_h=62):
    ims = [Image.open(p).convert("RGB") for p in paths]
    h = max(i.height for i in ims)
    ims = [i.resize((int(i.width * h / i.height), h), Image.LANCZOS) for i in ims]
    W = sum(i.width for i in ims) + pad * (len(ims) + 1)
    canvas = Image.new("RGB", (W, h + title_h + pad), "white")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(canvas)
    x = pad
    for im, t in zip(ims, titles):
        canvas.paste(im, (x, title_h))
        w = d.textbbox((0, 0), t, font=font)[2]
        d.text((x + im.width // 2 - w // 2, 12), t, fill="black", font=font)
        x += im.width + pad
    canvas.save(out)
    print("->", out, canvas.size)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        import fcit_z0 as F
    order = ["TNG50", "EAGLE", "SIMBA"]
    paths = [render(c, F.graphs[c]) for c in order]
    compose(paths, order, os.path.join(OUT, "fig_pags.png"))
    for p in paths:
        os.remove(p)
