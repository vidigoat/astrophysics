"""
Render the consensus PAGs (from Code/Analysis/Consensus_Stability.py) to PNGs.

Directed consensus edges (a single orientation in >= ORIENT_THR of runs) are
drawn with an arrowhead; consensus-undirected edges are drawn as plain lines.
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
        g.attr("node", shape="ellipse", fontsize="11", fontname="Helvetica")
        g.attr("edge", arrowsize="0.8")
        n_dir = 0
        for _, r in df.iterrows():
            c = str(r["consensus"])
            if " --> " in c:
                a, b = [s.strip() for s in c.split(" --> ")]
                g.edge(a, b, dir="forward")
                n_dir += 1
            else:  # consensus-undirected "o-o"
                a, b = [s.strip() for s in c.split(" o-o ")]
                g.edge(a, b, dir="none")
        g.render(filename=base, directory=PLOTS, format="png", cleanup=True)
        print(f"{name}: {len(df)} edges ({n_dir} directed) -> Plots/{base}.png")


if __name__ == "__main__":
    main()
