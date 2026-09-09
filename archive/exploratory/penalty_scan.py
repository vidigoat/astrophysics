"""Calibrate sparsity: a graph must recover ESTABLISHED relations before any
novel edge in it is believable.  Scan penalty discount and record whether the
anchor relations survive."""

import pickle, numpy as np, pandas as pd, io, contextlib
from pytetrad.tools import TetradSearch as ts

D = "/Users/vidigoat/astrophysics/reanalysis/data/"

# Relations that MUST be recovered for the graph to be trusted.
ANCHORS = {
    "SHMR  M*-Mdm": ("STELLAR_MASS", "DM_MASS"),
    "MZR   M*-Zstar": ("STELLAR_MASS", "STAR_METALLICITY"),
    "FJ    M*-sigma": ("STELLAR_MASS", "VEL_DISP"),
    "Msize M*-Rhalf": ("STELLAR_MASS", "HALFMASS_RAD"),
    "MBH   M*-Mbh": ("STELLAR_MASS", "BH_MASS"),
    "MS    M*-sSFR": ("STELLAR_MASS", "SSFR"),
    "gas   M*-Mgas": ("STELLAR_MASS", "GAS_MASS"),
}


def edges_at(df, pen, trunc=7, alpha=0.01):
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    s.use_basis_function_lrt(truncation_limit=trunc, alpha=alpha)
    s.use_basis_function_bic(truncation_limit=trunc, penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E = {}
    for line in str(s.get_java()).split("\n"):
        p = line.strip().split()
        if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
            E[tuple(sorted([p[1], p[3]]))] = f"{p[1]} {p[2]} {p[3]}"
    return E


rows = []
for tag in ["TNG50", "EAGLE", "SIMBA"]:
    d = pickle.load(open(D + f"{tag}_clean.pkl", "rb"))
    cols = list(d.keys())
    df = pd.DataFrame(np.column_stack([d[c] for c in cols]), columns=cols)
    npos = len(cols) * (len(cols) - 1) // 2
    print(f"\n===== {tag}  N={len(df)}  ({npos} possible edges) =====")
    print(f'{"pen":>5} {"edges":>6} {"dens":>6}  anchors recovered')
    for pen in [1, 2, 3, 5, 8, 12, 15, 20, 30]:
        E = edges_at(df, pen)
        got = [k for k, (a, b) in ANCHORS.items() if tuple(sorted([a, b])) in E]
        rows.append({"sim": tag, "pen": pen, "n_edges": len(E), "n_anchors": len(got)})
        print(
            f'{pen:5d} {len(E):6d} {len(E)/npos:6.2f}  {len(got)}/7  {", ".join(k.split()[0] for k in got)}'
        )
pd.DataFrame(rows).to_csv("/Users/vidigoat/astrophysics/reanalysis/results/penalty_scan.csv", index=False)
