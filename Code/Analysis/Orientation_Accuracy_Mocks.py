"""
LOAD-BEARING TEST (audit critical #1): does FCIT orient edges CORRECTLY vs a known
ground truth, not merely self-consistently across codes?

Uses the 100 CPN mock datasets (each with a known true DAG) at the exact comparison
settings (truncation 7, penalty 8, consensus over NRUN runs, edge kept >=50%,
oriented >=60%). For each mock, among the edges that are TRUE adjacencies AND that
the consensus graph orients (commits to a determined direction), we score the
fraction whose direction matches the true DAG.  This is orientation ACCURACY vs
ground truth --- the thing cross-code agreement (0.78/0.91) does NOT establish.

Reports pooled accuracy + per-mock mean with a bootstrap 95% CI, plus skeleton F1
for continuity.  Output: Results/orientation_accuracy_mocks.csv + printed summary.
"""
import os, pickle, re
from collections import Counter
import numpy as np, pandas as pd
import pytetrad.tools.TetradSearch as ts

import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
MOCKS = pickle.load(open(f"{REPO}/Data/MockDatasets/mock_datasets.pkl", "rb"))
T, P, ALPHA, NRUN, PRES, ORI = 7, 8, 0.01, 15, 0.50, 0.60
N_MOCKS = 100
_M = ("<->", "-->", "<--", "o->", "<-o", "o-o")
_FL = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "o-o": "o-o", "<->": "<->"}
_dir = lambda m: "fwd" if m in ("-->", "o->") else "rev" if m in ("<--", "<-o") else "und"


def zc(x):
    x = np.asarray(x, float); s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)


def parse(g):
    edges, inb = [], False
    for raw in g.split("\n"):
        s = raw.strip()
        if s.startswith("Graph Edges"):
            inb = True; continue
        if not inb:
            continue
        if s.startswith("Graph "):
            break
        for mk in _M:
            if mk in s:
                body = s.split(".", 1)[1].strip() if "." in s.split(mk)[0] else s
                l, r = body.split(mk, 1); a, b = l.strip(), r.strip()
                edges.append(((a, b), mk) if a <= b else ((b, a), _FL[mk])); break
    return edges


def consensus(df):
    pres, marks = Counter(), {}
    for _ in range(NRUN):
        s = ts.TetradSearch(df); s.set_verbose(False)
        s.use_basis_function_lrt(truncation_limit=T, alpha=ALPHA)
        s.use_basis_function_bic(truncation_limit=T, penalty_discount=P)
        s.run_fcit()
        for key, cm in parse(str(s.get_java())):
            pres[key] += 1
            marks.setdefault(key, Counter())[cm] += 1
    g = {}
    for key, c in pres.items():
        if c / NRUN < PRES:
            continue
        top, tn = marks[key].most_common(1)[0]
        g[key] = top if tn / c >= ORI else "o-o"
    return g


def true_maps(true_edges):
    """canonical skeleton set + direction dict from the true DAG edge set (parent,child)."""
    skel, tdir = set(), {}
    for (u, v) in true_edges:
        key = (u, v) if u <= v else (v, u)
        skel.add(key)
        tdir[key] = "fwd" if key == (u, v) else "rev"   # fwd = canonical a is the parent
    return skel, tdir


def main():
    rows = []
    for i, mock in enumerate(MOCKS[:N_MOCKS]):
        df = mock["data"].copy()
        df = pd.DataFrame({c: zc(df[c].values) for c in df.columns})
        g = consensus(df)
        skel, tdir = true_maps(mock["true_edges"])
        pred_skel = set(g.keys())
        # skeleton F1
        tp = len(skel & pred_skel); fp = len(pred_skel - skel); fn = len(skel - pred_skel)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # ORIENTATION ACCURACY: of true-adjacency edges the consensus ORIENTS, fraction correct
        oriented = [(k, g[k]) for k in (skel & pred_skel) if _dir(g[k]) != "und"]
        n_oriented = len(oriented)
        n_correct = sum(1 for k, m in oriented if _dir(m) == tdir[k])
        rows.append(dict(mock=i, n_true=len(skel), n_pred=len(pred_skel), skel_f1=round(f1, 3),
                         n_oriented=n_oriented, n_correct=n_correct,
                         orient_acc=(n_correct / n_oriented) if n_oriented else np.nan))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_MOCKS} mocks done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{REPO}/Results/orientation_accuracy_mocks.csv", index=False)
    # pooled = total correct / total oriented across all mocks
    tot_or = int(d["n_oriented"].sum()); tot_co = int(d["n_correct"].sum())
    pooled = tot_co / tot_or if tot_or else float("nan")
    # per-mock mean + bootstrap CI (mocks with >=1 oriented edge)
    per = d["orient_acc"].dropna().values
    rs = np.random.RandomState(7)
    boot = [np.mean(per[rs.randint(0, len(per), len(per))]) for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n" + "=" * 60)
    print(f"ORIENTATION ACCURACY vs known DAG (t=7, p=8, consensus>=60%)")
    print(f"  POOLED: {tot_co}/{tot_or} = {pooled:.3f}   (of all true-edge orientations, fraction correct)")
    print(f"  per-mock mean: {np.mean(per):.3f}  [95% CI {lo:.3f}-{hi:.3f}]  over {len(per)} mocks")
    print(f"  chance for a binary orientation = 0.50")
    print(f"  mean skeleton F1 at p=8: {d['skel_f1'].mean():.3f}")
    print(f"  mean oriented true-edges per mock: {d['n_oriented'].mean():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
