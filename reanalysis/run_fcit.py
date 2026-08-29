"""Consensus FCIT over repeated runs, on the corrected/uncensored datasets."""
import os, sys, time, pickle, json
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D   = '/Users/vidigoat/astrophysics/reanalysis/data/'
OUT = '/Users/vidigoat/astrophysics/reanalysis/results/'
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.01
TRUNC = 7
PEN   = 15


def one_run(df, seed):
    s = ts.TetradSearch(df)
    s.set_verbose(False)
    try:
        s.set_seed(seed)
    except Exception:
        pass
    s.use_basis_function_lrt(truncation_limit=TRUNC, alpha=ALPHA)
    s.use_basis_function_bic(truncation_limit=TRUNC, penalty_discount=PEN)
    s.run_fcit()
    return str(s.get_java())


def parse(gs):
    edges = []
    for line in gs.split('\n'):
        line = line.strip()
        if not line or line[0].isalpha():
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0].rstrip('.').isdigit():
            a, mark, b = parts[1], parts[2], parts[3]
            if set(mark) <= set('<->o-'):
                edges.append((a, mark, b))
    return edges


def consensus(tag, fname, nruns):
    with open(D + fname, 'rb') as f:
        d = pickle.load(f)
    cols = list(d.keys())
    df = pd.DataFrame(np.column_stack([d[c] for c in cols]), columns=cols)
    print(f'--- {tag}: N={len(df)} vars={len(cols)} runs={nruns}', flush=True)

    pres, marks = {}, {}
    t0 = time.time()
    for i in range(nruns):
        for a, m, b in parse(one_run(df, 1000 + i)):
            key = tuple(sorted([a, b]))
            pres[key] = pres.get(key, 0) + 1
            oriented = f'{a} {m} {b}' if [a, b] == list(key) else f'{b} {m[::-1].translate(str.maketrans("<>", "><"))} {a}'
            marks.setdefault(key, {})
            marks[key][oriented] = marks[key].get(oriented, 0) + 1
        if (i + 1) % 10 == 0:
            print(f'    {i+1}/{nruns}  {time.time()-t0:.0f}s', flush=True)

    rows = []
    for key, c in sorted(pres.items(), key=lambda kv: -kv[1]):
        frac = c / nruns
        if frac < 0.5:
            continue
        best, bc = max(marks[key].items(), key=lambda kv: kv[1])
        rows.append({'a': key[0], 'b': key[1], 'presence': round(frac, 3),
                     'edge': best, 'orient_frac': round(bc / c, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT + f'consensus_{tag}.csv', index=False)
    print(out.to_string(index=False), flush=True)
    print(f'    total {time.time()-t0:.0f}s -> consensus_{tag}.csv\n', flush=True)
    return out


if __name__ == '__main__':
    tag, fname, nruns = sys.argv[1], sys.argv[2], int(sys.argv[3])
    consensus(tag, fname, nruns)
