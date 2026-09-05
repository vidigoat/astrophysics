"""Shared statistics for the v3 (A&A) analysis.

Every routine works on plain numpy arrays in log10 units.  Nothing here knows
about a particular simulation; the per-code loaders live in load_*.py.
"""

import numpy as np


def ols(y, *cols, boot=0, seed=0):
    """OLS of y on the columns (plus intercept).  Returns coefficients (without
    intercept) and, if boot>0, their bootstrap standard errors."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    if not boot:
        return b[1:], None
    rng = np.random.default_rng(seed)
    bs = np.empty((boot, X.shape[1]))
    n = len(y)
    for i in range(boot):
        idx = rng.integers(0, n, n)
        bs[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    return b[1:], bs.std(axis=0)[1:]


def resid(v, Z):
    X = np.column_stack([np.ones(len(v))] + [np.asarray(z, float) for z in Z])
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def pcorr(x, y, Z=(), boot=0, seed=0):
    """Partial Pearson correlation of x and y given the list Z."""
    r = np.corrcoef(resid(x, Z), resid(y, Z))[0, 1]
    if not boot:
        return r, None
    rng = np.random.default_rng(seed)
    n = len(x)
    Zs = [np.asarray(z, float) for z in Z]
    vals = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        vals.append(
            np.corrcoef(resid(x[idx], [z[idx] for z in Zs]), resid(y[idx], [z[idx] for z in Zs]))[0, 1]
        )
    return r, float(np.std(vals))


def scatter_reduction(y, base_cols, extra_cols):
    """Percent reduction in residual scatter of y from adding extra_cols to base_cols."""

    def sd(cols):
        return np.std(resid(y, cols))

    a = sd(base_cols)
    b = sd(base_cols + extra_cols)
    return 100.0 * (1.0 - b / a)


def binned(x, f, edges, min_n=30):
    """Apply f to the subsample in each bin of x; returns centres, values, counts."""
    out, cen, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < min_n:
            continue
        out.append(f(m))
        cen.append(0.5 * (lo + hi))
        cnt.append(int(m.sum()))
    return np.array(cen), np.array(out), np.array(cnt)


def window(d, lo=9.5, hi=11.0):
    return (d["STELLAR_MASS"] > lo) & (d["STELLAR_MASS"] < hi)


def describe(tag, d):
    n = len(d["STELLAR_MASS"])
    print(f"{tag}: N={n}")
    for k, v in d.items():
        v = np.asarray(v, float)
        print(
            f"   {k:14s} med={np.nanmedian(v):8.3f}  16-84: {np.nanpercentile(v,16):8.3f} {np.nanpercentile(v,84):8.3f}"
        )
