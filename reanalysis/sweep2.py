"""Clean sweep. Drop TNG50's total-matter half-mass radius (definitionally
inconsistent with the stellar radii used by EAGLE/SIMBA and near-collinear with
halo mass). Report condition number, and bootstrap every partial correlation."""

import pickle, numpy as np, itertools

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
LO, HI = 9.5, 11.0
V = [
    "STELLAR_MASS",
    "GAS_MASS",
    "DM_MASS",
    "BH_MASS",
    "LOG_SIGMA",
    "STAR_METALLICITY",
    "GAS_METALLICITY",
    "SSFR",
]
NICE = {
    "STELLAR_MASS": "M*",
    "GAS_MASS": "Mgas",
    "DM_MASS": "Mhalo",
    "BH_MASS": "Mbh",
    "LOG_SIGMA": "sigma",
    "STAR_METALLICITY": "Zstar",
    "GAS_METALLICITY": "Zgas",
    "SSFR": "sSFR",
}
rng = np.random.default_rng(101)


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    d["LOG_SIGMA"] = np.log10(np.maximum(d["VEL_DISP"], 1e-3))
    m = (d["STELLAR_MASS"] > LO) & (d["STELLAR_MASS"] < HI)
    return {k: v[m] for k, v in d.items() if k in V}


def pc(x, y, Z):
    def r(v):
        A = np.column_stack([np.ones(len(v))] + list(Z))
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b

    return float(np.corrcoef(r(x), r(y))[0, 1])


data = {t: load(t) for t in ["TNG50", "EAGLE", "SIMBA"]}
print("Condition number of the standardised design matrix (>30 = unstable partials):")
for t, d in data.items():
    X = np.column_stack([(d[k] - d[k].mean()) / d[k].std() for k in V])
    print(f'   {t:7s} cond = {np.linalg.cond(X):8.1f}   N={len(d["STELLAR_MASS"])}')
print()
print(f'{"pair":18s}' + "".join(f"{t:>24s}" for t in ["TNG50", "EAGLE", "SIMBA"]))
print("-" * 90)
flips = []
rows = []
for a, b in itertools.combinations(V, 2):
    out = {}
    for t, d in data.items():
        rest = [d[k] for k in V if k not in (a, b)]
        v = pc(d[a], d[b], rest)
        n = len(d[a])
        bs = []
        for _ in range(300):
            i = rng.choice(n, n, replace=True)
            bs.append(pc(d[a][i], d[b][i], [r[i] for r in rest]))
        out[t] = (v, np.std(bs))
    vals = [out[t][0] for t in ["TNG50", "EAGLE", "SIMBA"]]
    errs = [out[t][1] for t in ["TNG50", "EAGLE", "SIMBA"]]
    if max(abs(v) for v in vals) < 0.20:
        continue
    # a flip needs BOTH signs significant at 3 sigma
    sig = [(v, e) for v, e in zip(vals, errs) if abs(v) > 3 * e and abs(v) > 0.15]
    sgn = set(np.sign(v) for v, e in sig)
    flag = (
        "SIGN FLIP (3sig)"
        if len(sgn) > 1
        else ("universal" if all(abs(v) > 0.25 for v in vals) and len(set(np.sign(vals))) == 1 else "")
    )
    if flag.startswith("SIGN"):
        flips.append((a, b, vals, errs))
    rows.append((max(abs(v) for v in vals), a, b, vals, errs, flag))
for _, a, b, vals, errs, flag in sorted(rows, reverse=True):
    print(
        f'{NICE[a]+"-"+NICE[b]:18s}'
        + "".join(f"{v:>16.3f} +-{e:5.3f}" for v, e in zip(vals, errs))
        + f"  {flag}"
    )
print()
print("=" * 80)
print("SIGN FLIPS surviving 3-sigma on BOTH sides:")
for a, b, vals, errs in flips:
    print(
        f"  {NICE[a]:7s}-{NICE[b]:7s}  "
        + "  ".join(f"{t}={v:+.3f}+-{e:.3f}" for t, v, e in zip(["TNG50", "EAGLE", "SIMBA"], vals, errs))
    )
