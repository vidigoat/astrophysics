"""Map EVERY triple in each simulation as chain or fork, using the scatter identity
alone.  For X -> Y -> Z (chain, Y mediates):
      sigma^2(Z|X) = a^2 sigma^2(Y|X) + sigma^2(Z|Y),   a = dZ/dY
Ratio  R = sigma(Z|X)_obs / sigma(Z|X)_chain.
  R ~ 1  : chain, Y fully mediates X's influence on Z
  R < 1  : X predicts Z BETTER than mediation allows -> direct path / common cause (FORK)
  R > 1  : extra variance, Y is not on the path at all
The deficit 1-R^2 measures the fraction of Z's information that bypasses Y."""

import pickle, numpy as np, itertools

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
rng = np.random.default_rng(909)
# only definitionally comparable variables
V = ["STELLAR_MASS", "DM_MASS", "BH_MASS", "STAR_METALLICITY", "GAS_METALLICITY", "SSFR"]
N = {
    "STELLAR_MASS": "M*",
    "DM_MASS": "Mh",
    "BH_MASS": "Mbh",
    "STAR_METALLICITY": "Z*",
    "GAS_METALLICITY": "Zg",
    "SSFR": "sSFR",
}


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    return {k: v[m] for k, v in d.items() if k in V}


def sd(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.std(y - A @ b))


def sl(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b[1]


def R(x, y, z):
    a = sl(y, z)
    den = np.sqrt(a * a * sd(y, [x]) ** 2 + sd(z, [y]) ** 2)
    return sd(z, [x]) / den if den > 0 else np.nan


data = {t: load(t) for t in ["TNG50", "EAGLE", "SIMBA"]}
print("R = observed sigma(Z|X) / chain-predicted.   R~1 chain (Y mediates), R<1 bypass.")
print(f"N: " + ", ".join(f'{t}={len(d["STELLAR_MASS"])}' for t, d in data.items()))
print()
rows = []
for x, y, z in itertools.permutations(V, 3):
    if (x, z) != tuple(sorted([x, z])):
        continue  # avoid double counting X<->Z
    vals, errs = [], []
    for t in ["TNG50", "EAGLE", "SIMBA"]:
        d = data[t]
        n = len(d[x])
        vals.append(R(d[x], d[y], d[z]))
        errs.append(
            np.std(
                [R(d[x][i], d[y][i], d[z][i]) for i in (rng.choice(n, n, replace=True) for _ in range(150))]
            )
        )
    if any(not np.isfinite(v) for v in vals):
        continue
    # interesting = codes DISAGREE about whether Y mediates
    med = [abs(v - 1) < 3 * e or abs(v - 1) < 0.06 for v, e in zip(vals, errs)]
    disagree = len(set(med)) > 1
    rows.append((max(vals) - min(vals), x, y, z, vals, errs, med, disagree))

print(f'{"X -> Y -> Z":26s}' + "".join(f"{t:>18s}" for t in ["TNG50", "EAGLE", "SIMBA"]) + "  mediates?")
print("-" * 90)
for spread, x, y, z, vals, errs, med, dis in sorted(rows, reverse=True)[:18]:
    tag = "".join("Y" if m else "." for m in med)
    star = " <== SPLIT" if dis and spread > 0.15 else ""
    print(
        f'{N[x]+" -> "+N[y]+" -> "+N[z]:26s}'
        + "".join(f"{v:>12.3f}+-{e:4.3f}" for v, e in zip(vals, errs))
        + f"  {tag}{star}"
    )

print()
print("=" * 90)
print("TRIPLES WHERE THE CODES DISAGREE ABOUT WHETHER THE MIDDLE VARIABLE MEDIATES")
print("(one code says the path runs through Y, another says it bypasses Y entirely)")
print("=" * 90)
for spread, x, y, z, vals, errs, med, dis in sorted(rows, reverse=True):
    if not (dis and spread > 0.20):
        continue
    print(
        f"  {N[x]} -> {N[y]} -> {N[z]:6s}  "
        + "  ".join(
            f'{t}:{v:.2f}{"(chain)" if m else "(bypass)"}'
            for t, v, m in zip(["TNG", "EAG", "SIM"], vals, med)
        )
    )
