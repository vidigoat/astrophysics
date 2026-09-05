"""Does EAGLE's ratio 0.648 equal what a PURE FORK predicts from its own parameters?
Pure fork:  M* = b*Mh + e1  (var s1^2),   M_BH = c*Mh + e2  (var s2^2),  e1 _||_ e2
Then everything below is algebra, no free parameters:
   a   = cov(MBH,M*)/var(M*) = c*b*V / (b^2 V + s1^2)
   var(MBH|M*) = s2^2 + c^2 V - (c b V)^2/(b^2 V + s1^2)
   R_fork = s2 / sqrt(a^2 s1^2 + var(MBH|M*))
Pure chain: M_BH depends on M_h ONLY through M*, which forces R = 1 identically."""

import pickle, numpy as np

D = "/Users/vidigoat/astrophysics/reanalysis/data/"
rng = np.random.default_rng(1234)


def load(t):
    d = dict(pickle.load(open(D + f"{t}_clean.pkl", "rb")))
    m = (d["STELLAR_MASS"] > 9.5) & (d["STELLAR_MASS"] < 11.0)
    return {k: v[m] for k, v in d.items()}


def sd(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.std(y - A @ b))


def sl(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b[1]


print("=" * 80)
print("Is the measured ratio consistent with a PURE FORK built from that code's own")
print("parameters?  R_fork is fully determined -- zero free parameters.")
print("=" * 80)
print(
    f'{"":8s}{"b":>8s}{"c":>8s}{"s1":>8s}{"s2":>8s}{"R_fork pred":>14s}{"R_chain":>10s}{"R measured":>14s}{"which":>9s}'
)
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    mh, ms, bh = d["DM_MASS"], d["STELLAR_MASS"], d["BH_MASS"]
    V = np.var(mh)
    b_ = sl(mh, ms)
    c_ = sl(mh, bh)
    s1 = sd(ms, [mh])
    s2 = sd(bh, [mh])
    a_fork = c_ * b_ * V / (b_ * b_ * V + s1 * s1)
    var_bh_given_ms = s2 * s2 + c_ * c_ * V - (c_ * b_ * V) ** 2 / (b_ * b_ * V + s1 * s1)
    R_fork = s2 / np.sqrt(a_fork**2 * s1 * s1 + max(var_bh_given_ms, 1e-12))
    a_meas = sl(ms, bh)
    R_meas = sd(bh, [mh]) / np.sqrt(a_meas**2 * sd(ms, [mh]) ** 2 + sd(bh, [ms]) ** 2)
    n = len(mh)
    bs = []
    for _ in range(400):
        i = rng.choice(n, n, replace=True)
        am = sl(ms[i], bh[i])
        bs.append(sd(bh[i], [mh[i]]) / np.sqrt(am**2 * sd(ms[i], [mh[i]]) ** 2 + sd(bh[i], [ms[i]]) ** 2))
    e = np.std(bs)
    which = "FORK" if abs(R_meas - R_fork) < abs(R_meas - 1.0) else "CHAIN"
    print(
        f"{t:8s}{b_:>8.3f}{c_:>8.3f}{s1:>8.3f}{s2:>8.3f}{R_fork:>14.3f}{1.0:>10.3f}{R_meas:>10.3f}+-{e:4.3f}{which:>9s}"
    )

print()
print("Distance of the measurement from each hypothesis, in sigma:")
print(f'{"":8s}{"|R-1|/err (chain)":>22s}{"|R-Rfork|/err (fork)":>24s}{"verdict":>16s}')
for t in ["TNG50", "EAGLE", "SIMBA"]:
    d = load(t)
    mh, ms, bh = d["DM_MASS"], d["STELLAR_MASS"], d["BH_MASS"]
    V = np.var(mh)
    b_ = sl(mh, ms)
    c_ = sl(mh, bh)
    s1 = sd(ms, [mh])
    s2 = sd(bh, [mh])
    a_fork = c_ * b_ * V / (b_ * b_ * V + s1 * s1)
    vg = s2 * s2 + c_ * c_ * V - (c_ * b_ * V) ** 2 / (b_ * b_ * V + s1 * s1)
    R_fork = s2 / np.sqrt(a_fork**2 * s1 * s1 + max(vg, 1e-12))
    a_meas = sl(ms, bh)
    R_meas = sd(bh, [mh]) / np.sqrt(a_meas**2 * sd(ms, [mh]) ** 2 + sd(bh, [ms]) ** 2)
    n = len(mh)
    bs = []
    for _ in range(400):
        i = rng.choice(n, n, replace=True)
        am = sl(ms[i], bh[i])
        bs.append(sd(bh[i], [mh[i]]) / np.sqrt(am**2 * sd(ms[i], [mh[i]]) ** 2 + sd(bh[i], [ms[i]]) ** 2))
    e = np.std(bs)
    d1, d2 = abs(R_meas - 1) / e, abs(R_meas - R_fork) / e
    v = "pure CHAIN" if d1 < 3 else ("pure FORK" if d2 < 3 else "intermediate")
    print(f"{t:8s}{d1:>22.1f}{d2:>24.1f}{v:>16s}")
