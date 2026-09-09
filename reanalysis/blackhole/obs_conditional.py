"""The conditional test on real galaxies, and the forward model of each simulation
through the real selection (Sect. 6 of the paper; Table 3).

Samples
  Marasco et al. (2021), 55 galaxies: dynamical M_BH, halo masses from extended rotation
    curves, tracer kinematics, Schwarzschild models or X-rays, stellar mass from their
    tabulated stellar fraction.
  Gaspari et al. (2019), 85 early types with X-ray atmospheres: M500 from their R500,
    converted to M200 with a factor 1.4; K-band luminosity with M/L_K = 0.75 as M_star.

Both tables were parsed from the arXiv sources into data_obs/*.csv (see README).

Outputs results/obs_conditional.csv and prints every number quoted in Sect. 6.
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ols, pcorr, resid
from headline import tng50, eagle, simba

HERE = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(HERE, "data_obs")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
M500_TO_M200 = 1.4          # standard NFW conversion at these concentrations
ML_K = 0.75                 # K-band mass-to-light ratio


def marasco():
    R = list(csv.DictReader(open(os.path.join(OBS, "marasco2021.csv"))))
    g = lambda k: np.array([float(r[k]) for r in R])
    return dict(bh=g("logMBH"), ms=g("logMstar"), mh=g("logMh"), e_bh=g("e_logMBH"),
                e_mh=g("e_logMh"), late=g("ttype") > 0, name="Marasco et al. (2021)")


def gaspari(m200=True):
    G = list(csv.DictReader(open(os.path.join(OBS, "gaspari2019.csv"))))
    g = lambda k: np.array([float(r[k]) for r in G])
    mh = g("logM500") + (np.log10(M500_TO_M200) if m200 else 0.0)
    bcg = np.array([r["central"].startswith(("BCG", "BGG")) for r in G])
    return dict(bh=g("logMBH"), ms=g("logLK") + np.log10(ML_K), mh=mh, sig=g("logsig"),
                mbulge=g("logMbulge"), e_bh=g("e_logMBH"),
                e_mh=np.maximum(1.5 * g("e_logTxc"), 0.10), bcg=bcg, name="Gaspari et al. (2019)")


def chain_ratio(bh, ms, mh):
    """sigma^2(BH|Mh) vs the value a chain Mh -> M* -> BH predicts (Wright 1921)."""
    a = np.polyfit(ms, bh, 1)[0]
    return np.var(resid(bh, [mh])) / (a ** 2 * np.var(resid(ms, [mh])) + np.var(resid(bh, [ms])))


def fit(lab, bh, ms, mh, rows):
    (a, b), (ea, eb) = ols(bh, ms, mh, boot=2000, seed=1)
    rh = pcorr(bh, mh, [ms])[0]; rs = pcorr(bh, ms, [mh])[0]
    R = chain_ratio(bh, ms, mh)
    print(f"{lab:34s} N={len(bh):3d}  a={a:+.2f}+-{ea:.2f}  b={b:+.2f}+-{eb:.2f}  "
          f"r(BH,Mh|M*)={rh:+.2f}  r(BH,M*|Mh)={rs:+.2f}  R={R:.2f}")
    rows.append(dict(sample=lab, N=len(bh), a=a, e_a=ea, b=b, e_b=eb, r_halo=rh, r_star=rs, chain_R=R))
    return a, b


def forward(obs, sims, tol=0.15, n=3000, seed=7):
    """Push each simulation through the real selection.

    For every real galaxy, simulated centrals within `tol` dex of its stellar mass form a
    pool; one is drawn, the galaxy's own quoted errors are added to log M_BH and log M_h,
    and the conditional fit is repeated.  Real galaxies with no analogue in a given box are
    dropped for that code, and the observed exponent is recomputed on the same subset, so
    the comparison is like for like.  Returns the median forward exponent, its 16-84 range,
    and the fraction of draws that reach the observed value.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for code, d in sims.items():
        MS, MH, BH = d["MS"], d["MH"], d["BH"]
        pool = [np.where(np.abs(MS - m) < tol)[0] for m in obs["ms"]]
        keep = np.array([len(p) > 0 for p in pool])
        pool = [p for p, k in zip(pool, keep) if k]
        ms_o, e_bh, e_mh = obs["ms"][keep], obs["e_bh"][keep], obs["e_mh"][keep]
        b_obs = ols(obs["bh"][keep], ms_o, obs["mh"][keep])[0][1]
        bs = np.empty(n); bs0 = np.empty(n)
        for i in range(n):
            k = np.array([p[rng.integers(len(p))] for p in pool])
            bs0[i] = ols(BH[k], MS[k], MH[k])[0][1]                      # no errors: intrinsic at these masses
            bs[i] = ols(BH[k] + rng.normal(0, e_bh), MS[k], MH[k] + rng.normal(0, e_mh))[0][1]
        p = float(np.mean(bs >= b_obs))
        lo, hi = np.percentile(bs, [16, 84])
        out[code] = dict(N_used=int(keep.sum()), N_total=int(len(keep)), b_obs=b_obs,
                         b_intrinsic_at_these_masses=float(np.median(bs0)),
                         b_fwd=float(np.median(bs)), lo=lo, hi=hi, P=p)
        drop = "" if keep.all() else f"  ({(~keep).sum()} of {len(keep)} real galaxies have no analogue in this box)"
        print(f"    {code:6s} N={keep.sum():3d}  b_obs={b_obs:+.2f} | sim at these masses: "
              f"b_intrinsic={np.median(bs0):+.2f} -> b_with_errors={np.median(bs):+.2f} "
              f"[{lo:+.2f},{hi:+.2f}]  P={'<3e-4' if p==0 else f'{p:.4f}'}{drop}")
    return out


if __name__ == "__main__":
    rows = []
    M, G = marasco(), gaspari()
    print("=== conditional fits, log M_BH = a log M* + b log M_h + c ===")
    fit("Marasco, all", M["bh"], M["ms"], M["mh"], rows)
    fit("Marasco, early types", M["bh"][~M["late"]], M["ms"][~M["late"]], M["mh"][~M["late"]], rows)
    fit("Marasco, late types", M["bh"][M["late"]], M["ms"][M["late"]], M["mh"][M["late"]], rows)
    fit("Gaspari, all", G["bh"], G["ms"], G["mh"], rows)
    k = ~G["bcg"]
    fit("Gaspari, non-central", G["bh"][k], G["ms"][k], G["mh"][k], rows)
    fit("Gaspari, bulge mass as M*", G["bh"], G["mbulge"], G["mh"], rows)

    print("\n=== with the velocity dispersion also held fixed (Gaspari) ===")
    (a, s, b), _ = ols(G["bh"], G["ms"], G["sig"], G["mh"])
    print(f"  M_BH ~ M*^{a:+.2f} sigma^{s:+.2f} M_h^{b:+.2f}   "
          f"r(BH,Mh|M*,sig)={pcorr(G['bh'],G['mh'],[G['ms'],G['sig']])[0]:+.2f}  "
          f"r(BH,sig|M*,Mh)={pcorr(G['bh'],G['sig'],[G['ms'],G['mh']])[0]:+.2f}")
    sims = {"TNG50": tng50(), "EAGLE": eagle(), "SIMBA": simba()}
    for code, d in sims.items():
        m = (d["MS"] > 9.5) & (d["MS"] < 11.0) & (d["MH"] > 10.5)
        print(f"  same statistic in {code:6s}: r(BH,Mh|M*,sig) = "
              f"{pcorr(d['BH'][m], d['MH'][m], [d['MS'][m], d['SIG'][m]])[0]:+.2f}")

    print("\n=== forward model: each simulation through the real selection and errors ===")
    Gn = {k2: (v[k] if isinstance(v, np.ndarray) and v.shape == G["bh"].shape else v) for k2, v in G.items()}
    fwd_rows = []
    for lab, obs in [("Marasco", M), ("Gaspari non-central", Gn)]:
        print(f"  vs {lab}:")
        for code, r in forward(obs, sims).items():
            fwd_rows.append(dict(sample=lab, code=code, **r))
    with open(os.path.join(OUT, "obs_forward.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fwd_rows[0])); w.writeheader(); w.writerows(fwd_rows)
    print(f"-> {os.path.join(OUT,'obs_forward.csv')}")

    with open(os.path.join(OUT, "obs_conditional.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n-> {os.path.join(OUT,'obs_conditional.csv')}")
