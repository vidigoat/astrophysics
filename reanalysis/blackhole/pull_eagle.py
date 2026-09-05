"""Pull EAGLE z=0 central galaxies with halo-level quantities and the main
progenitor branch, from the public database (McAlpine et al. 2016).

Runs:
  RefL0100N1504     reference physics, 100 Mpc  (the run used in the paper)
  RefL0050N0752     reference physics,  50 Mpc  (control for AGNdT9)
  AGNdT9L0050N0752  AGN heating dT = 10^9 K instead of 10^8.5 K, 50 Mpc

Writes Data/eagle_v3/<run>_z0.csv and <run>_branch.csv (raw), then
reanalysis/v3/data/EAGLE_<run>.pkl with the common variable set plus
  M200, R200, VMAX, CONC_V (= log Vmax/V200), BH_SUBGRID, Z_FORM (z at which the
  main-branch M200 first exceeded half its z=0 value).
"""

import os, sys, io, base64, pickle, urllib.request, urllib.parse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "Data", "eagle_v3")
os.makedirs(RAW, exist_ok=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
USER, PW = open(os.path.join(ROOT, "Data", "virgodb_auth")).read().split()
G_KPC = 4.30091e-6  # kpc (km/s)^2 / Msun

RUNS = ["RefL0100N1504", "RefL0050N0752", "AGNdT9L0050N0752"]


def query(sql):
    url = "http://virgodb.dur.ac.uk:8080/Eagle/?action=doQuery&SQL=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{USER}:{PW}".encode()).decode())
    txt = urllib.request.urlopen(req, timeout=1800).read().decode()
    if not txt.startswith("#OK"):
        raise RuntimeError(txt[:500])
    return pd.read_csv(io.StringIO(txt), comment="#")


Z0_SQL = """
SELECT SH.GalaxyID, SH.GroupID, SH.TopLeafID, SH.MassType_DM, SH.MassType_BH, SH.BlackHoleMass,
       SH.Vmax, SH.Stars_Metallicity, SH.SF_Metallicity, SH.StarFormationRate, SH.MassType_Star,
       AP.Mass_Star, AP.Mass_Gas, AP.VelDisp, AP.SFR,
       FOF.Group_M_Crit200, FOF.Group_R_Crit200
FROM {run}_SubHalo AS SH, {run}_Aperture AS AP, {run}_FOF AS FOF
WHERE SH.SnapNum=28 AND SH.SubGroupNumber=0
  AND SH.GalaxyID=AP.GalaxyID AND AP.ApertureSize=30
  AND SH.GroupID=FOF.GroupID
  AND AP.Mass_Star>1.0e8
"""

BRANCH_SQL = """
SELECT D.GalaxyID AS RootID, P.SnapNum, P.Redshift, FOF.Group_M_Crit200 AS M200, P.MassType_DM, P.BlackHoleMass
FROM {run}_SubHalo AS D, {run}_SubHalo AS P, {run}_Aperture AS AP, {run}_FOF AS FOF
WHERE D.SnapNum=28 AND D.SubGroupNumber=0
  AND D.GalaxyID=AP.GalaxyID AND AP.ApertureSize=30 AND AP.Mass_Star>1.0e8
  AND P.GalaxyID BETWEEN D.GalaxyID AND D.TopLeafID
  AND P.GroupID=FOF.GroupID
"""


def pull(run):
    f0 = os.path.join(RAW, f"{run}_z0.csv")
    fb = os.path.join(RAW, f"{run}_branch.csv")
    if not os.path.exists(f0):
        df = query(Z0_SQL.format(run=run))
        df.to_csv(f0, index=False)
        print(run, "z0 rows", len(df))
    if not os.path.exists(fb):
        # chunk by GalaxyID residue so no single query exceeds the server's 30-min limit
        parts = []
        nchunk = 12 if "L0100" in run else 1
        for k in range(nchunk):
            extra = f" AND (D.GalaxyID % {nchunk}) = {k}" if nchunk > 1 else ""
            parts.append(query(BRANCH_SQL.format(run=run) + extra))
            print(run, "branch chunk", k, len(parts[-1]), flush=True)
        br = pd.concat(parts, ignore_index=True)
        br.to_csv(fb, index=False)
        print(run, "branch rows", len(br))
    return pd.read_csv(f0), pd.read_csv(fb)


def z_form(branch, roots, m200_z0, frac=0.5):
    """Redshift at which the main-branch M200 first exceeds frac*M200(z=0),
    interpolated linearly in log M between the bracketing snapshots."""
    out = np.full(len(roots), np.nan)
    br = branch.sort_values(["RootID", "SnapNum"])
    grp = {k: v for k, v in br.groupby("RootID")}
    for i, (rid, m0) in enumerate(zip(roots, m200_z0)):
        g = grp.get(rid)
        if g is None or len(g) < 3 or not np.isfinite(m0) or m0 <= 0:
            continue
        z = g["Redshift"].to_numpy(float)
        m = g["M200"].to_numpy(float)
        ok = m > 0
        z, m = z[ok], m[ok]
        if len(z) < 3:
            continue
        # walk from high z to low z; find first crossing above frac*m0
        target = np.log10(frac * m0)
        lm = np.log10(m)
        above = lm >= target
        if above.all():
            out[i] = z.max()
            continue
        # last index (in ascending snap order) that is below, then the next is above
        idx = np.where(~above)[0]
        j = idx.max()
        if j + 1 >= len(z):
            out[i] = z[j]
            continue
        z1, z2, l1, l2 = z[j], z[j + 1], lm[j], lm[j + 1]
        out[i] = z1 + (target - l1) * (z2 - z1) / (l2 - l1) if l2 != l1 else z2
    return out


def build(run):
    df, br = pull(run)
    n = len(df)
    m200 = df["Group_M_Crit200"].to_numpy(float)
    r200 = df["Group_R_Crit200"].to_numpy(float)  # Msun, pkpc
    v200 = np.sqrt(G_KPC * m200 / np.maximum(r200, 1e-3))
    vmax = df["Vmax"].to_numpy(float)
    bh_part = df["MassType_BH"].to_numpy(float)
    bh_sub = df["BlackHoleMass"].to_numpy(float)
    zf = z_form(br, df["GalaxyID"].to_numpy(), m200)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = {
            "STELLAR_MASS": np.log10(df["Mass_Star"].to_numpy(float)),
            "GAS_MASS": np.log10(np.maximum(df["Mass_Gas"].to_numpy(float), 1.0)),
            "BH_MASS": np.where(bh_part > 0, np.log10(np.maximum(bh_part, 1.0)), -10.0),
            "BH_SUBGRID": np.where(bh_sub > 0, np.log10(np.maximum(bh_sub, 1.0)), -10.0),
            "DM_MASS": np.log10(df["MassType_DM"].to_numpy(float)),
            "M200": np.log10(m200),
            "CONC_V": np.log10(vmax / v200),
            "VMAX": np.log10(vmax),
            "LOG_SIGMA": np.log10(np.maximum(df["VelDisp"].to_numpy(float), 1e-3)),
            "STAR_METALLICITY": np.log10(np.maximum(df["Stars_Metallicity"].to_numpy(float), 1e-8)),
            "GAS_METALLICITY": np.log10(np.maximum(df["SF_Metallicity"].to_numpy(float), 1e-8)),
            "SSFR": np.log10(df["SFR"].to_numpy(float)) - np.log10(df["Mass_Star"].to_numpy(float)),
            "Z_FORM": zf,
        }
    mask = np.isfinite(d["STELLAR_MASS"]) & np.isfinite(d["DM_MASS"]) & (m200 > 0) & np.isfinite(d["CONC_V"])
    mask &= d["STELLAR_MASS"] > 8.5
    d = {k: v[mask] for k, v in d.items()}
    tag = run
    with open(os.path.join(OUT, f"EAGLE_{tag}.pkl"), "wb") as f:
        pickle.dump(d, f)
    print(
        f'{run}: {n} -> {mask.sum()} centrals;  z_form finite {np.isfinite(d["Z_FORM"]).mean():.2f};  '
        f'BH seeded {(d["BH_MASS"]>-10).mean():.2f}'
    )
    return d


if __name__ == "__main__":
    for run in sys.argv[1:] or RUNS:
        build(run)
