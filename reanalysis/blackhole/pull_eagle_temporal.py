"""Main-progenitor-branch quantities for EAGLE centrals at selected snapshots, so
that a temporal (time-tiered) causal test can be run:  does the halo at z=1 or z=2
predict the black hole at z=0 once the galaxy at z=0 is known?

SnapNum: 28 = z 0.00, 19 = z 1.00, 15 = z 2.01, 12 = z 3.02.
Writes reanalysis/v3/data/EAGLE_<run>_temporal.pkl with columns like M200_z1, MSTAR_z1, MBH_z1.
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
SNAPS = {28: "z0", 19: "z1", 15: "z2", 12: "z3"}


def query(sql):
    url = "http://virgodb.dur.ac.uk:8080/Eagle/?action=doQuery&SQL=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{USER}:{PW}".encode()).decode())
    txt = urllib.request.urlopen(req, timeout=1800).read().decode()
    if not txt.startswith("#OK"):
        raise RuntimeError(txt[:500])
    return pd.read_csv(io.StringIO(txt), comment="#")


SQL = """
SELECT D.GalaxyID AS RootID, P.SnapNum, P.GalaxyID AS ProgID, P.MassType_DM, P.BlackHoleMass, P.MassType_BH,
       P.Vmax, P.StarFormationRate, P.MassType_Star, P.SubGroupNumber,
       AP.Mass_Star, AP.Mass_Gas, AP.SFR, AP.VelDisp,
       FOF.Group_M_Crit200, FOF.Group_R_Crit200
FROM {run}_SubHalo AS D, {run}_SubHalo AS P, {run}_Aperture AS AP, {run}_Aperture AS AP0, {run}_FOF AS FOF
WHERE D.SnapNum=28 AND D.SubGroupNumber=0
  AND D.GalaxyID=AP0.GalaxyID AND AP0.ApertureSize=30 AND AP0.Mass_Star>1.0e9
  AND P.GalaxyID BETWEEN D.GalaxyID AND D.TopLeafID
  AND P.SnapNum IN (28,19,15,12)
  AND P.GalaxyID=AP.GalaxyID AND AP.ApertureSize=30
  AND P.GroupID=FOF.GroupID
"""


def build(run):
    f = os.path.join(RAW, f"{run}_temporal.csv")
    if not os.path.exists(f):
        df = query(SQL.format(run=run))
        df.to_csv(f, index=False)
        print(run, "rows", len(df))
    df = pd.read_csv(f)
    # one row per (root, snap); keep the most massive progenitor if duplicates
    df = df.sort_values(["RootID", "SnapNum", "MassType_DM"], ascending=[True, True, False]).drop_duplicates(
        ["RootID", "SnapNum"]
    )
    piv = {}
    roots = np.sort(df["RootID"].unique())
    idx = {r: i for i, r in enumerate(roots)}
    n = len(roots)
    for snap, tag in SNAPS.items():
        sub = df[df["SnapNum"] == snap]
        ii = np.array([idx[r] for r in sub["RootID"]])

        def col(name, fill=np.nan):
            a = np.full(n, fill)
            a[ii] = sub[name].to_numpy(float)
            return a

        with np.errstate(divide="ignore", invalid="ignore"):
            piv[f"M200_{tag}"] = np.log10(col("Group_M_Crit200"))
            piv[f"MDM_{tag}"] = np.log10(col("MassType_DM"))
            piv[f"MSTAR_{tag}"] = np.log10(col("Mass_Star"))
            piv[f"MGAS_{tag}"] = np.log10(np.maximum(col("Mass_Gas"), 1.0))
            bh = col("BlackHoleMass")
            piv[f"MBH_{tag}"] = np.where(bh > 0, np.log10(np.maximum(bh, 1.0)), np.nan)
            bhp = col("MassType_BH")
            piv[f"MBHP_{tag}"] = np.where(bhp > 0, np.log10(np.maximum(bhp, 1.0)), np.nan)
            piv[f"SFR_{tag}"] = np.log10(np.maximum(col("SFR"), 1e-4))
            piv[f"SIGMA_{tag}"] = np.log10(np.maximum(col("VelDisp"), 1e-3))
            piv[f"VMAX_{tag}"] = np.log10(col("Vmax"))
            piv[f"CENTRAL_{tag}"] = col("SubGroupNumber") == 0
    piv["RootID"] = roots
    with open(os.path.join(OUT, f"EAGLE_{run}_temporal.pkl"), "wb") as fh:
        pickle.dump(piv, fh)
    have = {tag: int(np.isfinite(piv[f"M200_{tag}"]).sum()) for tag in SNAPS.values()}
    print(run, "roots", n, "with progenitor at", have)


if __name__ == "__main__":
    for run in sys.argv[1:] or ["RefL0050N0752"]:
        build(run)
