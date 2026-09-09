"""Snapshot-complete EAGLE central catalogues at z = 0.5, 1, 2, 3 (RefL0100N1504), for the
redshift-evolution test of the conditional relation log M_BH | (log M*, log M200).
Writes Data/eagle_v3/RefL0100N1504_snap<N>.csv."""
import os, sys, io, base64, urllib.request, urllib.parse
import pandas as pd
from pull_eagle import query, RAW
SQL = """
SELECT SH.GalaxyID, SH.MassType_DM, SH.BlackHoleMass, SH.MassType_BH, SH.Vmax, SH.StarFormationRate,
       AP.Mass_Star, AP.VelDisp, AP.SFR, FOF.Group_M_Crit200, FOF.Group_R_Crit200
FROM RefL0100N1504_SubHalo AS SH, RefL0100N1504_Aperture AS AP, RefL0100N1504_FOF AS FOF
WHERE SH.SnapNum={snap} AND SH.SubGroupNumber=0
  AND SH.GalaxyID=AP.GalaxyID AND AP.ApertureSize=30
  AND SH.GroupID=FOF.GroupID AND FOF.SnapNum={snap}
  AND AP.Mass_Star>3.0e9
"""
for snap in [23, 19, 15, 12]:
    f = os.path.join(RAW, f"RefL0100N1504_snap{snap}.csv")
    if os.path.exists(f): print(snap, "exists"); continue
    df = query(SQL.format(snap=snap)); df.to_csv(f, index=False); print(snap, len(df), flush=True)
