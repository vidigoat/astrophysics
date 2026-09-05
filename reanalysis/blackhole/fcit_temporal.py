"""FCIT with TIME-ORDERED tiers on EAGLE main-progenitor branches.

Tier 0: z=2 quantities, tier 1: z=1, tier 2: z=0.  Time order is the one piece of
background knowledge nobody disputes, so arrowheads between tiers are now
identifiable where the z=0-only search left circles.  Within each tier the halo
is additionally placed before the baryons (the halo-tier-0 prior of the paper).

Usage: python fcit_temporal.py [run] [penalty]
"""

import sys, io, contextlib, pickle
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

run = sys.argv[1] if len(sys.argv) > 1 else "RefL0050N0752"
PEN = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
TRUNC = 7

d = pickle.load(open(f"data/EAGLE_{run}_temporal.pkl", "rb"))
TIERS = [
    (0, ["M200_z2"]),
    (1, ["MSTAR_z2", "MBH_z2"]),
    (2, ["M200_z1"]),
    (3, ["MSTAR_z1", "MBH_z1"]),
    (4, ["M200_z0"]),
    (5, ["MSTAR_z0", "MBH_z0", "SFR_z0", "SIGMA_z0"]),
]
V = [v for _, vs in TIERS for v in vs]
ok = np.ones(len(d["RootID"]), bool)
for v in V:
    ok &= np.isfinite(d[v])
ok &= (d["MSTAR_z0"] > 9.0) & (d["M200_z0"] > 10.5) & d["CENTRAL_z1"] & d["CENTRAL_z2"]
df = pd.DataFrame({v: d[v][ok] for v in V})
df = (df - df.mean()) / df.std()
print(f"{run}: N={len(df)}  penalty={PEN}  truncation={TRUNC}")

s = ts.TetradSearch(df)
s.set_verbose(False)
for t, vs in TIERS:
    for v in vs:
        s.add_to_tier(t, v)
# no baryonic quantity, at any epoch, may cause a halo mass at any epoch
HALO = [v for v in V if v.startswith("M200")]
for b in V:
    if b in HALO:
        continue
    for h in HALO:
        s.set_forbidden(b, h)
s.use_basis_function_lrt(truncation_limit=TRUNC, alpha=0.01)
s.use_basis_function_bic(truncation_limit=TRUNC, penalty_discount=PEN)
with contextlib.redirect_stdout(io.StringIO()):
    s.run_fcit()
E = []
for line in str(s.get_java()).split("\n"):
    p = line.strip().split()
    if len(p) >= 4 and p[0].rstrip(".").isdigit() and set(p[2]) <= set("<->o-"):
        E.append((p[1], p[2], p[3]))
print(f"{len(E)} edges")
for a, m, b in sorted(E):
    flag = ""
    if "MBH_z0" in (a, b):
        flag = "   <-- into/out of M_BH(z=0)"
    if "MBH_z1" in (a, b) and ("MSTAR_z0" in (a, b) or "SFR_z0" in (a, b)):
        flag = "   <-- BH feedback on later galaxy"
    print(f"   {a:10s} {m:4s} {b:10s}{flag}")
