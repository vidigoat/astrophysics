"""Broken-slope model for the conditional halo dependence:
   log M_BH = c + a log M* + b_lo (x - x_c) [x<x_c] + b_hi (x - x_c) [x>=x_c],   x = log M200.
The break x_c is found by profile least squares; errors by bootstrap.  Run for EAGLE at z=0..3
(snapshot-complete centrals) and for TNG50/SIMBA at z=0."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from headline import tng50, eagle, simba
XC=np.arange(11.3,13.6,0.05)
def design(ms,mh,xc):
    lo=np.minimum(mh-xc,0); hi=np.maximum(mh-xc,0)
    return np.column_stack([np.ones(len(ms)),ms,lo,hi])
def fit(bh,ms,mh):
    best=None
    for xc in XC:
        X=design(ms,mh,xc); b,res,*_=np.linalg.lstsq(X,bh,rcond=None); r=((bh-X@b)**2).sum()
        if best is None or r<best[0]: best=(r,xc,b)
    return best
def boot(bh,ms,mh,n=200,seed=0):
    rng=np.random.default_rng(seed); out=[]
    for i in range(n):
        idx=rng.integers(0,len(bh),len(bh)); r,xc,b=fit(bh[idx],ms[idx],mh[idx]); out.append([xc,b[1],b[2],b[3]])
    return np.array(out)
def report(lab,bh,ms,mh):
    r,xc,b=fit(bh,ms,mh); bs=boot(bh,ms,mh)
    r1=((bh-np.column_stack([np.ones(len(ms)),ms,mh])@np.linalg.lstsq(np.column_stack([np.ones(len(ms)),ms,mh]),bh,rcond=None)[0])**2).sum()
    n=len(bh); dbic=n*np.log(r/r1)+np.log(n)   # broken has one extra parameter (xc) + one slope: use 2
    dbic=n*np.log(r/r1)+2*np.log(n)
    print(f"{lab:22s} N={n:5d}  x_c={xc:.2f}±{bs[:,0].std():.2f}  a={b[1]:+.2f}  b_lo={b[2]:+.2f}±{bs[:,2].std():.2f}  b_hi={b[3]:+.2f}±{bs[:,3].std():.2f}   ΔBIC(broken−single)={dbic:+.0f}")
if __name__=='__main__':
    Om,OL=0.307,0.693; Z={28:0.0,23:0.503,19:1.004,15:2.012,12:3.017}
    for s in [28,23,19,15,12]:
        f='../../Data/eagle_v3/RefL0100N1504_z0.csv' if s==28 else f'../../Data/eagle_v3/RefL0100N1504_snap{s}.csv'
        d=pd.read_csv(f); d=d[(d.BlackHoleMass>0)&(d.Group_M_Crit200>10**10.5)&(d.Mass_Star>10**9.5)&(d.Mass_Star<1e11)]
        report(f"EAGLE z={Z[s]:.1f}",np.log10(d.BlackHoleMass.values),np.log10(d.Mass_Star.values),np.log10(d.Group_M_Crit200.values))
    for lab,fn in [('TNG50 z=0',tng50),('SIMBA z=0',simba)]:
        d=fn(); report(lab,d['bh'],d['ms'],d['mh'])
