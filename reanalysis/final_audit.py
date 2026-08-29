"""Recompute every quantitative claim in main_v2.tex from the corrected data."""
import pickle, numpy as np
np.set_printoptions(suppress=True)
def load(t):
    d=dict(pickle.load(open(f'data/{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(np.asarray(d['VEL_DISP'],float),1e-3))
    return {k:np.asarray(v,float) for k,v in d.items()}
def pcorr(x,y,Z):
    X=np.column_stack([np.ones(len(x))]+list(Z))
    rx=x-X@np.linalg.lstsq(X,x,rcond=None)[0]; ry=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    return np.corrcoef(rx,ry)[0,1]
D={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
W=lambda d:(d['STELLAR_MASS']>9.5)&(d['STELLAR_MASS']<11.0)
rng=np.random.default_rng(7)

print("="*62); print("1. JOINT REGRESSION  log M_BH = a logM* + b logM_h + c"); print("="*62)
for t,d in D.items():
    m=W(d); X=np.column_stack([np.ones(m.sum()),d['STELLAR_MASS'][m],d['DM_MASS'][m]])
    y=d['BH_MASS'][m]; b=np.linalg.lstsq(X,y,rcond=None)[0]
    bs=[]
    for _ in range(2000):
        i=rng.integers(0,m.sum(),m.sum()); bs.append(np.linalg.lstsq(X[i],y[i],rcond=None)[0])
    e=np.std(bs,axis=0)
    print(f"  {t:6s} M*^{b[1]:+.2f}+/-{e[1]:.2f}   M_h^{b[2]:+.2f}+/-{e[2]:.2f}   (N={m.sum()})")

print("="*62); print("2. SCREENING  r(M_BH,M_h) raw -> | M*   and mirror"); print("="*62)
for t,d in D.items():
    m=W(d)
    raw=np.corrcoef(d['BH_MASS'][m],d['DM_MASS'][m])[0,1]
    scr=pcorr(d['BH_MASS'][m],d['DM_MASS'][m],[d['STELLAR_MASS'][m]])
    rawS=np.corrcoef(d['BH_MASS'][m],d['STELLAR_MASS'][m])[0,1]
    mir=pcorr(d['BH_MASS'][m],d['STELLAR_MASS'][m],[d['DM_MASS'][m]])
    print(f"  {t:6s} halo {raw:+.3f} -> {scr:+.3f}   |   stellar {rawS:+.3f} -> {mir:+.3f}")

print("="*62); print("3. VARIANCE RATIO R (chain=1)"); print("="*62)
for t,d in D.items():
    m=W(d); ms,mh,bh=d['STELLAR_MASS'][m],d['DM_MASS'][m],d['BH_MASS'][m]
    a=np.polyfit(ms,bh,1)[0]
    lhs=np.var(bh-np.polyval(np.polyfit(mh,bh,1),mh))
    rhs=a**2*np.var(ms-np.polyval(np.polyfit(mh,ms,1),mh))+np.var(bh-np.polyval(np.polyfit(ms,bh,1),ms))
    Rs=[]
    for _ in range(600):
        i=rng.integers(0,m.sum(),m.sum())
        a2=np.polyfit(ms[i],bh[i],1)[0]
        l=np.var(bh[i]-np.polyval(np.polyfit(mh[i],bh[i],1),mh[i]))
        r=a2**2*np.var(ms[i]-np.polyval(np.polyfit(mh[i],ms[i],1),mh[i]))+np.var(bh[i]-np.polyval(np.polyfit(ms[i],bh[i],1),ms[i]))
        Rs.append(l/r)
    print(f"  {t:6s} R = {lhs/rhs:.3f} +/- {np.std(Rs):.3f}")

print("="*62); print("4. SCATTER REDUCTION  (does adding M_h help, given M* and sigma?)"); print("="*62)
for t,d in D.items():
    m=W(d); y=d['BH_MASS'][m]
    def sd(cols):
        X=np.column_stack([np.ones(m.sum())]+cols)
        return np.std(y-X@np.linalg.lstsq(X,y,rcond=None)[0])
    gal=sd([d['STELLAR_MASS'][m],d['LOG_SIGMA'][m]]); both=sd([d['STELLAR_MASS'][m],d['LOG_SIGMA'][m],d['DM_MASS'][m]])
    halo=sd([d['DM_MASS'][m]])
    print(f"  {t:6s} halo adds {100*(1-both/gal):5.1f}%   galaxy adds once halo known {100*(1-both/halo):5.1f}%")

print("="*62); print("5. FEEDBACK  r(M_BH,M_gas|M*,M_h,sigma), M_h<11.5"); print("="*62)
for t,d in D.items():
    m=(d['DM_MASS']<11.5)&np.isfinite(d['BH_MASS'])&np.isfinite(d['GAS_MASS'])
    print(f"  {t:6s} {pcorr(d['BH_MASS'][m],d['GAS_MASS'][m],[d['STELLAR_MASS'][m],d['DM_MASS'][m],d['LOG_SIGMA'][m]]):+.3f}  (N={m.sum()})")

print("="*62); print("6. CENSORING REVERSAL (TNG50, r(M_BH,M_h|M*))"); del D
print("="*62)
for tag,lab in [('clean','seeded & star-forming'),('full','all subhaloes incl. floored')]:
    d=dict(pickle.load(open(f'data/TNG50_{tag}.pkl','rb'))); g={k:np.asarray(v,float) for k,v in d.items()}
    m=(g['STELLAR_MASS']>9.5)&(g['STELLAR_MASS']<11.0)&np.isfinite(g['BH_MASS'])
    print(f"  {lab:28s} {pcorr(g['BH_MASS'][m],g['DM_MASS'][m],[g['STELLAR_MASS'][m]]):+.3f}  (N={m.sum()})")
