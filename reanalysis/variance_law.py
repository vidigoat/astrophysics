"""Robustness of the variance identity, and what it predicts for the real universe."""
import pickle, numpy as np
D='/Users/vidigoat/astrophysics/reanalysis/data/'
rng=np.random.default_rng(88)
def load(t):
    return dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
def sd(y,X):
    A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return float(np.std(y-A@b))
def sl(x,y):
    A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,y,rcond=None); return b[1]
def ratio(bh,ms,mh):
    a=sl(ms,bh)
    return sd(bh,[mh])/np.sqrt(a**2*sd(ms,[mh])**2+sd(bh,[ms])**2)

print('VARIANCE IDENTITY RATIO  (=1 means pure chain M_halo -> M* -> M_BH)')
print('bootstrapped 500x, across three mass windows\n')
print(f'{"window":>14s}'+''.join(f'{t:>20s}' for t in ['TNG50','EAGLE','SIMBA']))
for lo,hi in [(9.0,12.0),(9.5,11.0),(10.0,11.5)]:
    row=f'{f"{lo}-{hi}":>14s}'
    for t in ['TNG50','EAGLE','SIMBA']:
        d=load(t); m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)
        bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
        n=len(bh)
        if n<200: row+=f'{"--":>20s}'; continue
        v=ratio(bh,ms,mh)
        bs=[ratio(bh[i],ms[i],mh[i]) for i in (rng.choice(n,n,replace=True) for _ in range(500))]
        row+=f'{v:>13.3f}+-{np.std(bs):5.3f}'
    print(row)

print()
print('='*78)
print('WHAT THE REAL UNIVERSE PREDICTS')
print('The identity needs only three published numbers. Chain predicts:')
print('   sigma(M_BH|M_halo) = sqrt( a^2 sigma(M*|M_halo)^2 + sigma(M_BH|M*)^2 )')
print('='*78)
print(f'{"a (MBH-M* slope)":>22s}{"sd(M*|Mh)":>12s}{"sd(MBH|M*)":>13s}{"=> chain sd(MBH|Mh)":>22s}')
for a,s1,s2,lab in [(1.17,0.16,0.29,'Kormendy&Ho13 + Behroozi13'),
                    (1.00,0.16,0.29,'slope=1'),
                    (1.17,0.20,0.29,'wider SHMR scatter'),
                    (1.17,0.16,0.38,'wider MBH-M* scatter')]:
    print(f'{a:>22.2f}{s1:>12.2f}{s2:>13.2f}{np.sqrt(a*a*s1*s1+s2*s2):>22.3f}   {lab}')
print()
print('A FORK instead would give sigma(M_BH|M_halo) SMALLER than sigma(M_BH|M*),')
print('because the halo would then be the better predictor. In the simulations:')
print(f'{"":8s}{"sd(MBH|M*)":>14s}{"sd(MBH|Mhalo)":>16s}{"which is tighter":>20s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t); m=(d['STELLAR_MASS']>9.5)&(d['STELLAR_MASS']<11.0)
    bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
    a_,b_=sd(bh,[ms]),sd(bh,[mh])
    print(f'{t:8s}{a_:>14.3f}{b_:>16.3f}{("stellar" if a_<b_ else "HALO"):>20s}')
