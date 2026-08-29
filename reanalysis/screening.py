"""The crispest form of the result: a SCREENING test.
If M_halo screens off M* from M_BH, then r(M_BH,M*|M_halo)~0 while r(M_BH,M_halo|M*) stays large.
If M* screens off M_halo, the reverse. This is a statement about which relation is
FUNDAMENTAL and which is a projection."""
import pickle, numpy as np
D='/Users/vidigoat/astrophysics/reanalysis/data/'
LO,HI=9.5,11.0

def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z))
        b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])

def boot_ci(f,n,reps=500,seed=3):
    g=np.random.default_rng(seed)
    v=[f(g.choice(n,n,replace=True)) for _ in range(reps)]
    return np.percentile(v,[16,84])

print(f'Common mass window {LO} < log M* < {HI}.  Pearson r, 68% bootstrap interval.\n')
hdr=f'{"":8s}{"r(Mbh,M*)":>18s}{"r(Mbh,M*|Mh)":>18s}{"r(Mbh,Mh)":>18s}{"r(Mbh,Mh|M*)":>18s}'
print(hdr); print('-'*len(hdr))
for t in ['TNG50','EAGLE','SIMBA']:
    d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
    m=(d['STELLAR_MASS']>LO)&(d['STELLAR_MASS']<HI)
    bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
    n=len(bh)
    fs=[lambda i: pc(bh[i],ms[i],[]),
        lambda i: pc(bh[i],ms[i],[mh[i]]),
        lambda i: pc(bh[i],mh[i],[]),
        lambda i: pc(bh[i],mh[i],[ms[i]])]
    row=f'{t:8s}'
    for f in fs:
        v=f(np.arange(n)); lo,hi=boot_ci(f,n)
        row+=f'{v:>10.3f} [{lo:.2f},{hi:.2f}]'
    print(row)

print()
print('Reading: a partial correlation that COLLAPSES toward zero means the controlled')
print('variable SCREENS OFF the other -- the surviving one is the fundamental link.')
print()
print('Also: black hole mass at fixed halo mass vs at fixed stellar mass (scatter, dex)')
print(f'{"":8s}{"sd(Mbh|M*)":>14s}{"sd(Mbh|Mh)":>14s}{"which is tighter":>20s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
    m=(d['STELLAR_MASS']>LO)&(d['STELLAR_MASS']<HI)
    bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
    def sd(x):
        A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,bh,rcond=None)
        return float(np.std(bh-A@b))
    a,b_=sd(ms),sd(mh)
    print(f'{t:8s}{a:>14.3f}{b_:>14.3f}{("stellar mass" if a<b_ else "halo mass"):>20s}')
