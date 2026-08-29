"""Falsification tests for the headline claim.
The claim: in SIMBA the black hole is causally wired to the baryon cycle
(gas mass, sSFR); in TNG50 and EAGLE it is not.
Threats: (a) unequal N gives SIMBA more power; (b) penalty choice; (c) sampling noise.
"""
import pickle, io, contextlib
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts
D='/Users/vidigoat/astrophysics/reanalysis/data/'
rng=np.random.default_rng(7)

def graph_at(df,pen,trunc=7,alpha=0.01):
    s=ts.TetradSearch(df); s.set_verbose(False)
    s.use_basis_function_lrt(truncation_limit=trunc,alpha=alpha)
    s.use_basis_function_bic(truncation_limit=trunc,penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_fcit()
    E=[]
    for line in str(s.get_java()).split('\n'):
        p=line.strip().split()
        if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
            E.append((p[1],p[2],p[3]))
    return E

data={t:pickle.load(open(D+f'{t}_clean.pkl','rb')) for t in ['TNG50','EAGLE','SIMBA']}
NMIN=min(len(d['STELLAR_MASS']) for d in data.values())
print(f'Matching all three simulations to the smallest clean sample: N = {NMIN}\n')

TARGETS=[('BH_MASS','GAS_MASS'),('BH_MASS','SSFR'),('BH_MASS','STELLAR_MASS'),
         ('BH_MASS','VEL_DISP'),('BH_MASS','DM_MASS')]

def has(E,a,b):
    for x,m,y in E:
        if {x,y}=={a,b}: return f'{x} {m} {y}'
    return '.'

for pen in [3,5,8]:
    print('='*96)
    print(f'penalty discount = {pen}   |  N matched to {NMIN}  |  30 bootstrap draws, % of draws edge present')
    print('='*96)
    print(f'{"pair":26s}'+''.join(f'{t:>22s}' for t in ['TNG50','EAGLE','SIMBA']))
    counts={t:{p:0 for p in TARGETS} for t in data}
    example={t:{p:'.' for p in TARGETS} for t in data}
    for t,d in data.items():
        cols=list(d.keys()); n=len(d['STELLAR_MASS'])
        for it in range(30):
            idx=rng.choice(n,NMIN,replace=False)
            df=pd.DataFrame(np.column_stack([d[c][idx] for c in cols]),columns=cols)
            E=graph_at(df,pen)
            for p in TARGETS:
                e=has(E,*p)
                if e!='.':
                    counts[t][p]+=1
                    if example[t][p]=='.': example[t][p]=e
    for p in TARGETS:
        row=f'{p[0]}-{p[1]:16s}'
        for t in ['TNG50','EAGLE','SIMBA']:
            row+=f'{100*counts[t][p]/30:>8.0f}%  {example[t][p].replace("BH_MASS","BH").replace("STELLAR_MASS","M*").replace("GAS_MASS","Mgas").replace("VEL_DISP","sig").replace("DM_MASS","Mdm"):11s}'
        print(row)
    print()
