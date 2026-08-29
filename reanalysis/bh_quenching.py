"""The physics question: what CAUSES black hole mass, and what CAUSES quenching,
in each code -- and does the answer track each code's accretion prescription?"""
import pickle, io, contextlib, itertools
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D='/Users/vidigoat/astrophysics/reanalysis/data/'

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

def pcorr(x,y,Z):
    """partial correlation of x,y controlling for columns Z (list of arrays)"""
    def resid(v):
        if not Z: return v-v.mean()
        A=np.column_stack([np.ones(len(v))]+list(Z))
        beta,*_=np.linalg.lstsq(A,v,rcond=None)
        return v-A@beta
    rx,ry=resid(x),resid(y)
    return float(np.corrcoef(rx,ry)[0,1])

PEN=5
print('='*78)
print(f'CAUSAL NEIGHBOURHOOD OF BH_MASS AND SSFR  (penalty discount = {PEN})')
print('='*78)
store={}
for tag in ['TNG50','EAGLE','SIMBA']:
    d=pickle.load(open(D+f'{tag}_clean.pkl','rb'))
    cols=list(d.keys()); df=pd.DataFrame(np.column_stack([d[c] for c in cols]),columns=cols)
    E=graph_at(df,PEN); store[tag]=(d,E)
    print(f'\n--- {tag}  (N={len(df)}, {len(E)} edges) ---')
    for target in ['BH_MASS','SSFR']:
        nb=[f'{a} {m} {b}' for a,m,b in E if a==target or b==target]
        print(f'  {target}: ' + ('; '.join(nb) if nb else 'NO EDGES'))

print()
print('='*78)
print('IS THE BLACK HOLE A *DIRECT* CAUSE OF QUENCHING, OR IS M* THE CONFOUNDER?')
print('Bluck/Piotrowska ask this with random forests, which cannot separate the two.')
print('='*78)
print(f'{"sim":7s} {"r(sSFR,Mbh)":>12s} {"r(sSFR,Mbh|M*)":>15s} {"r(sSFR,M*)":>12s} {"r(sSFR,M*|Mbh)":>15s} {"r(sSFR,Mbh|M*,sig)":>19s}')
for tag in ['TNG50','EAGLE','SIMBA']:
    d,_=store[tag]
    s,mb,ms,sg=d['SSFR'],d['BH_MASS'],d['STELLAR_MASS'],d['VEL_DISP']
    print(f'{tag:7s} {pcorr(s,mb,[]):12.3f} {pcorr(s,mb,[ms]):15.3f} '
          f'{pcorr(s,ms,[]):12.3f} {pcorr(s,ms,[mb]):15.3f} {pcorr(s,mb,[ms,sg]):19.3f}')

print()
print('='*78)
print('WHAT SETS BLACK HOLE MASS?  partial correlation of M_BH with each candidate,')
print('controlling for ALL other variables (i.e. the DIRECT dependence).')
print('='*78)
cands=['STELLAR_MASS','GAS_MASS','DM_MASS','VEL_DISP','HALFMASS_RAD','GAS_METALLICITY','STAR_METALLICITY','SSFR']
print(f'{"variable":18s}' + ''.join(f'{t:>12s}' for t in ['TNG50','EAGLE','SIMBA']))
for c in cands:
    row=f'{c:18s}'
    for tag in ['TNG50','EAGLE','SIMBA']:
        d,_=store[tag]
        others=[d[k] for k in cands if k!=c]
        row+=f'{pcorr(d["BH_MASS"],d[c],others):12.3f}'
    print(row)
