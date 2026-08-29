"""Regenerate the penalty-dependence table on CORRECTED data at matched N.
Reports, for each penalty discount: per-code edge counts, the union over codes,
how many edges are present in all three, and how many of those are recovered
with the SAME orientation by all three (the 'invariant' count)."""
import pickle, io, contextlib, numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts
D='data/'
V=['STELLAR_MASS','DM_MASS','BH_MASS','GAS_MASS','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
TRUNC=7; rng=np.random.default_rng(4242)

def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    return {k:np.asarray(d[k],float) for k in V}

def graph(d,idx,pen):
    df=pd.DataFrame({k:d[k][idx] for k in V}); df=(df-df.mean())/df.std()
    s=ts.TetradSearch(df); s.set_verbose(False)
    s.add_to_tier(0,'DM_MASS')
    for v in [x for x in V if x!='DM_MASS']: s.add_to_tier(1,v)
    s.use_basis_function_lrt(truncation_limit=TRUNC,alpha=0.01)
    s.use_basis_function_bic(truncation_limit=TRUNC,penalty_discount=pen)
    with contextlib.redirect_stdout(io.StringIO()): s.run_fcit()
    E={}
    for line in str(s.get_java()).split('\n'):
        p=line.strip().split()
        if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
            a,m,b=p[1],p[2],p[3]; k=tuple(sorted((a,b)))
            E[k]=m if (a,b)==k else m[::-1].translate(str.maketrans('<>','><'))
    return E

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
N=min(len(d['STELLAR_MASS']) for d in data.values())
idx={t:rng.permutation(len(d['STELLAR_MASS']))[:N] for t,d in data.items()}
print(f'matched N = {N} per code, truncation {TRUNC}\n')
print('%3s  %-14s %6s %14s %10s' % ('p','edges (T/E/S)','union','present in all 3','invariant'))
for pen in [3,5,8,12]:
    G={t:graph(data[t],idx[t],pen) for t in data}
    union=set().union(*[set(g) for g in G.values()])
    allthree=[k for k in union if all(k in g for g in G.values())]
    inv=[k for k in allthree
         if len({G[t][k] for t in G})==1 and any(c in G['TNG50'][k] for c in '<>')]
    print('%3d  %-14s %6d %14d %10d' % (
        pen, '/'.join(str(len(G[t])) for t in ['TNG50','EAGLE','SIMBA']),
        len(union), len(allthree), len(inv)))
