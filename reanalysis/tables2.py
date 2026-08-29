"""Within- vs cross-code agreement with GENUINELY matched draw sizes.
Every draw is the same size for every code: half the smallest sample."""
import pickle, io, contextlib, itertools
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts
D='reanalysis/data/'
rng=np.random.default_rng(24680)
V=['STELLAR_MASS','DM_MASS','BH_MASS','GAS_MASS','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    m=(d['STELLAR_MASS']>9.5)&(d['STELLAR_MASS']<11.0)
    return {k:v[m] for k,v in d.items() if k in V}
def graph(d,idx,pen=5,nr=15):
    df=pd.DataFrame(np.column_stack([(d[c][idx]-d[c][idx].mean())/d[c][idx].std() for c in V]),columns=V)
    pres={}
    for i in range(nr):
        s=ts.TetradSearch(df); s.set_verbose(False)
        try: s.set_seed(9000+i)
        except Exception: pass
        s.use_basis_function_lrt(truncation_limit=7,alpha=0.01)
        s.use_basis_function_bic(truncation_limit=7,penalty_discount=pen)
        with contextlib.redirect_stdout(io.StringIO()): s.run_fcit()
        for line in str(s.get_java()).split('\n'):
            p=line.strip().split()
            if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
                k=tuple(sorted([p[1],p[3]])); pres[k]=pres.get(k,0)+1
    return {k for k,c in pres.items() if c/nr>=0.5}
def jac(a,b): return len(a&b)/max(len(a|b),1)

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
H=min(len(d['STELLAR_MASS']) for d in data.values())//2
print(f'matched draw size = {H} galaxies for EVERY code (half the smallest sample)')
print(f'sample sizes: '+', '.join(f'{t}={len(d["STELLAR_MASS"])}' for t,d in data.items()))
print()
NP=12
wi,cr={t:[] for t in data},{}
for a,b in itertools.combinations(data,2): cr[(a,b)]=[]
for p in range(NP):
    halves={}
    for t,d in data.items():
        n=len(d['STELLAR_MASS']); perm=rng.permutation(n)
        halves[t]=(graph(d,perm[:H]),graph(d,perm[H:2*H]))
        wi[t].append(jac(*halves[t]))
    for a,b in itertools.combinations(data,2):
        cr[(a,b)].append(jac(halves[a][0],halves[b][0]))
print(f'{"comparison":<24}{"median Jaccard":>16}{"16-84%":>18}')
allw=[]
for t in data:
    v=np.array(wi[t]); allw+=list(v)
    print(f'{"Within "+t:<24}{np.median(v):>16.2f}{f"{np.percentile(v,16):.2f}-{np.percentile(v,84):.2f}":>18}')
allc=[]
for (a,b),v in cr.items():
    v=np.array(v); allc+=list(v)
    print(f'{a+" vs "+b:<24}{np.median(v):>16.2f}{f"{np.percentile(v,16):.2f}-{np.percentile(v,84):.2f}":>18}')
print()
print(f'  OVERALL within-code median = {np.median(allw):.2f}  ({np.percentile(allw,16):.2f}-{np.percentile(allw,84):.2f})')
print(f'  OVERALL cross-code  median = {np.median(allc):.2f}  ({np.percentile(allc,16):.2f}-{np.percentile(allc,84):.2f})')
