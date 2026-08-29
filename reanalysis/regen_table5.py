"""Regenerate the null-test table (Table 5) on CORRECTED data at TRUE equal power.

Fixes two defects in rebuild_matched.py:
  (1) h = min(N, n//2) gave TNG50 half-draws of 1833 while EAGLE/SIMBA got 3667,
      so every cross-code Jaccard involving TNG50 compared a sparser graph against
      a denser one and was biased low. Here every draw is the SAME size for every
      code: h = 1833 (half the smallest clean sample).
  (2) TetradSearch has no set_seed method, so the previous `try: s.set_seed(...)`
      was a silent no-op and the "consensus over 50 runs" averaged 50 identical
      calls. The search is deterministic given the data, so one run per subsample
      is used here; the variability measured is the real one, from resampling.
"""
import pickle, io, contextlib, itertools, numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts

D='data/'
V=['STELLAR_MASS','DM_MASS','BH_MASS','GAS_MASS','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
PEN=5; TRUNC=7; NPART=20
rng=np.random.default_rng(4242)

def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    return {k:np.asarray(d[k],float) for k in V}

def graph(d, idx):
    df=pd.DataFrame({k:d[k][idx] for k in V})
    df=(df-df.mean())/df.std()
    s=ts.TetradSearch(df); s.set_verbose(False)
    s.use_basis_function_lrt(truncation_limit=TRUNC,alpha=0.01)
    s.use_basis_function_bic(truncation_limit=TRUNC,penalty_discount=PEN)
    with contextlib.redirect_stdout(io.StringIO()): s.run_fcit()
    E={}
    for line in str(s.get_java()).split('\n'):
        p=line.strip().split()
        if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
            a,m,b=p[1],p[2],p[3]
            key=tuple(sorted((a,b)))
            # orientation recorded only if an arrowhead is determined
            E[key] = m if (a,b)==key else m[::-1].translate(str.maketrans('<>','><'))
    return E

def jac(x,y):
    sx,sy=set(x),set(y); return len(sx&sy)/max(len(sx|sy),1)

def orient_agree(x,y):
    shared=[k for k in set(x)&set(y) if '>' in x[k] or '<' in x[k]]
    shared=[k for k in shared if '>' in y[k] or '<' in y[k]]
    if not shared: return None,0
    same=sum(1 for k in shared if x[k]==y[k])
    return same/len(shared), len(shared)

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
H=min(len(d['STELLAR_MASS']) for d in data.values())//2
print(f'equal-power half-draw size h = {H} for every code')
print(f'clean sample sizes: ' + ', '.join(f'{t}={len(d["STELLAR_MASS"])}' for t,d in data.items()))
print(f'partitions: {NPART}, penalty {PEN}, truncation {TRUNC}\n')

wi={t:[] for t in data}; cr={p:[] for p in itertools.combinations(data,2)}
wo={t:[] for t in data}; co={p:[] for p in itertools.combinations(data,2)}
for _ in range(NPART):
    halves={}
    for t,d in data.items():
        p=rng.permutation(len(d['STELLAR_MASS']))
        halves[t]=(graph(d,p[:H]), graph(d,p[H:2*H]))
    for t in data:
        wi[t].append(jac(*halves[t]))
        o,_n=orient_agree(*halves[t])
        if o is not None: wo[t].append(o)
    for a,b in cr:
        cr[(a,b)].append(jac(halves[a][0],halves[b][0]))
        o,_n=orient_agree(halves[a][0],halves[b][0])
        if o is not None: co[(a,b)].append(o)

def med(v): return float(np.median(v)) if len(v) else float('nan')
print('%-22s %8s %8s' % ('Comparison','Jaccard','Orient'))
for t in data:
    print('%-22s %8.2f %8s' % ('Within '+t, med(wi[t]),
          ('%.2f'%med(wo[t])) if wo[t] else '--'))
for a,b in cr:
    print('%-22s %8.2f %8s' % (f'{a} vs {b}', med(cr[(a,b)]),
          ('%.2f'%med(co[(a,b)])) if co[(a,b)] else '--'))
allw=[v for t in wi for v in wi[t]]; allc=[v for p in cr for v in cr[p]]
print(f'\nwithin-code Jaccard median {np.median(allw):.2f} '
      f'(16-84%: {np.percentile(allw,16):.2f}-{np.percentile(allw,84):.2f})')
print(f'cross-code  Jaccard median {np.median(allc):.2f} '
      f'(16-84%: {np.percentile(allc,16):.2f}-{np.percentile(allc,84):.2f})')
