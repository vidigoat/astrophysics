"""Rebuild the cross-code matched comparison of the paper's Section 5 on CORRECTED
data, keeping the paper's own procedure: matched N, standardised variables,
consensus over 50 runs, edges classified invariant / conflicting / majority /
code-specific."""
import pickle, io, contextlib, itertools, os
import numpy as np, pandas as pd
from pytetrad.tools import TetradSearch as ts
D='/Users/vidigoat/astrophysics/reanalysis/data/'
OUT='/Users/vidigoat/astrophysics/reanalysis/results/'
rng=np.random.default_rng(4242)
V=['STELLAR_MASS','DM_MASS','BH_MASS','GAS_MASS','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
NRUNS=50; PEN=5; TRUNC=7
NRUNS_PART=10; NPART=8

def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    return {k:v for k,v in d.items() if k in V}

def run(df,seed):
    s=ts.TetradSearch(df); s.set_verbose(False)
    try: s.set_seed(seed)
    except Exception: pass
    s.use_basis_function_lrt(truncation_limit=TRUNC,alpha=0.01)
    s.use_basis_function_bic(truncation_limit=TRUNC,penalty_discount=PEN)
    with contextlib.redirect_stdout(io.StringIO()): s.run_fcit()
    E=[]
    for line in str(s.get_java()).split('\n'):
        p=line.strip().split()
        if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
            E.append((p[1],p[2],p[3]))
    return E

def consensus(d,idx,nr=None):
    nr=nr or NRUNS
    df=pd.DataFrame(np.column_stack([(d[c][idx]-d[c][idx].mean())/d[c][idx].std() for c in V]),columns=V)
    pres,ori={},{}
    for i in range(nr):
        for a,m,b in run(df,3000+i):
            k=tuple(sorted([a,b])); pres[k]=pres.get(k,0)+1
            head = a if m.endswith('>') else (b if m.startswith('<') else None)
            key=(k,head); ori[key]=ori.get(key,0)+1
    out={}
    for k,c in pres.items():
        if c/nr<0.5: continue
        cands={h:n for (kk,h),n in ori.items() if kk==k}
        best=max(cands,key=cands.get) if cands else None
        out[k]= best if (best is not None and cands[best]/c>=0.6) else None
    return out

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
N=min(len(d['STELLAR_MASS']) for d in data.values())
print(f'matched N = {N} per code, penalty {PEN}, {NRUNS} runs\n')
G={t:consensus(d,rng.choice(len(d['STELLAR_MASS']),N,replace=False)) for t,d in data.items()}
for t in G: print(f'  {t}: {len(G[t])} edges')
union=set().union(*[set(g) for g in G.values()])
inv=conf=maj=spec=0; invlist=[];conflist=[]
for e in union:
    inn=[t for t in G if e in G[t]]
    if len(inn)==3:
        dirs={G[t][e] for t in inn}
        if len(dirs)==1 and None not in dirs: inv+=1; invlist.append(e)
        elif len(dirs)==1: inv+=0; maj+=0; conf+=0; invlist.append(e) if False else None
        else: conf+=1; conflist.append(e)
    elif len(inn)==2: maj+=1
    else: spec+=1
allthree=sum(1 for e in union if len([t for t in G if e in G[t]])==3)
print(f'\n  union {len(union)} | present in all 3: {allthree} | invariant(same dir) {inv} | '
      f'conflicting {conf} | two codes {maj} | one code {spec}')
print('\n  invariant edges:'); [print('    ',a,'->',b) for (a,b) in invlist]
print('  conflicting edges:'); [print('    ',a,'-',b) for (a,b) in conflist]

print('\n--- within-code vs cross-code skeleton Jaccard, 8 disjoint partitions ---')
def jac(x,y):
    sx,sy=set(x),set(y); return len(sx&sy)/max(len(sx|sy),1)
wi,cr=[],[]
for _ in range(NPART):
    halves={}
    for t,d in data.items():
        n=len(d['STELLAR_MASS']); p=rng.permutation(n); h=min(N, n//2)
        halves[t]=(consensus(d,p[:h],NRUNS_PART),consensus(d,p[h:2*h],NRUNS_PART))
    for t in data: wi.append(jac(halves[t][0],halves[t][1]))
    for a,b in itertools.combinations(data,2): cr.append(jac(halves[a][0],halves[b][0]))
print(f'  within-code Jaccard median {np.median(wi):.2f}  (16-84%: {np.percentile(wi,16):.2f}-{np.percentile(wi,84):.2f})')
print(f'  cross-code  Jaccard median {np.median(cr):.2f}  (16-84%: {np.percentile(cr,16):.2f}-{np.percentile(cr,84):.2f})')
