import pickle, io, contextlib
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

def fit(x,y):
    """OLS slope in log-log, plus orthogonal scatter in dex."""
    A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,y,rcond=None)
    r=y-A@b; return b[1], float(np.std(r))

OUT=[]
def P(s=''):
    print(s); OUT.append(s)

data={}
for tag in ['TNG50','EAGLE','SIMBA']:
    d=pickle.load(open(D+f'{tag}_clean.pkl','rb'))
    d=dict(d); d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    data[tag]=d

P('='*80)
P('CAUSAL NEIGHBOURHOOD OF M_BH AND sSFR   (corrected data, penalty discount 5)')
P('='*80)
for tag in ['TNG50','EAGLE','SIMBA']:
    d=data[tag]; cols=[c for c in d if c!='LOG_SIGMA']
    df=pd.DataFrame(np.column_stack([d[c] for c in cols]),columns=cols)
    E=graph_at(df,5)
    P(f'\n{tag}  N={len(df)}  {len(E)} edges')
    for t in ['BH_MASS','SSFR']:
        nb=[f'{a} {m} {b}' for a,m,b in E if a==t or b==t]
        P(f'   {t:9s}: '+('  |  '.join(nb) if nb else '*** NO EDGES ***'))

P(); P('='*80)
P('CENSORING ARTEFACT TEST')
P('Same search on the FULL sample, where M_BH is pinned at the floor for 66% of')
P('TNG50 and 57% of SIMBA subhaloes.  If the BH edges change, the published BH')
P('result was tracking the SEEDING threshold, not accretion.')
P('='*80)
for tag in ['TNG50','EAGLE','SIMBA']:
    f=pickle.load(open(D+f'{tag}_full.pkl','rb'))
    cols=[c for c in f if c not in ('BH_SEEDED','STARFORMING')]
    df=pd.DataFrame(np.column_stack([f[c] for c in cols]),columns=cols)
    df=df.replace([np.inf,-np.inf],np.nan).fillna(df.median(numeric_only=True))
    E=graph_at(df,5)
    nb=[f'{a} {m} {b}' for a,m,b in E if 'BH_MASS' in (a,b)]
    P(f'\n{tag} FULL (censored) N={len(df)}')
    P(f'   BH_MASS  : '+('  |  '.join(nb) if nb else '*** NO EDGES ***'))

P(); P('='*80)
P('SCALING EXPONENTS ON THE CORRECTED DATA  (log-log OLS slope, orthogonal scatter)')
P('='*80)
rel=[('M_BH ~ M*^a',      'STELLAR_MASS','BH_MASS'),
     ('M_BH ~ sigma^a',   'LOG_SIGMA','BH_MASS'),
     ('M_BH ~ M_halo^a',  'DM_MASS','BH_MASS'),
     ('M* ~ M_halo^a',    'DM_MASS','STELLAR_MASS'),
     ('sigma ~ M*^a',     'STELLAR_MASS','LOG_SIGMA'),
     ('R_half ~ M*^a',    'STELLAR_MASS','HALFMASS_RAD'),
     ('Z* ~ M*^a',        'STELLAR_MASS','STAR_METALLICITY'),
     ('sSFR ~ M*^a',      'STELLAR_MASS','SSFR')]
P(f'{"relation":18s}'+''.join(f'{t:>22s}' for t in ['TNG50','EAGLE','SIMBA']))
for name,xk,yk in rel:
    row=f'{name:18s}'
    for tag in ['TNG50','EAGLE','SIMBA']:
        d=data[tag]; a,sc=fit(d[xk],d[yk]); row+=f'{a:>13.3f} +-{sc:6.3f}'
    P(row)

open('/Users/vidigoat/astrophysics/reanalysis/results/physics.txt','w').write('\n'.join(OUT))
