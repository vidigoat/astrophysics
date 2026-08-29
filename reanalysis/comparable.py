"""Only definitionally COMPARABLE variables across the three codes.
DROPPED: HALFMASS_RAD (TNG=total-matter, others=stellar) and GAS_MASS
(TNG=all bound gas incl. hot halo, EAGLE=30kpc, SIMBA=CAESAR galaxy).
KEPT: M*, M_halo(DM), M_BH, sigma, Z_star, Z_gas, sSFR."""
import pickle, numpy as np, itertools
D='/Users/vidigoat/astrophysics/reanalysis/data/'
LO,HI=9.5,11.0
V=['STELLAR_MASS','DM_MASS','BH_MASS','LOG_SIGMA','STAR_METALLICITY','GAS_METALLICITY','SSFR']
N={'STELLAR_MASS':'M*','DM_MASS':'Mhalo','BH_MASS':'Mbh','LOG_SIGMA':'sigma',
   'STAR_METALLICITY':'Zstar','GAS_METALLICITY':'Zgas','SSFR':'sSFR'}
rng=np.random.default_rng(202)
def load(t,lo=LO,hi=HI):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)
    return {k:v[m] for k,v in d.items() if k in V}
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)) if Z else np.ones((len(v),1))
        b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def boot(f,n,reps=400):
    return np.std([f(rng.choice(n,n,replace=True)) for _ in range(reps)])

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
for t,d in data.items():
    X=np.column_stack([(d[k]-d[k].mean())/d[k].std() for k in V])
    print(f'{t:7s} N={len(d["STELLAR_MASS"]):5d}  cond={np.linalg.cond(X):5.1f}')
print()
print(f'{"pair":16s}'+''.join(f'{t:>22s}' for t in ['TNG50','EAGLE','SIMBA'])+'   flag')
print('-'*88)
for a,b in itertools.combinations(V,2):
    vals,errs=[],[]
    for t in ['TNG50','EAGLE','SIMBA']:
        d=data[t]; rest=[d[k] for k in V if k not in (a,b)]
        vals.append(pc(d[a],d[b],rest))
        errs.append(boot(lambda i: pc(d[a][i],d[b][i],[r[i] for r in rest]),len(d[a])))
    if max(abs(v) for v in vals)<0.18: continue
    sig=[(v,e) for v,e in zip(vals,errs) if abs(v)>3*e and abs(v)>0.15]
    sgn=set(np.sign(v) for v,e in sig)
    flag='SIGN FLIP' if len(sgn)>1 else ('universal' if all(abs(v)>0.25 for v in vals) and len(set(np.sign(vals)))==1 else '')
    print(f'{N[a]+"-"+N[b]:16s}'+''.join(f'{v:>15.3f}+-{e:5.3f}' for v,e in zip(vals,errs))+f'   {flag}')

print()
print('='*88)
print('THE FMR, as the literature defines it:  r(Z_gas, sSFR | M*)')
print('Observed (Mannucci+2010, Curti+2020): NEGATIVE - at fixed M*, higher SFR means')
print('lower metallicity, interpreted as pristine gas inflow raising SFR and diluting Z.')
print('='*88)
print(f'{"":8s}{"r(Zgas,sSFR|M*)":>20s}{"+-":>8s}{"sign matches obs?":>20s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t,8.0,12.0)
    v=pc(d['GAS_METALLICITY'],d['SSFR'],[d['STELLAR_MASS']])
    e=boot(lambda i: pc(d['GAS_METALLICITY'][i],d['SSFR'][i],[d['STELLAR_MASS'][i]]),len(d['SSFR']))
    print(f'{t:8s}{v:>20.3f}{e:>8.3f}{("YES" if v<0 else "NO  <-- WRONG SIGN"):>20s}')
print()
print('Same, split by stellar mass:')
print(f'{"bin":>12s}'+''.join(f'{t:>14s}' for t in ['TNG50','EAGLE','SIMBA']))
for lo,hi in [(8.5,9.5),(9.5,10.0),(10.0,10.5),(10.5,11.5)]:
    row=f'{f"{lo}-{hi}":>12s}'
    for t in ['TNG50','EAGLE','SIMBA']:
        d=load(t,lo,hi); n=len(d['SSFR'])
        if n<150: row+=f'{"--":>14s}'; continue
        row+=f'{pc(d["GAS_METALLICITY"],d["SSFR"],[d["STELLAR_MASS"]]):>10.3f}({n:4d})'[:14]
    print(row)
