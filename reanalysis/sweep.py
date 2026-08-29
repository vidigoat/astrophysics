"""Systematic screening sweep. For EVERY node, which dependences SURVIVE when all
other measured variables are held fixed? Then: where do the codes DISAGREE in sign?
A sign disagreement that is not explained by variable definitions is physics."""
import pickle, numpy as np, itertools
D='/Users/vidigoat/astrophysics/reanalysis/data/'
LO,HI=9.5,11.0
V=['STELLAR_MASS','GAS_MASS','DM_MASS','BH_MASS','HALFMASS_RAD','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
NICE={'STELLAR_MASS':'M*','GAS_MASS':'Mgas','DM_MASS':'Mhalo','BH_MASS':'Mbh',
      'HALFMASS_RAD':'Rhalf','LOG_SIGMA':'sigma','STAR_METALLICITY':'Zstar',
      'GAS_METALLICITY':'Zgas','SSFR':'sSFR'}
def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    m=(d['STELLAR_MASS']>LO)&(d['STELLAR_MASS']<HI)
    return {k:v[m] for k,v in d.items() if k in V}
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
print('Full partial correlation matrix: r(X, Y | all 7 other variables).')
print(f'Common window {LO}<logM*<{HI}.  N = ' +
      ', '.join(f'{t}:{len(d["STELLAR_MASS"])}' for t,d in data.items()))
print()
res={}
for a,b in itertools.combinations(V,2):
    row={}
    for t,d in data.items():
        rest=[d[k] for k in V if k not in (a,b)]
        row[t]=pc(d[a],d[b],rest)
    res[(a,b)]=row

print(f'{"pair":22s}{"TNG50":>10s}{"EAGLE":>10s}{"SIMBA":>10s}   flag')
print('-'*66)
strong=[]; flips=[]
for (a,b),row in sorted(res.items(), key=lambda kv:-max(abs(v) for v in kv[1].values())):
    vals=[row[t] for t in ['TNG50','EAGLE','SIMBA']]
    mx=max(abs(v) for v in vals)
    if mx<0.20: continue
    sgn=set(np.sign(v) for v in vals if abs(v)>0.15)
    flag=''
    if len(sgn)>1: flag='SIGN FLIP'; flips.append((a,b,vals))
    if all(abs(v)>0.25 for v in vals) and len(sgn)==1: flag='universal'
    strong.append((a,b,vals,flag))
    print(f'{NICE[a]+"-"+NICE[b]:22s}'+''.join(f'{v:>10.3f}' for v in vals)+f'   {flag}')

print()
print('='*66)
print('SIGN FLIPS  (codes disagree on the DIRECTION of a direct dependence)')
print('='*66)
for a,b,vals in flips:
    print(f'  {NICE[a]:9s} - {NICE[b]:9s}  TNG50={vals[0]:+.3f}  EAGLE={vals[1]:+.3f}  SIMBA={vals[2]:+.3f}')
print()
print('UNIVERSAL  (all three agree, strong) -- candidate real physics')
for a,b,vals,f in strong:
    if f=='universal':
        print(f'  {NICE[a]:9s} - {NICE[b]:9s}  {vals[0]:+.3f} / {vals[1]:+.3f} / {vals[2]:+.3f}')
