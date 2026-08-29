import pickle, numpy as np
D='/Users/vidigoat/astrophysics/reanalysis/data/'
rng=np.random.default_rng(23)
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def slope(x,y):
    A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,y,rcond=None); return b[1]

print('='*72)
print('EAGLE halo slope vs the self-regulation prediction  M_BH ~ M_halo^(5/3)')
print('Booth & Schaye (2009,2010): BH grows until its injected energy unbinds the')
print('halo gas, giving M_BH proportional to halo binding energy -> exponent 5/3.')
print('='*72)
for t in ['TNG50','EAGLE','SIMBA']:
    d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
    bh,mh,ms=d['BH_MASS'],d['DM_MASS'],d['STELLAR_MASS']
    lo,hi=np.percentile(mh,[5,95]); m=(mh>lo)&(mh<hi)
    s_all=slope(mh[m],bh[m])
    # slope at fixed stellar mass = the DIRECT halo dependence
    def res(y,X):
        A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return y-A@b
    s_dir=slope(res(mh[m],[ms[m]]),res(bh[m],[ms[m]]))
    print(f'  {t:7s} raw  M_BH ~ M_halo^{s_all:5.2f}    direct (at fixed M*)  ^{s_dir:5.2f}')

print()
print('='*72)
print('SECOND FINGERPRINT: negative gas partial = the BH has EXPELLED its gas.')
print('r(M_BH, M_gas | M*, M_halo, sigma).  Negative => ejective feedback.')
print('Bootstrapped, matched N, common mass window 9.5-11.0.')
print('='*72)
LO,HI=9.5,11.0
sub={}
for t in ['TNG50','EAGLE','SIMBA']:
    d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
    m=(d['STELLAR_MASS']>LO)&(d['STELLAR_MASS']<HI)
    sub[t]={k:v[m] for k,v in d.items()}
N=min(len(s['STELLAR_MASS']) for s in sub.values())
print(f'matched N = {N}\n')
print(f'{"":8s}{"r(Mbh,Mgas|rest)":>20s}{"68% interval":>22s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=sub[t]; n=len(d['STELLAR_MASS']); vals=[]
    for _ in range(300):
        i=rng.choice(n,N,replace=False)
        vals.append(pc(d['BH_MASS'][i],d['GAS_MASS'][i],
                       [d['STELLAR_MASS'][i],d['DM_MASS'][i],np.log10(np.maximum(d['VEL_DISP'][i],1e-3))]))
    v=np.array(vals); lo,hi=np.percentile(v,[16,84])
    print(f'{t:8s}{np.median(v):>20.3f}{f"[{lo:+.3f}, {hi:+.3f}]":>22s}')
print()
print('Feedback modes for reference:')
print('  TNG50 : thermal quasar mode + KINETIC wind mode below an accretion threshold')
print('  EAGLE : single-mode stochastic THERMAL heating only (no kinetic/jet channel)')
print('  SIMBA : two-mode bipolar KINETIC jets + X-ray feedback')
