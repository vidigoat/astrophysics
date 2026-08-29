import pickle, numpy as np
D='reanalysis/data/'; P='Data/'
rng=np.random.default_rng(31415)
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def sd(y,X):
    A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return float(np.std(y-A@b))
def sl(x,y):
    A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,y,rcond=None); return b[1]
def load(t,lo=9.5,hi=11.0):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi); return {k:v[m] for k,v in d.items()}
CH=[]
def chk(l,c,a,tol=0.02):
    ok=abs(c-a)<=tol; CH.append(ok)
    print(f'{"OK " if ok else "!! "} {l:42s} paper={c:>8}  actual={a:>8.3f}')

print('--- VARIANCE RATIO R ---')
for t,cR,cF in [('TNG50',1.021,None),('EAGLE',0.648,0.614),('SIMBA',0.980,None)]:
    d=load(t); bh,ms,mh=d['BH_MASS'],d['STELLAR_MASS'],d['DM_MASS']
    a=sl(ms,bh); R=sd(bh,[mh])/np.sqrt(a*a*sd(ms,[mh])**2+sd(bh,[ms])**2)
    chk(f'{t} R',cR,R,0.005)
    if cF:
        V=np.var(mh); b_=sl(mh,ms); c_=sl(mh,bh); s1=sd(ms,[mh]); s2=sd(bh,[mh])
        af=c_*b_*V/(b_*b_*V+s1*s1); vg=s2*s2+c_*c_*V-(c_*b_*V)**2/(b_*b_*V+s1*s1)
        chk(f'{t} fork prediction',cF,s2/np.sqrt(af**2*s1*s1+max(vg,1e-12)),0.01)

print('\n--- MASS TREND r(bh,Mh|M*) ---')
d=load('EAGLE',0,20)
for lo,hi,cl in [(9.0,9.5,0.41),(10.5,11.0,0.77),(11.0,12.0,0.85)]:
    m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)
    chk(f'EAGLE {lo}-{hi}',cl,pc(d['BH_MASS'][m],d['DM_MASS'][m],[d['STELLAR_MASS'][m]]),0.02)

print('\n--- DIRECT HALO EXPONENT (EAGLE) ---')
def res(y,X):
    A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return y-A@b
d=dict(pickle.load(open(D+'EAGLE_clean.pkl','rb')))
for lo,hi,cl in [(10.5,11.0,0.23),(11.0,11.5,0.73),(11.5,12.0,0.84),(12.0,12.5,1.21)]:
    m=(d['DM_MASS']>lo)&(d['DM_MASS']<hi)
    chk(f'exponent {lo}-{hi}',cl,sl(res(d['DM_MASS'][m],[d['STELLAR_MASS'][m]]),
                                    res(d['BH_MASS'][m],[d['STELLAR_MASS'][m]])),0.02)
mh,bh=d['DM_MASS'],d['BH_MASS']; lo,hi=np.percentile(mh,[2,98]); m=(mh>lo)&(mh<hi)
chk('EAGLE full-sample exponent',0.97,sl(mh[m],bh[m]),0.02)

print('\n--- FMR ---')
def prep(fn):
    d={k:np.asarray(v,float) for k,v in pickle.load(open(P+fn+'.pkl','rb')).items()}
    ok=(d['SFR']>0)&(d['GAS_METALLICITY']>0)
    return d['STELLAR_MASS'][ok],np.log10(d['GAS_METALLICITY'][ok]),np.log10(d['SFR'][ok])-d['STELLAR_MASS'][ok]
for fn,t,clo,chi in [('eagle_final','EAGLE',-0.66,0.45),('simba_final','SIMBA',-0.51,-0.80)]:
    ms,z,ss=prep(fn)
    m1=(ms>9.0)&(ms<10.0); m2=(ms>10.5)
    chk(f'{t} FMR low',clo,pc(z[m1],ss[m1],[ms[m1]]),0.02)
    chk(f'{t} FMR high',chi,pc(z[m2],ss[m2],[ms[m2]]),0.02)

print('\n--- 1.5 dex NOISE CLAIM ---')
d=load('TNG50'); bh,ms,mh=d['BH_MASS'],d['STELLAR_MASS'],d['DM_MASS']
e=load('EAGLE'); target=pc(e['BH_MASS'],e['STELLAR_MASS'],[e['DM_MASS']])
v15=np.median([pc(bh,ms+rng.normal(0,1.5,len(ms)),[mh]) for _ in range(30)])
ok=v15<=target+0.03; CH.append(ok)
print(f'{"OK " if ok else "!! "} 1.5 dex reaches EAGLE: EAGLE={target:.3f}, TNG50+1.5dex={v15:.3f}')
print(f'\n{"="*66}\nPASSED {sum(CH)}/{len(CH)}   FAILED {len(CH)-sum(CH)}')
