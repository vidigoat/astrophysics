"""Re-derive every numeric claim in main_v2.tex directly from the data."""
import pickle, numpy as np
D='reanalysis/data/'; P='Data/'
rng=np.random.default_rng(2718)
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def sd(y,X):
    A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return float(np.std(y-A@b))
def load(t,lo=9.5,hi=11.0):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)
    return {k:v[m] for k,v in d.items()}
CH=[]
def chk(label,claim,actual,tol=0.02):
    ok = abs(claim-actual)<=tol
    CH.append(ok)
    print(f'{"OK " if ok else "!! "} {label:44s} paper={claim:>8}  actual={actual:>8.3f}')

print('--- CENSORING (Table 1) ---')
for t,fn,cl_n,cl_pct,cl_seed in [('TNG50','tng50_final',10992,66.0,3740),
                                  ('EAGLE','eagle_final',11887,0.0,11885),
                                  ('SIMBA','simba_final',32757,56.5,14239)]:
    d={k:np.asarray(v,float) for k,v in pickle.load(open(P+fn+'.pkl','rb')).items()}
    n=len(d['BH_MASS']); fl=int(np.sum(d['BH_MASS']<=-9.99))
    chk(f'{t} N',cl_n,n,0); chk(f'{t} floor %',cl_pct,100*fl/n,0.1); chk(f'{t} seeded',cl_seed,n-fl,0)

print('\n--- SAMPLE SIZES in 9.5<logM*<11 ---')
for t,cl in [('TNG50',1135),('EAGLE',3800),('SIMBA',9419)]:
    chk(f'{t} N in window',cl,len(load(t)['BH_MASS']),0)

print('\n--- JOINT REGRESSION EXPONENTS ---')
for t,ca,cb in [('TNG50',0.97,-0.04),('EAGLE',0.22,0.98),('SIMBA',1.12,0.16)]:
    d=load(t); X=np.column_stack([np.ones(len(d['BH_MASS'])),d['STELLAR_MASS'],d['DM_MASS']])
    b,*_=np.linalg.lstsq(X,d['BH_MASS'],rcond=None)
    chk(f'{t} a (stellar)',ca,b[1],0.015); chk(f'{t} b (halo)',cb,b[2],0.015)

print('\n--- SCREENING TABLE ---')
for t,vals in [('TNG50',(0.876,0.799,0.599,-0.063)),('EAGLE',(0.863,0.182,0.916,0.624)),
               ('SIMBA',(0.677,0.388,0.604,0.058))]:
    d=load(t); bh,ms,mh=d['BH_MASS'],d['STELLAR_MASS'],d['DM_MASS']
    for lab,cl,act in [('r(bh,M*)',vals[0],pc(bh,ms,[])),('r(bh,M*|Mh)',vals[1],pc(bh,ms,[mh])),
                       ('r(bh,Mh)',vals[2],pc(bh,mh,[])),('r(bh,Mh|M*)',vals[3],pc(bh,mh,[ms]))]:
        chk(f'{t} {lab}',cl,act,0.01)

print('\n--- OBSERVATION TABLE (scatter reduction) ---')
for t,c1,c2 in [('TNG50',0.0,42.1),('EAGLE',18.6,1.9),('SIMBA',0.1,10.2)]:
    d=load(t); bh=d['BH_MASS']
    a=sd(bh,[d['STELLAR_MASS'],d['LOG_SIGMA']]); b=sd(bh,[d['STELLAR_MASS'],d['LOG_SIGMA'],d['DM_MASS']])
    chk(f'{t} halo adds %',c1,100*(a-b)/a,0.3)
    c=sd(bh,[d['DM_MASS']]); e=sd(bh,[d['DM_MASS'],d['STELLAR_MASS'],d['LOG_SIGMA']])
    chk(f'{t} galaxy adds %',c2,100*(c-e)/c,0.3)

print('\n--- DEFINITIONAL: r(Mhalo,Rhalf | rest) ---')
V=['STELLAR_MASS','GAS_MASS','DM_MASS','BH_MASS','HALFMASS_RAD','LOG_SIGMA','STAR_METALLICITY','GAS_METALLICITY','SSFR']
for t,cl in [('TNG50',0.976),('EAGLE',-0.130),('SIMBA',0.057)]:
    d=load(t); rest=[d[k] for k in V if k not in ('DM_MASS','HALFMASS_RAD')]
    chk(f'{t} r(Mh,Rhalf|rest)',cl,pc(d['DM_MASS'],d['HALFMASS_RAD'],rest),0.01)

print('\n--- CENSORING REVERSAL TEST ---')
raw={k:np.asarray(v,float) for k,v in pickle.load(open(P+'tng50_final.pkl','rb')).items()}
ms,mh,bh=raw['GAS_MASS'],raw['STELLAR_MASS'],raw['BH_MASS']   # corrected labels
m=(ms>9.5)&(ms<11.0)
chk('TNG50 with floored kept',0.600,pc(bh[m],mh[m],[ms[m]]),0.02)
m2=m&(bh>-9.99)
chk('TNG50 floored removed',-0.125,pc(bh[m2],mh[m2],[ms[m2]]),0.02)
e={k:np.asarray(v,float) for k,v in pickle.load(open(P+'eagle_final.pkl','rb')).items()}
me=(e['STELLAR_MASS']>9.5)&(e['STELLAR_MASS']<11.0)
chk('EAGLE comparison value',0.607,pc(e['BH_MASS'][me],e['DM_MASS'][me],[e['STELLAR_MASS'][me]]),0.02)

print('\n--- MASS RATIO EVIDENCE FOR THE BUG ---')
for t,fn,cl in [('TNG50 as labelled',None,1.24),('EAGLE',None,-2.09),('SIMBA',None,-1.99)]:
    pass
r={k:np.asarray(v,float) for k,v in pickle.load(open(P+'tng50_final.pkl','rb')).items()}
chk('TNG50 median log(M*/Mdm) as labelled',1.24,float(np.median(r['STELLAR_MASS']-r['DM_MASS'])),0.02)
chk('TNG50 median corrected',-2.18,float(np.median(r['GAS_MASS']-r['STELLAR_MASS'])),0.02)
for t,fn,cl in [('EAGLE','eagle_final',-2.09),('SIMBA','simba_final',-1.99)]:
    d={k:np.asarray(v,float) for k,v in pickle.load(open(P+fn+'.pkl','rb')).items()}
    chk(f'{t} median log(M*/Mdm)',cl,float(np.median(d['STELLAR_MASS']-d['DM_MASS'])),0.02)

print('\n--- GAS FEEDBACK FINGERPRINT (Mhalo<11.5) ---')
for t,cl in [('TNG50',-0.37),('EAGLE',0.08),('SIMBA',-0.47)]:
    d=load(t); m=d['DM_MASS']<11.5
    chk(f'{t} r(bh,Mgas|rest)',cl,pc(d['BH_MASS'][m],d['GAS_MASS'][m],
        [d['STELLAR_MASS'][m],d['DM_MASS'][m],d['LOG_SIGMA'][m]]),0.02)

print('\n--- MORPHOLOGY CONTROL ---')
for t,cl in [('TNG50',-0.030),('EAGLE',0.561),('SIMBA',0.075)]:
    d=load(t); m=(d['SSFR']>-10.6)&(d['SSFR']<-9.4)
    chk(f'{t} sSFR-matched',cl,pc(d['BH_MASS'][m],d['DM_MASS'][m],[d['STELLAR_MASS'][m]]),0.02)

print('\n--- 202 galaxies with M_BH>1e8 in TNG50 ---')
d=dict(pickle.load(open(D+'TNG50_clean.pkl','rb')))
chk('TNG50 N(M_BH>1e8)',202,int(np.sum(d['BH_MASS']>8.0)),0)

print(f'\n{"="*70}\nPASSED {sum(CH)}/{len(CH)}   FAILED {len(CH)-sum(CH)}')
