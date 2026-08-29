"""DERIVATION 1: the gas-retention correction to Booth & Schaye self-regulation.
   d log M_BH / d log M_h  =  5/3 + d log f_gas / d log M_h
DERIVATION 2: fork vs chain topology, with the variance identity each implies."""
import pickle, numpy as np
D='/Users/vidigoat/astrophysics/reanalysis/data/'
rng=np.random.default_rng(77)
def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3)); return d
def sl(x,y):
    A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,y,rcond=None); return b[1]
def sl_err(x,y,reps=500):
    n=len(x); return np.std([sl(x[i],y[i]) for i in (rng.choice(n,n,replace=True) for _ in range(reps))])

print('='*84)
print('DERIVATION 1   d log M_BH/d log M_h  =  5/3 + d log f_gas/d log M_h')
print('  LHS from the M_BH-M_halo relation; RHS from the gas fraction slope. Independent.')
print('  Should hold ONLY in a halo-self-regulated code.')
print('='*84)
print(f'{"":8s}{"LHS (BH slope)":>18s}{"dlogf/dlogMh":>16s}{"RHS = 5/3+that":>18s}{"LHS-RHS":>12s}{"verdict":>14s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t)
    mh,bh,mg=d['DM_MASS'],d['BH_MASS'],d['GAS_MASS']
    lo,hi=np.percentile(mh,[3,97]); m=(mh>lo)&(mh<hi)
    lhs=sl(mh[m],bh[m]); elhs=sl_err(mh[m],bh[m])
    fg=mg[m]-mh[m]                      # log f_gas = log Mgas - log Mhalo
    dfg=sl(mh[m],fg); edfg=sl_err(mh[m],fg)
    rhs=5/3+dfg
    diff=lhs-rhs; err=np.hypot(elhs,edfg)
    v='CONSISTENT' if abs(diff)<3*err else 'FALSIFIED'
    print(f'{t:8s}{lhs:>13.3f}+-{elhs:4.3f}{dfg:>11.3f}+-{edfg:4.3f}{rhs:>18.3f}{diff:>9.3f}+-{err:4.3f}{v:>14s}')

print()
print('='*84)
print('DERIVATION 2   Fork vs chain, and the variance identity each forces.')
print('  CHAIN  M_h -> M* -> M_BH   forces  r(M_BH, M_h | M*) = 0')
print('         and   Var(M_BH|M_h) = a^2 Var(M*|M_h) + Var(M_BH|M*)   with a = dlogMBH/dlogM*')
print('  FORK   M* <- M_h -> M_BH   forces  r(M_BH, M* | M_h) = 0')
print('='*84)
LO,HI=9.5,11.0
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def resid_sd(y,X):
    A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return float(np.std(y-A@b))
print(f'{"":8s}{"r(bh,Mh|M*)":>14s}{"r(bh,M*|Mh)":>14s}{"topology":>12s}'
      f'{"Var(bh|Mh) obs":>16s}{"chain predicts":>16s}{"ratio":>8s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t); m=(d['STELLAR_MASS']>LO)&(d['STELLAR_MASS']<HI)
    bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
    r1,r2=pc(bh,mh,[ms]),pc(bh,ms,[mh])
    topo='CHAIN' if abs(r1)<abs(r2)/3 else ('FORK' if abs(r2)<abs(r1)/3 else 'mixed')
    a=sl(ms,bh)
    obs=resid_sd(bh,[mh])
    pred=np.sqrt(a**2*resid_sd(ms,[mh])**2 + resid_sd(bh,[ms])**2)
    print(f'{t:8s}{r1:>14.3f}{r2:>14.3f}{topo:>12s}{obs:>16.3f}{pred:>16.3f}{obs/pred:>8.3f}')

print()
print('='*84)
print('DERIVATION 3   If TNG50 is a pure chain M_h -> M* -> M_BH, then the M_BH-M_halo')
print('  slope must EQUAL the product of the two link slopes.  Test the composition law.')
print('='*84)
print(f'{"":8s}{"b=dM*/dMh":>12s}{"a=dMBH/dM*":>13s}{"a*b":>10s}{"observed dMBH/dMh":>20s}{"verdict":>13s}')
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t); m=(d['STELLAR_MASS']>LO)&(d['STELLAR_MASS']<HI)
    bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
    b_=sl(mh,ms); a_=sl(ms,bh); obs=sl(mh,bh)
    e=np.hypot(sl_err(mh,bh), abs(a_)*sl_err(mh,ms))
    v='CHAIN HOLDS' if abs(a_*b_-obs)<3*e else 'extra path'
    print(f'{t:8s}{b_:>12.3f}{a_:>13.3f}{a_*b_:>10.3f}{obs:>15.3f}+-{e:4.3f}{v:>13s}')
