"""What would the headline test have returned WITHOUT each fix?
Fix 1: TNG50 column permutation.  Fix 2: remove censored (floored) black holes."""
import pickle, numpy as np
P='/Users/vidigoat/astrophysics/Data/'
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def jointfit(bh,ms,mh):
    X=np.column_stack([np.ones(len(bh)),ms,mh]); b,*_=np.linalg.lstsq(X,bh,rcond=None); return b[1],b[2]

raw={k:np.asarray(v,float) for k,v in pickle.load(open(P+'tng50_final.pkl','rb')).items()}
E={k:np.asarray(v,float) for k,v in pickle.load(open(P+'eagle_final.pkl','rb')).items()}

# ---- scenario A: exactly as submitted (wrong labels, censored BH kept)
A_ms,A_mh,A_bh=raw['STELLAR_MASS'],raw['DM_MASS'],raw['BH_MASS']
# ---- scenario B: labels fixed, censoring NOT fixed
B_ms,B_mh,B_bh=raw['GAS_MASS'],raw['STELLAR_MASS'],raw['BH_MASS']
# ---- scenario C: both fixed
seed=raw['BH_MASS']>-9.99
C_ms,C_mh,C_bh=raw['GAS_MASS'][seed],raw['STELLAR_MASS'][seed],raw['BH_MASS'][seed]

print('TNG50, the same headline test under three scenarios')
print('='*76)
for lab,(ms,mh,bh) in [('A  as submitted (both bugs)',(A_ms,A_mh,A_bh)),
                       ('B  labels fixed only',(B_ms,B_mh,B_bh)),
                       ('C  labels + censoring fixed',(C_ms,C_mh,C_bh))]:
    m=(ms>9.5)&(ms<11.0)
    if m.sum()<100:
        print(f'{lab:30s}  only {int(m.sum())} galaxies in the window'); continue
    a,b=jointfit(bh[m],ms[m],mh[m])
    r1=pc(bh[m],mh[m],[ms[m]]); r2=pc(bh[m],ms[m],[mh[m]])
    print(f'{lab:30s} N={int(m.sum()):5d}   M_BH ~ M*^{a:+.2f} Mh^{b:+.2f}   '
          f'r(bh,Mh|M*)={r1:+.3f}  r(bh,M*|Mh)={r2:+.3f}')

print()
m=(E['STELLAR_MASS']>9.5)&(E['STELLAR_MASS']<11.0)&(E['BH_MASS']>-9.99)
a,b=jointfit(E['BH_MASS'][m],E['STELLAR_MASS'][m],E['DM_MASS'][m])
print(f'{"EAGLE (never had the bug)":30s} N={int(m.sum()):5d}   M_BH ~ M*^{a:+.2f} Mh^{b:+.2f}   '
      f'r(bh,Mh|M*)={pc(E["BH_MASS"][m],E["DM_MASS"][m],[E["STELLAR_MASS"][m]]):+.3f}')
print()
print('In scenario A the variable called "STELLAR_MASS" is really the halo and')
print('"DM_MASS" is really the gas, so the numbers are not the quantities named.')
