import pickle, numpy as np
D='/Users/vidigoat/astrophysics/reanalysis/data/'
def res(y,X):
    A=np.column_stack([np.ones(len(y))]+list(X)); b,*_=np.linalg.lstsq(A,y,rcond=None); return y-A@b
def sl(x,y):
    A=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(A,y,rcond=None); return b[1]
print('DIRECT halo exponent  d log M_BH / d log M_halo  at fixed M*, per halo-mass bin.')
print('Booth & Schaye self-regulation predicts 5/3 = 1.67 once the BH can unbind the')
print('halo atmosphere.  A shallower value means growth is not yet halo-limited.\n')
BINS=[(10.5,11.0),(11.0,11.5),(11.5,12.0),(12.0,12.5),(12.5,14.0)]
print(f'{"log M_halo":>14s}'+''.join(f'{t:>20s}' for t in ['TNG50','EAGLE','SIMBA']))
for lo,hi in BINS:
    row=f'{f"{lo}-{hi}":>14s}'
    for t in ['TNG50','EAGLE','SIMBA']:
        d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
        m=(d['DM_MASS']>lo)&(d['DM_MASS']<hi); n=int(m.sum())
        if n<100: row+=f'{"-- (%d)"%n:>20s}'; continue
        bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
        row+=f'{sl(res(mh,[ms]),res(bh,[ms])):>13.2f}{f"({n})":>7s}'
    print(row)
print()
print('And the TOTAL (unconditioned) halo exponent, same bins:')
print(f'{"log M_halo":>14s}'+''.join(f'{t:>20s}' for t in ['TNG50','EAGLE','SIMBA']))
for lo,hi in BINS:
    row=f'{f"{lo}-{hi}":>14s}'
    for t in ['TNG50','EAGLE','SIMBA']:
        d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
        m=(d['DM_MASS']>lo)&(d['DM_MASS']<hi); n=int(m.sum())
        if n<100: row+=f'{"--":>20s}'; continue
        row+=f'{sl(d["DM_MASS"][m],d["BH_MASS"][m]):>13.2f}{f"({n})":>7s}'
    print(row)
