import pickle, numpy as np
D='/Users/vidigoat/astrophysics/reanalysis/data/'
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
BINS=[(9.0,9.5),(9.5,10.0),(10.0,10.5),(10.5,11.0),(11.0,12.0)]
print('Does the fingerprint depend on stellar mass?  r(M_BH, M_halo | M*) per bin.')
print('This says WHERE an observational test would have the most power.\n')
print(f'{"log M* bin":>14s}'+''.join(f'{t:>22s}' for t in ['TNG50','EAGLE','SIMBA']))
print(f'{"":>14s}'+''.join(f'{"r(bh,Mh|M*)":>13s}{"N":>9s}' for _ in range(3)))
for lo,hi in BINS:
    row=f'{f"{lo}-{hi}":>14s}'
    for t in ['TNG50','EAGLE','SIMBA']:
        d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
        m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)
        n=int(m.sum())
        if n<80: row+=f'{"--":>13s}{n:>9d}'; continue
        row+=f'{pc(d["BH_MASS"][m],d["DM_MASS"][m],[d["STELLAR_MASS"][m]]):>13.3f}{n:>9d}'
    print(row)

print()
print('Same, but the mirror:  r(M_BH, M* | M_halo)')
print(f'{"log M* bin":>14s}'+''.join(f'{t:>22s}' for t in ['TNG50','EAGLE','SIMBA']))
for lo,hi in BINS:
    row=f'{f"{lo}-{hi}":>14s}'
    for t in ['TNG50','EAGLE','SIMBA']:
        d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
        m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)
        n=int(m.sum())
        if n<80: row+=f'{"--":>13s}{n:>9d}'; continue
        row+=f'{pc(d["BH_MASS"][m],d["STELLAR_MASS"][m],[d["DM_MASS"][m]]):>13.3f}{n:>9d}'
    print(row)
