"""Regenerate the TNG50 corner plot from CORRECTED data.

The published figure was built from Data/tng50_final.pkl, in which the Gadget
SubhaloMassType indices were permuted (median STELLAR 10.28 > DM 9.03, i.e.
stars outweighing the halo). This uses reanalysis/data/TNG50_clean.pkl, in
which DM 10.28 > GAS 9.03 > STELLAR 8.10, and shows the eight variables the
cross-code analysis actually uses.
"""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

V=['STELLAR_MASS','DM_MASS','GAS_MASS','BH_MASS','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
LAB={'STELLAR_MASS':r'$\log M_\star$','DM_MASS':r'$\log M_{\rm h}$',
     'GAS_MASS':r'$\log M_{\rm gas}$','BH_MASS':r'$\log M_{\rm BH}$',
     'LOG_SIGMA':r'$\log \sigma$','STAR_METALLICITY':r'$Z_\star$',
     'GAS_METALLICITY':r'$Z_{\rm gas}$','SSFR':r'$\log\,{\rm sSFR}$'}
Q=(0.393,0.865,0.989); C=['#1B5E20','#2E7D32','#81C784']

d=dict(pickle.load(open('reanalysis/data/TNG50_clean.pkl','rb')))
d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
X={k:np.asarray(d[k],float) for k in V}
ok=np.all([np.isfinite(X[k]) for k in V],axis=0)
X={k:v[ok] for k,v in X.items()}
# axis ranges clipped to the 0.5-99.5 percentile so a handful of extreme
# low-metallicity objects do not compress every panel in that row/column
LIM={k:(np.percentile(v,0.5),np.percentile(v,99.5)) for k,v in X.items()}
n=len(X['STELLAR_MASS']); print('N =',n)
for k in V: print('  %-18s median %7.2f' % (k, np.median(X[k])))

m=len(V)
fig,ax=plt.subplots(m,m,figsize=(2.0*m,2.0*m))
plt.rcParams.update({'font.size':11})
rng=np.random.default_rng(0)
sub=rng.permutation(n)[:min(n,2500)]
for i in range(m):
    for j in range(m):
        a=ax[i,j]
        if j>i: a.axis('off'); continue
        if i==j:
            a.hist(X[V[i]],bins=40,color='#B8E0B8',edgecolor='#1B3D1B',lw=.6)
            a.set_yticks([])
        else:
            x,y=X[V[j]][sub],X[V[i]][sub]
            try:
                kde=st.gaussian_kde(np.vstack([x,y]))
                xg=np.linspace(x.min(),x.max(),60); yg=np.linspace(y.min(),y.max(),60)
                XX,YY=np.meshgrid(xg,yg); Z=kde(np.vstack([XX.ravel(),YY.ravel()])).reshape(60,60)
                Zs=np.sort(Z.ravel())[::-1]; cs=np.cumsum(Zs); cs/=cs[-1]
                lv=sorted({float(Zs[np.searchsorted(cs,q)]) for q in Q})
                if len(lv)>1: a.contourf(XX,YY,Z,levels=lv+[Z.max()],colors=C[-len(lv):])
            except Exception:
                a.plot(x,y,'.',ms=1,color='#4A9D4E',alpha=.3)
        a.set_xlim(*LIM[V[j]])
        if i!=j: a.set_ylim(*LIM[V[i]])
        if i==m-1: a.set_xlabel(LAB[V[j]],fontsize=11)
        else: a.set_xticklabels([])
        if j==0 and i>0: a.set_ylabel(LAB[V[i]],fontsize=11)
        elif j>0: a.set_yticklabels([])
        a.tick_params(labelsize=8)
plt.tight_layout(pad=0.3)
fig.savefig('paper/Plots/corner_tng50_FINAL.png',dpi=300,bbox_inches='tight',facecolor='white')
print('saved -> paper/Plots/corner_tng50_FINAL.png')
