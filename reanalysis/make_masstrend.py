import pickle, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL={'TNG50':'#1f77b4','EAGLE':'#d62728','SIMBA':'#2ca02c'}
def load(t):
    d=dict(pickle.load(open(f'data/{t}_clean.pkl','rb'))); return {k:np.asarray(v,float) for k,v in d.items()}
def pcorr(x,y,Z):
    X=np.column_stack([np.ones(len(x))]+list(Z))
    rx=x-X@np.linalg.lstsq(X,x,rcond=None)[0]; ry=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    return np.corrcoef(rx,ry)[0,1]
rng=np.random.default_rng(3)
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0))
bins=[(9.0,9.5),(9.5,10.0),(10.0,10.5),(10.5,11.0),(11.0,11.5)]
for t in COL:
    d=load(t); xs,ys,es=[],[],[]
    for lo,hi in bins:
        m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)&np.isfinite(d['BH_MASS'])
        if m.sum()<60: continue
        v=pcorr(d['BH_MASS'][m],d['DM_MASS'][m],[d['STELLAR_MASS'][m]])
        bs=[pcorr(d['BH_MASS'][m][i],d['DM_MASS'][m][i],[d['STELLAR_MASS'][m][i]])
            for i in (rng.integers(0,m.sum(),m.sum()) for _ in range(120))]
        xs.append((lo+hi)/2); ys.append(v); es.append(np.std(bs))
    axes[0].errorbar(xs,ys,yerr=es,fmt='o-',color=COL[t],label=t,lw=2,ms=6,capsize=3)
    # mirror
    xs2,ys2,es2=[],[],[]
    for lo,hi in bins:
        m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)&np.isfinite(d['BH_MASS'])
        if m.sum()<60: continue
        v=pcorr(d['BH_MASS'][m],d['STELLAR_MASS'][m],[d['DM_MASS'][m]])
        bs=[pcorr(d['BH_MASS'][m][i],d['STELLAR_MASS'][m][i],[d['DM_MASS'][m][i]])
            for i in (rng.integers(0,m.sum(),m.sum()) for _ in range(120))]
        xs2.append((lo+hi)/2); ys2.append(v); es2.append(np.std(bs))
    axes[1].errorbar(xs2,ys2,yerr=es2,fmt='o-',color=COL[t],label=t,lw=2,ms=6,capsize=3)
for ax,ttl,yl in [(axes[0],r'Halo at fixed galaxy',r'$r(M_{\rm BH},M_{\rm h}\,|\,M_\star)$'),
                  (axes[1],r'Galaxy at fixed halo',r'$r(M_{\rm BH},M_\star\,|\,M_{\rm h})$')]:
    ax.axhline(0,color='k',lw=.9); ax.axvspan(10.5,11.5,color='0.90',zorder=0)
    ax.set_xlabel(r'$\log M_\star\;[M_\odot]$',fontsize=11); ax.set_ylabel(yl,fontsize=11)
    ax.set_title(ttl,fontsize=11); ax.legend(frameon=False,fontsize=10); ax.set_ylim(-0.35,1.0)
plt.tight_layout(); fig.savefig('../paper/Plots/mass_trend.png',dpi=300,bbox_inches='tight',facecolor='white')
print('saved mass_trend.png')
