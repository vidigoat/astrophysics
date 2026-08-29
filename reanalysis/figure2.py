import pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D='/Users/vidigoat/astrophysics/reanalysis/data/'
rng=np.random.default_rng(5)
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
BINS=[(9.0,9.5),(9.5,10.0),(10.0,10.5),(10.5,11.0),(11.0,12.0)]
CEN=[np.mean(b) for b in BINS]
COL={'TNG50':'#C1272D','EAGLE':'#0B6E99','SIMBA':'#2E7D32'}
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0),sharey=True)
for ax,(what,lab) in zip(axes,[('halo',r'$r(M_{\rm BH},\,M_{\rm halo}\ |\ M_\star)$'),
                               ('star',r'$r(M_{\rm BH},\,M_\star\ |\ M_{\rm halo})$')]):
    for t in ['TNG50','EAGLE','SIMBA']:
        d=pickle.load(open(D+f'{t}_clean.pkl','rb'))
        xs,ys,es=[],[],[]
        for (lo,hi),c in zip(BINS,CEN):
            m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi); n=int(m.sum())
            if n<80: continue
            bh,ms,mh=d['BH_MASS'][m],d['STELLAR_MASS'][m],d['DM_MASS'][m]
            f=(lambda i: pc(bh[i],mh[i],[ms[i]])) if what=='halo' else (lambda i: pc(bh[i],ms[i],[mh[i]]))
            v=f(np.arange(n))
            bs=[f(rng.choice(n,n,replace=True)) for _ in range(200)]
            xs.append(c); ys.append(v); es.append(np.std(bs))
        ax.errorbar(xs,ys,yerr=es,marker='o',ms=7,lw=2.2,capsize=3,color=COL[t],label=t)
    ax.axhline(0,color='grey',lw=.8,ls=':')
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$',fontsize=12)
    ax.set_title(lab,fontsize=13)
    ax.grid(alpha=.2)
axes[0].set_ylabel('partial correlation',fontsize=12)
axes[0].legend(frameon=False,fontsize=11,loc='upper left')
axes[0].axvspan(10.5,12.0,color='goldenrod',alpha=.12)
axes[0].text(11.25,-0.30,'where dynamical\n$M_{\\rm BH}$ measurements\nactually exist',
             ha='center',fontsize=9,color='#8a6d1f')
fig.suptitle('The codes disagree most exactly where the data is',fontsize=14,weight='bold')
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig('/Users/vidigoat/astrophysics/reanalysis/results/mass_trend.png',dpi=170)
print('saved')
