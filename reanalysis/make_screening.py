import pickle, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
def load(t):
    d=dict(pickle.load(open(f'data/{t}_clean.pkl','rb'))); return {k:np.asarray(v,float) for k,v in d.items()}
def resid(y,Z):
    X=np.column_stack([np.ones(len(y))]+list(Z)); return y-X@np.linalg.lstsq(X,y,rcond=None)[0]
CODES=['TNG50','EAGLE','SIMBA']
fig,axes=plt.subplots(2,3,figsize=(15,8.6))
for j,t in enumerate(CODES):
    d=load(t); m=(d['STELLAR_MASS']>9.5)&(d['STELLAR_MASS']<11.0)&np.isfinite(d['BH_MASS'])
    ms,mh,bh=d['STELLAR_MASS'][m],d['DM_MASS'][m],d['BH_MASS'][m]
    for i,(cond,xlab,ylab,title) in enumerate([
        ([ms], r'$\Delta\log M_{\rm h}$ at fixed $M_\star$', r'$\Delta\log M_{\rm BH}$ at fixed $M_\star$','halo at fixed galaxy'),
        ([mh], r'$\Delta\log M_\star$ at fixed $M_{\rm h}$', r'$\Delta\log M_{\rm BH}$ at fixed $M_{\rm h}$','galaxy at fixed halo')]):
        ax=axes[i,j]
        x=resid(mh if i==0 else ms, cond); y=resid(bh,cond)
        ax.plot(x,y,'.',ms=1.6,alpha=.28,color='0.35')
        b=np.polyfit(x,y,1)[0]; r=np.corrcoef(x,y)[0,1]
        xx=np.linspace(np.percentile(x,1),np.percentile(x,99),50)
        ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),color='#C0392B',lw=2.2)
        ax.axhline(0,color='k',lw=.6); ax.axvline(0,color='k',lw=.6)
        ax.text(.04,.93,f'slope {b:+.2f}\n$r$ = {r:+.3f}',transform=ax.transAxes,fontsize=10,va='top')
        ax.set_xlabel(xlab,fontsize=10); ax.set_ylabel(ylab if j==0 else '',fontsize=10)
        ax.set_xlim(np.percentile(x,0.5),np.percentile(x,99.5)); ax.set_ylim(np.percentile(y,0.5),np.percentile(y,99.5))
        if i==0: ax.set_title(t,fontsize=13,pad=8)
plt.tight_layout(); fig.savefig('../paper/Plots/bh_screening.png',dpi=300,bbox_inches='tight',facecolor='white')
print('saved bh_screening.png  (columns: '+', '.join(CODES)+')')
