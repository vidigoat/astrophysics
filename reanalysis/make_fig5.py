"""Feedback fingerprint (left) and the metallicity-SFR relation (right).

The published version of the right-hand panel plotted only EAGLE and SIMBA.
TNG50 was absent, and TNG50 reproduces the Yates+2012 reversal, so its omission
made the accompanying conclusion read as a trade-off between codes when the data
show a ranking. All three codes are plotted here.
"""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

COL={'TNG50':'#1f77b4','EAGLE':'#d62728','SIMBA':'#2ca02c'}
def load(t):
    d=dict(pickle.load(open(f'data/{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    return {k:np.asarray(v,float) for k,v in d.items()}

def pcorr(x,y,Z):
    X=np.column_stack([np.ones(len(x))]+list(Z))
    rx=x-X@np.linalg.lstsq(X,x,rcond=None)[0]
    ry=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    return np.corrcoef(rx,ry)[0,1]

fig,axes=plt.subplots(1,2,figsize=(12.5,5.0))

# LEFT: r(M_BH, M_gas | M*, M_h, sigma) for M_h < 10^11.5
ax=axes[0]; vals={}
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t); m=(d['DM_MASS']<11.5)&np.isfinite(d['BH_MASS'])&np.isfinite(d['GAS_MASS'])
    vals[t]=pcorr(d['BH_MASS'][m],d['GAS_MASS'][m],
                  [d['STELLAR_MASS'][m],d['DM_MASS'][m],d['LOG_SIGMA'][m]])
ax.bar(list(vals),[vals[t] for t in vals],color=[COL[t] for t in vals],width=.6)
ax.axhline(0,color='k',lw=.9)
for i,t in enumerate(vals): ax.text(i,vals[t]+(0.02 if vals[t]>0 else -0.045),f'{vals[t]:+.2f}',ha='center',fontsize=11)
ax.set_ylabel(r'$r(M_{\rm BH},\,M_{\rm gas}\;|\;M_\star,M_{\rm h},\sigma)$',fontsize=11)
ax.set_title(r'Gas remaining at fixed mass and potential ($M_{\rm h}<10^{11.5}M_\odot$)',fontsize=11)
ax.set_ylim(min(vals.values())-.15, max(vals.values())+.15)

# RIGHT: r(Z_gas, sSFR | M*) in bins of stellar mass -- all three codes
ax=axes[1]
edges=[(9.0,9.5),(9.5,10.0),(10.0,10.5),(10.5,11.0),(11.0,12.0)]
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t); xs=[]; ys=[]
    for lo,hi in edges:
        m=(d['STELLAR_MASS']>lo)&(d['STELLAR_MASS']<hi)&np.isfinite(d['GAS_METALLICITY'])&np.isfinite(d['SSFR'])
        if m.sum()<40: continue
        xs.append((lo+hi)/2); ys.append(pcorr(d['GAS_METALLICITY'][m],d['SSFR'][m],[d['STELLAR_MASS'][m]]))
    ax.plot(xs,ys,'o-',color=COL[t],label=t,lw=2,ms=6)
    print(f'{t}: ' + '  '.join(f'{x:.2f}:{y:+.2f}' for x,y in zip(xs,ys)))
ax.axhline(0,color='k',lw=.9)
ax.axvspan(10.5,12.0,color='0.88',zorder=0)
ax.set_xlabel(r'$\log M_\star\;[M_\odot]$',fontsize=11)
ax.set_ylabel(r'$r(Z_{\rm gas},\,{\rm sSFR}\;|\;M_\star)$',fontsize=11)
ax.set_title('Metallicity--star-formation correlation at fixed stellar mass',fontsize=11)
ax.legend(frameon=False,fontsize=10)
plt.tight_layout()
fig.savefig('../paper/Plots/fig5_feedback_fmr.png',dpi=300,bbox_inches='tight',facecolor='white')
print('saved'); print('left panel:',{k:round(v,3) for k,v in vals.items()})
