import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
R='/Users/vidigoat/astrophysics/reanalysis/results/'
D='/Users/vidigoat/astrophysics/reanalysis/data/'
rng=np.random.default_rng(7)
NICE={'STELLAR_MASS':r'$M_\star$','DM_MASS':r'$M_{\rm h}$','BH_MASS':r'$M_{\rm BH}$',
      'GAS_MASS':r'$M_{\rm gas}$','LOG_SIGMA':r'$\sigma$','STAR_METALLICITY':r'$Z_\star$',
      'GAS_METALLICITY':r'$Z_{\rm gas}$','SSFR':'sSFR'}
POS={'DM_MASS':(0,1.6),'STELLAR_MASS':(0,0.55),'BH_MASS':(-1.25,-0.35),
     'LOG_SIGMA':(1.25,-0.35),'GAS_MASS':(-1.5,0.95),'SSFR':(1.5,0.95),
     'STAR_METALLICITY':(-0.85,-1.5),'GAS_METALLICITY':(0.85,-1.5)}
COL={'TNG50':'#C1272D','EAGLE':'#0B6E99','SIMBA':'#2E7D32'}

# ---------------- FIGURE 1 : the three corrected PAGs ----------------
fig,axes=plt.subplots(1,3,figsize=(15.5,5.4))
for ax,t in zip(axes,['TNG50','EAGLE','SIMBA']):
    df=pd.read_csv(R+f'consensus_corrected_{t}.csv')
    G=nx.Graph(); [G.add_node(k) for k in POS]
    hi=[]
    for _,r in df.iterrows():
        p=r['edge'].split(); a,m,b=p[0],p[1],p[2]
        G.add_edge(a,b)
        hi.append(('BH_MASS' in (a,b)) and ('DM_MASS' in (a,b)))
    edges=list(G.edges())
    ec=[]; ew=[]
    for (a,b) in edges:
        bh_halo = {a,b}=={'BH_MASS','DM_MASS'}
        bh_star = {a,b} in ({'BH_MASS','STELLAR_MASS'},{'BH_MASS','LOG_SIGMA'})
        ec.append('#D4A017' if bh_halo else ('#333333' if bh_star else '#BBBBBB'))
        ew.append(4.2 if (bh_halo or bh_star) else 1.4)
    nx.draw_networkx_edges(G,POS,ax=ax,edgelist=edges,edge_color=ec,width=ew)
    nx.draw_networkx_nodes(G,POS,ax=ax,node_size=1500,node_color='white',
                           edgecolors=COL[t],linewidths=2.2)
    nx.draw_networkx_labels(G,POS,{k:NICE[k] for k in POS},ax=ax,font_size=12)
    ax.set_title(f'{t}   ({len(df)} edges)',fontsize=14,weight='bold',color=COL[t])
    ax.set_axis_off(); ax.set_xlim(-2.1,2.1); ax.set_ylim(-2.1,2.15)
for a in axes:
    a.text(0,-2.02,'gold: BH$-$halo    black: BH$-$galaxy ($M_\\star$ or $\\sigma$)',
           ha='center',fontsize=9.5,color='0.35')
fig.suptitle('Consensus causal graphs after correction. In TNG50 and SIMBA the black hole attaches to the galaxy;\n'
             'in EAGLE it attaches to the halo instead.',fontsize=13.5,weight='bold')
fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(R+'fig1_pags.png',dpi=165)
print('fig1 done')

# ---------------- FIGURE 5 : feedback fingerprint + FMR ----------------
def pc(x,y,Z):
    def r(v):
        A=np.column_stack([np.ones(len(v))]+list(Z)); b,*_=np.linalg.lstsq(A,v,rcond=None); return v-A@b
    return float(np.corrcoef(r(x),r(y))[0,1])
def load(t):
    d=dict(pickle.load(open(D+f'{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3)); return d
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.0))
vals,errs=[],[]
for t in ['TNG50','EAGLE','SIMBA']:
    d=load(t); m=(d['STELLAR_MASS']>9.5)&(d['STELLAR_MASS']<11.0)&(d['DM_MASS']<11.5)
    f=lambda i: pc(d['BH_MASS'][m][i],d['GAS_MASS'][m][i],
                   [d['STELLAR_MASS'][m][i],d['DM_MASS'][m][i],d['LOG_SIGMA'][m][i]])
    n=int(m.sum()); vals.append(f(np.arange(n)))
    errs.append(np.std([f(rng.choice(n,n,replace=True)) for _ in range(300)]))
a1.bar(range(3),vals,yerr=errs,capsize=6,color=[COL[t] for t in ['TNG50','EAGLE','SIMBA']],width=.62)
a1.axhline(0,color='k',lw=1)
a1.set_xticks(range(3)); a1.set_xticklabels(['TNG50\nkinetic wind','EAGLE\nthermal only','SIMBA\nkinetic jets'],fontsize=11)
a1.set_ylabel(r'$r(M_{\rm BH},\,M_{\rm gas}\ |\ M_\star,M_{\rm h},\sigma)$',fontsize=12)
a1.set_title('Gas removed by the black hole',fontsize=13,weight='bold'); a1.grid(alpha=.2,axis='y')
for i,v in enumerate(vals):
    a1.text(i, v-0.055 if v<0 else v+0.03, f'{v:+.2f}',ha='center',fontsize=12,weight='bold')

P='/Users/vidigoat/astrophysics/Data/'
def prep(d):
    ms,z,s=d['STELLAR_MASS'],d['GAS_METALLICITY'],d['SFR']
    ok=(s>0)&(z>0); return ms[ok],np.log10(z[ok]),np.log10(s[ok])-ms[ok]
BINS=[(8.5,9.0),(9.0,9.5),(9.5,10.0),(10.0,10.5),(10.5,11.0),(11.0,12.0)]
for t,fn in [('EAGLE','eagle_final'),('SIMBA','simba_final')]:
    d={k:np.asarray(v,float) for k,v in pickle.load(open(P+fn+'.pkl','rb')).items()}
    ms,z,ss=prep(d); xs,ys,es=[],[],[]
    for lo,hi in BINS:
        m=(ms>lo)&(ms<hi); n=int(m.sum())
        if n<120: continue
        f=lambda i: pc(z[m][i],ss[m][i],[ms[m][i]])
        xs.append((lo+hi)/2); ys.append(f(np.arange(n)))
        es.append(np.std([f(rng.choice(n,n,replace=True)) for _ in range(150)]))
    a2.errorbar(xs,ys,yerr=es,marker='o',ms=7,lw=2.2,capsize=3,color=COL[t],label=t)
a2.axhline(0,color='k',lw=1,ls='--')
a2.axvspan(10.5,12,color='goldenrod',alpha=.13)
a2.text(11.2,-0.9,'observed sign\nreverses here\n(Yates+2012)',ha='center',fontsize=9,color='#8a6d1f')
a2.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$',fontsize=12)
a2.set_ylabel(r'$r(Z_{\rm gas},\,{\rm sSFR}\ |\ M_\star)$',fontsize=12)
a2.set_title('The metallicity twist at high mass',fontsize=13,weight='bold')
a2.legend(frameon=False,fontsize=11); a2.grid(alpha=.2)
fig.tight_layout(); fig.savefig(R+'fig5_feedback_fmr.png',dpi=165)
print('fig5 done')
