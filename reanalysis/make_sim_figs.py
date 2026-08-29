import pickle, io, contextlib, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pytetrad.tools import TetradSearch as ts
from render_pag import best_circle_layout, draw

V=['STELLAR_MASS','DM_MASS','BH_MASS','GAS_MASS','LOG_SIGMA',
   'STAR_METALLICITY','GAS_METALLICITY','SSFR']
SHORT={'STELLAR_MASS':r'$M_\star$','DM_MASS':r'$M_{\rm h}$','BH_MASS':r'$M_{\rm BH}$',
       'GAS_MASS':r'$M_{\rm gas}$','LOG_SIGMA':r'$\sigma$','STAR_METALLICITY':r'$Z_\star$',
       'GAS_METALLICITY':r'$Z_{\rm gas}$','SSFR':'sSFR'}
BARY=[v for v in V if v!='DM_MASS']

def load(t):
    d=dict(pickle.load(open(f'data/{t}_clean.pkl','rb')))
    d['LOG_SIGMA']=np.log10(np.maximum(d['VEL_DISP'],1e-3))
    return {k:np.asarray(d[k],float) for k in V}

def graph(d,idx):
    df=pd.DataFrame({k:d[k][idx] for k in V}); df=(df-df.mean())/df.std()
    s=ts.TetradSearch(df); s.set_verbose(False)
    s.add_to_tier(0,'DM_MASS')
    for v in BARY: s.add_to_tier(1,v)
    s.use_basis_function_lrt(truncation_limit=7,alpha=0.01)
    s.use_basis_function_bic(truncation_limit=7,penalty_discount=5)
    with contextlib.redirect_stdout(io.StringIO()): s.run_fcit()
    E=[]
    for line in str(s.get_java()).split('\n'):
        p=line.strip().split()
        if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
            E.append((p[1],p[2],p[3]))
    return E

data={t:load(t) for t in ['TNG50','EAGLE','SIMBA']}
N=min(len(d['STELLAR_MASS']) for d in data.values())
rng=np.random.default_rng(4242)
fig,axes=plt.subplots(1,3,figsize=(19,6.6))
summary={}
for ax,t in zip(axes,['TNG50','EAGLE','SIMBA']):
    idx=rng.permutation(len(data[t]['STELLAR_MASS']))[:N]
    E=graph(data[t],idx)
    hl={tuple(sorted((SHORT['BH_MASS'],SHORT['DM_MASS']))),
        tuple(sorted((SHORT['BH_MASS'],SHORT['STELLAR_MASS'])))}
    Es=[(SHORT[a],m,SHORT[b]) for a,m,b in E]
    nodes=sorted({v for a,_,b in Es for v in (a,b)})
    pos,cr=best_circle_layout(nodes,Es)
    draw(ax,Es,pos,title=t,highlight=hl,nodefs=12)
    nor=sum(1 for e in E if '>' in e[1] or '<' in e[1])
    bh_h=any({'BH_MASS','DM_MASS'}=={a,b} for a,_,b in E)
    bh_s=any({'BH_MASS','STELLAR_MASS'}=={a,b} for a,_,b in E)
    summary[t]=(len(E),nor,bh_h,bh_s,cr)
    print(f'{t}: {len(E)} edges, {nor} oriented, BH-halo={bh_h}, BH-stellar={bh_s}, crossings={cr}')
plt.tight_layout()
fig.savefig('../paper/Plots/fig1_pags.png',dpi=300,bbox_inches='tight',facecolor='white')
print('saved -> paper/Plots/fig1_pags.png   N =',N)
