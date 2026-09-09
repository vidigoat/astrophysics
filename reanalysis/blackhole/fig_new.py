"""Three new figures for the revised paper:
  fig_obs.png        real galaxies: screening test (residual of M_BH vs residual of M_h at fixed M*)
  fig_trajectory.png the (a,b) plane: EAGLE and SIMBA through cosmic time, TNG50 and the real Universe at z=0,
                     the population model (routing_model.py) as a curve
  fig_envelope.png   90th / 10th percentile of log M_BH at fixed M200 vs redshift, EAGLE and SIMBA
"""
import os, csv, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ols
from load_simba import load_simba
from common import window
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE)); OUT=os.path.join(ROOT,"paper","Plots","v3")
PT={"TNG50":"#2ca02c","EAGLE":"#1f77b4","SIMBA":"#ff7f0e"}; OBS="#333333"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"legend.fontsize":9.5,"axes.linewidth":0.9,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
    "axes.edgecolor":"black","xtick.color":"black","ytick.color":"black","savefig.facecolor":"white","mathtext.fontset":"dejavusans","axes.spines.top":True,"axes.spines.right":True})
def rs(y,*x):
    X=np.column_stack([np.ones(len(y))]+list(x)); return y-X@np.linalg.lstsq(X,y,rcond=None)[0]


from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch
def shade(c,f):   # blend colour c toward white by fraction f
    r,g,b=to_rgb(c); return (r+(1-r)*f,g+(1-g)*f,b+(1-b)*f)
MAR="#9467bd"; GAS="#d62728"   # sample colours: purple, red
# ---------------- Fig obs
R=list(csv.DictReader(open(os.path.join(ROOT,'Data/obs/marasco2021.csv')))); G=list(csv.DictReader(open(os.path.join(ROOT,'Data/obs/gaspari2019.csv'))))
bhR=np.array([float(r['logMBH']) for r in R]); msR=np.array([float(r['logMstar']) for r in R]); mhR=np.array([float(r['logMh']) for r in R]); late=np.array([float(r['ttype'])>0 for r in R])
bhG=np.array([float(r['logMBH']) for r in G]); msG=np.array([float(r['logLK']) for r in G])+np.log10(0.75); mhG=np.array([float(r['logM500']) for r in G]); isb=np.array([r['central'].startswith('BCG') or r['central'].startswith('BGG') for r in G])
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True,gridspec_kw={"wspace":0.06})
rng=np.random.default_rng(1)
for ax,(lab,bh,ms,mh,flag,names,col) in zip(axes,[("Marasco et al. (2021)",bhR,msR,mhR,late,("early type","late type"),MAR),("Gaspari et al. (2019)",bhG,msG,mhG,isb,("isolated / satellite","brightest group or cluster galaxy"),GAS)]):
    x=rs(mh,ms); y=rs(bh,ms); xx=np.linspace(x.min()-0.15,x.max()+0.15,60)
    bs=[]
    for _ in range(2000):
        k=rng.integers(0,len(x),len(x)); (a,b),_=ols(bh[k],ms[k],mh[k]); bs.append(b)
    (a,b),_=ols(bh,ms,mh); lo,hi=np.percentile(bs,[16,84])
    ax.fill_between(xx,lo*xx,hi*xx,color=col,alpha=0.12,lw=0)
    ax.plot(xx,5/3*xx,color="black",lw=1.1,ls="--",label="halo binding energy, $b=5/3$",zorder=1)
    ax.plot(xx,0*xx,color="black",lw=1.1,ls=":",label="galaxy-regulated, $b=0$",zorder=1)
    ax.plot(xx,b*xx,color=col,lw=2.4,label=f"fit  $b={b:.2f}\\pm{np.std(bs):.2f}$",zorder=2)
    ax.scatter(x[~flag],y[~flag],s=34,color=col,edgecolor="black",lw=0.5,zorder=3,label=names[0])
    ax.scatter(x[flag],y[flag],s=34,facecolor="white",edgecolor=col,lw=1.2,zorder=3,label=names[1])
    ax.set_title(lab,fontsize=12); ax.set_xlabel(r"residual $\log M_{\rm h}$ at fixed $M_\star$")
    ax.legend(loc="upper left",frameon=False,fontsize=9)
axes[0].set_ylabel(r"residual $\log M_{\rm BH}$ at fixed $M_\star$")
fig.savefig(os.path.join(OUT,"fig_obs.png"),dpi=220,bbox_inches="tight"); plt.close(fig)



# ---------------- Fig trajectory: grouped bar chart, classic journal style
E_b={3.0:(-0.09,0.09),2.0:(0.48,0.05),1.0:(0.80,0.04),0.5:(0.92,0.04),0.0:(1.14,0.05)}
S_b={3.0:(-0.26,0.16),2.0:(-0.74,0.11),1.0:(-0.75,0.08),0.5:(-0.22,0.09),0.0:(0.52,0.10)}
S100_b={2.0:(-0.85,0.04),1.0:(-0.69,0.03),0.0:(0.40,0.03)}
E50_b={3.0:(-0.38,0.27),2.0:(0.65,0.13),1.0:(0.67,0.09),0.5:(0.91,0.10),0.0:(1.25,0.11)}
T_b=(-0.02,0.02); OBS_b=(1.62,(0.13,0.17)); OBS_ols=(1.17,0.10)
with plt.rc_context({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.9,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
                     "axes.spines.top":True,"axes.spines.right":True,"legend.fontsize":9.5,"axes.edgecolor":"black","xtick.color":"black","ytick.color":"black","mathtext.fontset":"dejavusans"}):
    fig,ax=plt.subplots(figsize=(11,5.0))
    zs=[3.0,2.0,1.0,0.5,0.0]; xg=np.arange(len(zs)); w=0.13
    series=[("EAGLE, 100 Mpc",E_b,"#1f77b4"),("EAGLE, 50 Mpc",E50_b,"#aec7e8"),("SIMBA, 50 $h^{-1}$ Mpc",S_b,"#ff7f0e"),("SIMBA, 100 $h^{-1}$ Mpc",S100_b,"#ffbb78")]
    for k,(lab,D,c) in enumerate(series):
        xs=[x+(k-2.0)*w for x,z in zip(xg,zs) if z in D]; vs=[D[z][0] for z in zs if z in D]; es=[D[z][1] for z in zs if z in D]
        ax.bar(xs,vs,width=w,color=c,edgecolor="black",linewidth=0.6,yerr=es,error_kw=dict(ecolor="black",elinewidth=1.0,capsize=2.5),label=lab,zorder=3)
    x0=xg[-1]
    ax.bar([x0+2.0*w],[T_b[0]],width=w,color="#2ca02c",edgecolor="black",linewidth=0.6,yerr=[T_b[1]],error_kw=dict(ecolor="black",elinewidth=1.0,capsize=2.5),label="TNG50",zorder=3)
    ax.bar([x0+3.0*w],[OBS_ols[0]],width=w,color="#c7c7c7",edgecolor="black",linewidth=0.6,yerr=[OBS_ols[1]],error_kw=dict(ecolor="black",elinewidth=1.0,capsize=2.5),label="real galaxies, least squares",zorder=3)
    ax.bar([x0+4.0*w],[OBS_b[0]],width=w,color="#d62728",edgecolor="black",linewidth=0.6,yerr=[[OBS_b[1][0]],[OBS_b[1][1]]],error_kw=dict(ecolor="black",elinewidth=1.0,capsize=2.5),label="real galaxies, errors-in-variables",zorder=3)
    ax.axhline(0,color="black",lw=0.9,zorder=2)
    ax.axhline(5/3,color="black",lw=1.0,ls="--",label="halo binding energy, $b=5/3$",zorder=2)
    ax.axhline(4/3,color="black",lw=1.0,ls=":",label="momentum-driven wind, $b=4/3$",zorder=2)
    for x in xg[:-1]+0.5: ax.axvline(x,color="black",lw=0.6,zorder=1)
    ax.set_xticks(xg); ax.set_xticklabels([f"$z={z:g}$" for z in zs]); ax.set_xlim(-0.5,len(zs)-0.5+0.3)
    ax.set_ylim(-1.1,2.05); ax.set_ylabel(r"halo exponent $b$   ($M_{\rm BH}\propto M_\star^{\,a}M_{\rm h}^{\,b}$ at fixed $M_\star$)")
    h,l=ax.get_legend_handles_labels(); order=[7,8,0,1,2,3,4,5,6]
    ax.legend([h[i] for i in order],[l[i] for i in order],loc="upper left",frameon=False,ncol=2,handlelength=1.8,columnspacing=1.2)
    fig.savefig(os.path.join(OUT,"fig_trajectory.png"),dpi=230,bbox_inches="tight"); plt.close(fig)

# ---------------- Fig envelope: overlapping graded ridgelines
Om,OL=0.307,0.693; E=lambda z: np.sqrt(Om*(1+z)**3+OL)
def eagle(s):
    f=os.path.join(ROOT,'Data/eagle_v3/RefL0100N1504_z0.csv') if s==28 else os.path.join(ROOT,f'Data/eagle_v3/RefL0100N1504_snap{s}.csv')
    d=pd.read_csv(f); d=d[(d.BlackHoleMass>0)&(d.Group_M_Crit200>10**11.0)&(d.Mass_Star>10**9.0)]
    return np.log10(d.BlackHoleMass.values),np.log10(d.Group_M_Crit200.values)
ZE={28:0.0,23:0.503,19:1.004,15:2.012,12:3.017}; ZS={151:0.0,125:0.49,105:0.99,78:2.02,62:3.00}
def simba(snap):
    d,meta=load_simba(os.path.join(ROOT,f"Data/simba_variants/m50n512_s50_{snap:03d}.hdf5")); m=(d["BH_MASS"]>0)&(d["STELLAR_MASS"]>9.0)
    return d["BH_MASS"][m],d["M200"][m]
from scipy.stats import gaussian_kde
BIND="#d62728"
fig,axes=plt.subplots(1,2,figsize=(12.5,6.2),sharey=True,gridspec_kw={"wspace":0.06})
H=1.25  # ridge height (overlap)
for ax,(code,loader,Z,msg) in zip(axes,[("EAGLE",eagle,ZE,"the top of the distribution rises\nwith the halo binding energy"),("SIMBA",simba,ZS,"the top of the distribution\nstays where it is")]):
    c=PT[code]; snaps=sorted(Z,key=lambda k:Z[k]); grid=np.linspace(5.0,9.8,500); p90_0=None; p90s=[]
    for j,s in enumerate(snaps):
        bh,mh=loader(s); k=(mh>=12.0)&(mh<12.5); x=bh[k]; z=Z[s]
        kde=gaussian_kde(x,bw_method=0.22)(grid); kde=kde/kde.max()*H
        base=j; col=shade(c,0.62*j/4)
        ax.fill_between(grid,base,base+kde,color=col,lw=0,zorder=20-j)
        ax.plot(grid,base+kde,color="white",lw=1.4,zorder=20-j)
        p90=np.percentile(x,90); p90s.append(p90)
        if p90_0 is None: p90_0=p90
        ax.plot([p90,p90],[base,base+0.72],color="black",lw=2.6,solid_capstyle="butt",zorder=21)
        ax.text(5.08,base+0.16,f"$z={z:.1f}$",ha="left",va="bottom",fontsize=12,color="black",zorder=22)
        ax.text(5.08,base+0.03,f"$N={k.sum()}$",ha="left",va="bottom",fontsize=9,color="0.45",zorder=22)
    zz=np.array([Z[s] for s in snaps]); yy=np.arange(len(snaps))+0.36
    ax.plot(p90_0+2/3*np.log10(E(zz)),yy,ls="--",color=BIND,lw=1.6,zorder=23)
    ax.scatter(p90_0+2/3*np.log10(E(zz)),yy,s=34,color=BIND,edgecolor="black",lw=0.6,zorder=24)
    ax.set_xlim(5.0,9.8); ax.set_ylim(-0.05,len(snaps)-1+H+0.05); ax.set_yticks([]); ax.tick_params(right=False,top=False)
    ax.set_xlabel(r"$\log\,M_{\rm BH}\;[M_\odot]$   for haloes with $10^{12}<M_{200}<10^{12.5}\,M_\odot$")
    ax.set_title(code,fontsize=13,pad=26)
    ax.text(0.5,1.015,msg.replace("\n"," "),transform=ax.transAxes,ha="center",va="bottom",fontsize=10,color="0.3",style="italic")
fig.text(0.5,0.0,"black bar: 90th percentile of each distribution.   red: the $z=0$ 90th percentile scaled by the halo binding energy, $E(z)^{2/3}$",ha="center",va="top",fontsize=10.5,color="0.35")
fig.savefig(os.path.join(OUT,"fig_envelope.png"),dpi=250,bbox_inches="tight"); plt.close(fig)
print("done")
