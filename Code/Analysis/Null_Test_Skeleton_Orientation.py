"""
Strengthened, honest null test. Separates skeleton (adjacency) agreement from
ORIENTATION agreement, and measures orientation agreement only among edges that
BOTH graphs orient (removing the undirected-edge scoring noise that biases the
raw same-direction fraction at small N). Reports:
  - within-code, disjoint N=5,000 halves (fully independent) for all three codes
  - within-code, disjoint N=10,000 halves for SIMBA (gold standard, only code with
    a big enough parent for two independent full-size draws)
  - cross-code at N=10,000 (the main-analysis draws, seed 42)
Bootstrap 95% CIs on the orientation-agreement metric.
Output: Results/null_strong.csv + printed summary.
"""
import os, pickle
from collections import Counter
import numpy as np, pandas as pd
import pytetrad.tools.TetradSearch as ts

import os as _os
REPO=_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); DATA=REPO+"/Data"; RESULTS=REPO+"/Results"
ALPHA,PRES,ORI,NRUN,T,P=0.01,0.50,0.60,40,7,8
CODES=["TNG50","EAGLE","SIMBA"]
BN={"DM_MASS":"halo_mass","STELLAR_MASS":"stellar_mass","GAS_MASS":"gas_mass","BH_MASS":"bh_mass",
    "BARYONIC_MASS":"baryon_mass","HALFMASS_RAD":"size","VEL_DISP":"veldisp","STAR_METALLICITY":"Z_star",
    "GAS_METALLICITY":"Z_gas","PHOTOMETRIC_R":"r_mag","PHOTOMETRIC_U":"u_mag","COLOUR":"colour","SFR":"sfr"}
_M=("<->","-->","<--","o->","<-o","o-o"); _FL={"-->":"<--","<--":"-->","o->":"<-o","<-o":"o->","o-o":"o-o","<->":"<->"}
_dir=lambda m:"fwd" if m in("-->","o->") else "rev" if m in("<--","<-o") else "und"

def zc(x): x=np.asarray(x,float); s=np.std(x); return (x-np.mean(x))/(s if s>0 else 1.0)
def load(code):
    d=pickle.load(open(f"{DATA}/{code.lower()}_final.pkl","rb")); o={}
    for k,nm in BN.items():
        v=np.asarray(d[k],float)
        if nm in("size","sfr"): v=np.log10(np.maximum(v,1e-3))
        o[nm]=v
    m=np.ones(len(o["stellar_mass"]),bool)
    for v in o.values(): m&=np.isfinite(v)
    o={k:zc(v[m]) for k,v in o.items()}; return o,len(o["stellar_mass"])
def frame(o,idx): return pd.DataFrame({k:v[idx] for k,v in o.items()})
def parse(g):
    e=[];inb=False
    for raw in g.split("\n"):
        s=raw.strip()
        if s.startswith("Graph Edges"): inb=True;continue
        if not inb: continue
        if s.startswith("Graph "): break
        for mk in _M:
            if mk in s:
                b=s.split(".",1)[1].strip() if "." in s.split(mk)[0] else s
                l,r=b.split(mk,1);a,bb=l.strip(),r.strip()
                e.append(((a,bb),mk) if a<=bb else ((bb,a),_FL[mk]));break
    return e
def consensus(df):
    pr,mk=Counter(),{}
    for _ in range(NRUN):
        S=ts.TetradSearch(df);S.set_verbose(False)
        S.use_basis_function_lrt(truncation_limit=T,alpha=ALPHA)
        S.use_basis_function_bic(truncation_limit=T,penalty_discount=P)
        S.run_fcit()
        for k,c in parse(str(S.get_java())): pr[k]+=1;mk.setdefault(k,Counter())[c]+=1
    g={}
    for k,c in pr.items():
        if c/NRUN<PRES: continue
        top,tn=mk[k].most_common(1)[0]; g[k]=top if tn/c>=ORI else "o-o"
    return g

def orient_vec(g1,g2):
    o1={k:_dir(v) for k,v in g1.items() if _dir(v)!="und"}
    o2={k:_dir(v) for k,v in g2.items() if _dir(v)!="und"}
    both=sorted(set(o1)&set(o2))
    return np.array([1 if o1[k]==o2[k] else 0 for k in both])
def skel_jac(g1,g2):
    s1,s2=set(g1),set(g2); return len(s1&s2)/len(s1|s2) if (s1|s2) else 0.0
def ci(v,B=4000):
    if len(v)==0: return (float('nan'),float('nan'),float('nan'))
    rs=np.random.RandomState(7); f=v.mean()
    s=[v[rs.randint(0,len(v),len(v))].mean() for _ in range(B)]
    return f,float(np.percentile(s,2.5)),float(np.percentile(s,97.5))

def main():
    std={c:load(c) for c in CODES}
    for c in CODES: print(f"{c}: parent {std[c][1]}")
    rows=[]

    # within-code disjoint 5k (all three)
    g5={}
    for c in CODES:
        o,n=std[c]; idx=np.random.RandomState(1).choice(n,10000,replace=False)
        g5[(c,1)]=consensus(frame(o,idx[:5000])); g5[(c,2)]=consensus(frame(o,idx[5000:10000]))
    print("\n== WITHIN-CODE, disjoint 5k, ORIENTATION agreement (oriented-in-both) ==")
    for c in CODES:
        v=orient_vec(g5[(c,1)],g5[(c,2)]); f,lo,hi=ci(v); j=skel_jac(g5[(c,1)],g5[(c,2)])
        rows.append(dict(kind="within5k",comp=c,skel_jac=round(j,3),n_or_both=len(v),
                         orient_agree=round(f,3),lo=round(lo,3),hi=round(hi,3)))
        print(f"  {c}: orient {int(v.sum())}/{len(v)}={f:.2f} [{lo:.2f}-{hi:.2f}]  skelJaccard={j:.2f}")

    # SIMBA gold standard: two disjoint 10k
    o,n=std["SIMBA"]; idx=np.random.RandomState(2).choice(n,20000,replace=False)
    gs1=consensus(frame(o,idx[:10000])); gs2=consensus(frame(o,idx[10000:20000]))
    v=orient_vec(gs1,gs2); f,lo,hi=ci(v); j=skel_jac(gs1,gs2)
    rows.append(dict(kind="within10k",comp="SIMBA",skel_jac=round(j,3),n_or_both=len(v),
                     orient_agree=round(f,3),lo=round(lo,3),hi=round(hi,3)))
    print(f"\n== SIMBA GOLD STANDARD, two disjoint 10k ==\n  SIMBA: orient {int(v.sum())}/{len(v)}={f:.2f} [{lo:.2f}-{hi:.2f}]  skelJaccard={j:.2f}")

    # cross-code at 10k (main-analysis draws, seed 42)
    g10={c:consensus(frame(std[c][0],np.random.RandomState(42).choice(std[c][1],10000,replace=False))) for c in CODES}
    print("\n== CROSS-CODE at 10k, ORIENTATION agreement (oriented-in-both) ==")
    for i in range(3):
        for j2 in range(i+1,3):
            a,b=CODES[i],CODES[j2]; v=orient_vec(g10[a],g10[b]); f,lo,hi=ci(v); jc=skel_jac(g10[a],g10[b])
            rows.append(dict(kind="cross10k",comp=f"{a}-{b}",skel_jac=round(jc,3),n_or_both=len(v),
                             orient_agree=round(f,3),lo=round(lo,3),hi=round(hi,3)))
            print(f"  {a}-{b}: orient {int(v.sum())}/{len(v)}={f:.2f} [{lo:.2f}-{hi:.2f}]  skelJaccard={jc:.2f}")

    pd.DataFrame(rows).to_csv(RESULTS+"/null_strong.csv",index=False)
    win=[r["orient_agree"] for r in rows if r["kind"].startswith("within") and r["n_or_both"]>0]
    cro=[r["orient_agree"] for r in rows if r["kind"]=="cross10k" and r["n_or_both"]>0]
    print(f"\nMEAN within-code orientation agreement: {np.nanmean(win):.2f}")
    print(f"MEAN cross-code orientation agreement:  {np.nanmean(cro):.2f}")
    print("Chance = 0.50 (two oriented edges agreeing by chance)")
    print("DONE")

if __name__=="__main__": main()
