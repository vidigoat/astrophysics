"""Black-hole mass ceiling from the binding energy of the COOLING atmosphere.

Halo: NFW with c(M,z) (Duffy+2008 relaxed, 200c).  Hot gas: same shape as the NFW density
(scaled), total f_gas(M) M200 within R200, isothermal at T = mu m_p V200^2 / (2 k) * 1.5 (Tvir).
Cooling: Tozzi & Norman 2001 fit at Z = 0.3 Zsun.  r_cool: t_cool(r) = t_H(z) = 1/H(z).
E_cool = integral_0^{r_cool} G M(<r) rho_gas(r) 4 pi r dr    (Marasco+2021 eq. E_cool)
E_200  = same integral to R200.
Ceiling: eps M_BH c^2 = E  ->  M_BH,max = E / (eps c^2).
Outputs slope d log M_BH,max / d log M200 and the redshift shift at fixed M200 for both closures.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
G=6.674e-8; Msun=1.989e33; kpc=3.086e21; mp=1.673e-24; kB=1.381e-16; c_light=2.998e10; mu=0.59; X=0.75
Om,OL,h=0.307,0.693,0.6777
def E(z): return np.sqrt(Om*(1+z)**3+OL)
def Hz(z): return 100*h*E(z)*1e5/(1e3*kpc)          # s^-1
def rho_c(z): return 3*Hz(z)**2/(8*np.pi*G)
def conc(M,z): return 5.71*(M/(2e12/h))**(-0.084)*(1+z)**(-0.47)      # Duffy+08 M200c relaxed
def fgas(M): return np.clip(0.10*(M/1e14)**0.25,0.02,0.156)             # hot-gas fraction within R200
def Lambda(T):   # Tozzi & Norman 2001, Z=0.3 Zsun, erg cm^3 s^-1 (per n_e n_H); kT in keV
    kT=kB*T/1.602e-9
    return 1e-22*(8.6e-3*kT**-1.7+5.8e-2*kT**0.5+6.3e-2)
def halo(M,z):
    R=(3*M/(4*np.pi*200*rho_c(z)))**(1/3); cc=conc(M,z); rs=R/cc
    m=lambda x: np.log(1+x)-x/(1+x)
    Mr=lambda r: M*m(r/rs)/m(cc)
    rho_nfw=lambda r: M/(4*np.pi*rs**3*m(cc))/((r/rs)*(1+r/rs)**2)
    fg=fgas(M); rho_gas=lambda r: fg*rho_nfw(r)
    V200=np.sqrt(G*M/R); T=1.5*mu*mp*V200**2/(2*kB)     # ~ T_vir (factor 1.5 -> matches T500-M scaling roughly)
    return R,Mr,rho_gas,T
def r_cool(M,z):
    R,Mr,rho_gas,T=halo(M,z); tH=1/Hz(z)
    def tcool(r):
        n=rho_gas(r)/(mu*mp); ne=n*(X+1)/2*0.9; nH=n*X*0.9   # rough
        return 1.5*n*kB*T/(ne*nH*Lambda(T))
    f=lambda r: np.log(tcool(r)/tH)
    if f(R)<0: return R
    if f(1e-3*R)>0: return 1e-3*R
    return brentq(f,1e-3*R,R)
def Ebind(M,z,rmax):
    R,Mr,rho_gas,T=halo(M,z)
    return quad(lambda r: G*Mr(r)*rho_gas(r)*4*np.pi*r,1e-4*R,rmax,limit=200)[0]
def ceilings(M,z):
    R=halo(M,z)[0]; rc=r_cool(M,z)
    return Ebind(M,z,rc),Ebind(M,z,R),rc/R,halo(M,z)[3]
if __name__=='__main__':
    Ms=10**np.arange(12.0,14.6,0.5)*Msun
    print("z   logM200   T[keV]  r_cool/R200   logE_cool  logE_200   [erg]")
    tab={}
    for z in [0,1,2,3]:
        for M in Ms:
            Ec,E2,fr,T=ceilings(M,z); tab[(z,np.log10(M/Msun))]=(Ec,E2)
            print(f"{z}   {np.log10(M/Msun):5.1f}   {kB*T/1.602e-9:5.2f}    {fr:6.3f}      {np.log10(Ec):6.2f}     {np.log10(E2):6.2f}")
    print("\nceiling slope d logE / d logM200 over 12.5-14.0:")
    for z in [0,1,2,3]:
        for lab,k in [('cooling',0),('R200',1)]:
            s=(np.log10(tab[(z,14.0)][k])-np.log10(tab[(z,12.5)][k]))/1.5
            print(f"  z={z} {lab:8s}: {s:.2f}")
    print("\nredshift shift at fixed M200=10^13 relative to z=0 (dex):   [pure binding prediction 2/3 logE(z)]")
    for z in [1,2,3]:
        print(f"  z={z}: cooling {np.log10(tab[(z,13.0)][0]/tab[(0,13.0)][0]):+.2f}   R200 {np.log10(tab[(z,13.0)][1]/tab[(0,13.0)][1]):+.2f}   2/3logE {2/3*np.log10(E(z)):+.2f}")
    # normalisation: eps needed for the observed envelope logMBH(M200=1e13) ~ 8.9 (Gaspari 90th pct at 13)
    Ec,E2,_,_=ceilings(1e13*Msun,0)
    for lab,Eb in [('cooling',Ec),('R200',E2)]:
        eps=Eb/(10**8.9*Msun*c_light**2); print(f"\ncoupling eps = E/(M_BH c^2) needed for the observed envelope at 1e13 ({lab}): {eps:.4f}")
