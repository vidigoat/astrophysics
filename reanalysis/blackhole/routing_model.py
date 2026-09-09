"""Population model for the routing rotation of the black hole - galaxy - halo connection.

Haloes: z=0 masses log-uniform in [11.3, 14.0] (N=6000); mean growth histories integrated backwards
with the Fakhouri, Ma & Boylan-Kolchin (2010) mean accretion rate.
Galaxies: M*(M_h, z) from Moster et al. (2013) with 0.2 dex lognormal scatter (fixed per halo).
  Moster's slope in the analysis window swings from 0.6 at z=0 to 1.2 at z=3, which by itself
  drives the total slope T.  The paper therefore uses SHMR_SLOPE = 0.88, EAGLE's measured value,
  pinned to Moster's normalisation at 1e12 at each epoch; set SHMR_SLOPE = None for plain Moster.
Black holes: seeded at 1e5 Msun when M* first exceeds 10^9.3.  Growth law (supply-limited):
    dM_BH/dt = A * M_BH^p * SFR-proxy      with SFR-proxy = dM*/dt
  p = 1 gives Bondi-like runaway in a growing galaxy (M_BH ∝ M*^2 in the unregulated regime).
Ceiling: M_BH <= M_ceil,   M_ceil = K * M_h^beta * E(z)^k    (halo ceiling: beta=5/3, k=2/3)
                        or M_ceil = K' * M*^beta'           (galaxy ceiling, k=0)
At each z the population conditional exponents (a, b) are measured in 9.5 < log M* < 11, M_h > 10^10.5.
"""
import numpy as np
Om,OL,h=0.307,0.693,0.6777
E=lambda z: np.sqrt(Om*(1+z)**3+OL)
def tH_gyr(z):   # lookback-free cosmic time: t(z) in Gyr (flat LCDM analytic)
    return 2/(3*np.sqrt(OL))*np.arcsinh(np.sqrt(OL/Om)*(1+z)**-1.5)/(h*100/978.0)  # 1/H0 = 978/h Gyr
SHMR_SLOPE=0.88          # EAGLE's measured d log M* / d log M200 in the window; None -> plain Moster
def moster_raw(Mh,z):
    x=z/(1+z); N=0.0351-0.0247*x; M1=10**(11.590+1.195*x); be=1.376-0.826*x; ga=0.608+0.329*x
    return Mh*2*N/((Mh/M1)**-be+(Mh/M1)**ga)
def moster(Mh,z):
    if SHMR_SLOPE is None: return moster_raw(Mh,z)
    return moster_raw(1e12,z)*(Mh/1e12)**SHMR_SLOPE
def dMdt(M,z):   # Fakhouri+10 mean, Msun/yr
    return 46.1*(M/1e12)**1.1*(1+1.11*z)*E(z)
def run(K=None,beta=5/3,k=2/3,galaxy_ceiling=False,Kg=None,betag=1.0,A=None,p=1.0,seed=0,N=6000,zmax=6.0,nz=400):
    rng=np.random.default_rng(seed)
    logM0=rng.uniform(11.3,14.0,N); M=10**logM0
    zs=np.linspace(0,zmax,nz)[::-1]              # from high z to 0
    # integrate halo mass backwards to zmax
    Mz=np.empty((nz,N)); Mz[-1]=M
    for i in range(nz-1,0,-1):                     # zs[i] -> zs[i-1] is forward in time; we go backward
        pass
    # simpler: backward integration from z=0
    Mb=M.copy(); hist={0:M.copy()}
    zgrid=np.linspace(0,zmax,nz)
    Mback=np.empty((nz,N)); Mback[0]=M
    for i in range(1,nz):
        z1,z2=zgrid[i-1],zgrid[i]; dt=(tH_gyr(z1)-tH_gyr(z2))*1e9
        Mb=Mb-dMdt(Mb,0.5*(z1+z2))*dt; Mb=np.maximum(Mb,1e8); Mback[i]=Mb
    # forward in time: index from high z to low z
    order=np.arange(nz)[::-1]; scat=rng.normal(0,0.2,N)
    Mbh=np.zeros(N); seeded=np.zeros(N,bool); Msprev=None; out={}
    for j,i in enumerate(order):
        z=zgrid[i]; Mh=Mback[i]; Ms=moster(Mh,z)*10**scat
        if Msprev is None: Msprev=Ms; continue
        dt=(tH_gyr(zgrid[order[j-1]])-tH_gyr(z))*1e9
        dMs=np.maximum(Ms-Msprev,0)
        new=(~seeded)&(Ms>10**9.3); Mbh[new]=1e5; seeded|=new
        g=seeded
        Mbh[g]=Mbh[g]+A*Mbh[g]**p*dMs[g]/1e10          # A scaled so p=1 gives dlnM_BH = A dM*/1e10
        if galaxy_ceiling: ceil=Kg*Ms**betag
        else: ceil=K*Mh**beta*E(z)**k
        Mbh[g]=np.minimum(Mbh[g],ceil[g])
        Msprev=Ms
        if any(abs(z-zz)<zgrid[1]/2 for zz in (0,0.5,1,2,3)):
            w=g&(Ms>10**9.5)&(Ms<1e11)&(Mh>10**10.5)
            if w.sum()>50:
                y=np.log10(Mbh[w]); X=np.column_stack([np.ones(w.sum()),np.log10(Ms[w]),np.log10(Mh[w])])
                c=np.linalg.lstsq(X,y,rcond=None)[0]; T=np.polyfit(np.log10(Mh[w]),y,1)[0]
                frac=np.mean(Mbh[w]>0.5*ceil[w])
                out[round(z,2)]=(c[1],c[2],T,frac,w.sum())
    return out
if __name__=='__main__':
    # Ceiling normalised to pass through M_BH = 1e8 at M_h = 10^12.5 today.
    K=1e8/(10**12.5)**(5/3)
    fmt=lambda out: " | ".join(f"z={z:.1f}: a={v[0]:+.2f} b={v[1]:+.2f} T={v[2]:.2f} f={v[3]:.2f}"
                               for z,v in sorted(out.items()))
    print("Sect. 8.2: halo ceiling M_ceil = K M_h^(5/3) E(z)^(2/3), growth dM_BH = A M_BH^(1/2) dM*")
    print("  (an unregulated black hole then grows as M*^2, which is what both codes show at z=3)\n")
    print(f"  A=4e3  (the paper's fiducial, EAGLE-like)\n    {fmt(run(K=K,A=4e3,p=0.5))}")
    print(f"  A=6e3  (faster growth, earlier arrival)\n    {fmt(run(K=K,A=6e3,p=0.5))}")
    print(f"  A=3e3  (slower growth, later arrival)\n    {fmt(run(K=K,A=3e3,p=0.5))}")
    print("\n  galaxy ceiling instead of a halo one (M_ceil proportional to M*): b stays near zero, as in TNG50")
    print(f"    {fmt(run(galaxy_ceiling=True,Kg=1e-3,betag=1.0,A=4e3,p=0.5))}")
