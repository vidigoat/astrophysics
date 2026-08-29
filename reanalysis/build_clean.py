"""
Corrected, censoring-aware dataset construction for the three simulations.

Three defects in the original pipeline are fixed here:

  (1) TNG50 column mapping. Gadget/IllustrisTNG SubhaloMassType indices are
      0=gas, 1=DM, 2=(unused), 3=tracers, 4=stars, 5=BH.  The original prep read
      0->DM, 1->stellar, 4->gas, which permutes gas/DM/stars.  Proof: as labelled
      TNG50 had median log(M*/M_DM) = +1.24, i.e. stars outweighing dark matter,
      while EAGLE and SIMBA give -2.09 and -1.99.  Relabelling gives -2.18.

  (2) Censored variables entering a continuous-score causal search.
      M_BH sits at the -10 floor for 66.0% of TNG50 and 56.5% of SIMBA subhaloes
      (0.0% for EAGLE); SFR is exactly zero for 3.6/6.1/30.8%.  A variable that is
      a two-thirds point mass at a constant is not continuous, so any edge into it
      is at least partly recovering the SEEDING/quenching indicator rather than a
      physical dependence.  We therefore analyse the seeded, star-forming
      subsample, and separately keep the censoring indicators to test the artefact.

  (3) Algebraic identities in the variable set.
      BARYONIC_MASS = M* + M_gas is a deterministic function of two other search
      variables, which violates causal faithfulness by construction and guarantees
      spurious edges.  COLOUR = U - R likewise.  Both are dropped.

Bands: TNG50 SubhaloStellarPhotometrics is ordered U,B,V,K,g,r,i,z, so the original
index 2 is Buser V, not SDSS r.  TNG50 photometry is therefore NOT comparable with
EAGLE/SIMBA SDSS magnitudes and is excluded from cross-code work.
"""
import os, pickle, numpy as np

DATA = '/Users/vidigoat/astrophysics/Data/'
OUT  = '/Users/vidigoat/astrophysics/reanalysis/data/'
os.makedirs(OUT, exist_ok=True)

FLOOR = -9.99
COMMON = ['STELLAR_MASS','GAS_MASS','DM_MASS','BH_MASS',
          'HALFMASS_RAD','VEL_DISP','STAR_METALLICITY','GAS_METALLICITY','SSFR']


def load(name):
    with open(DATA + name + '.pkl','rb') as f:
        return {k: np.asarray(v, dtype=float) for k, v in pickle.load(f).items()}


def fix_tng50(d):
    """Undo the index permutation.  Stored label -> true quantity:
         DM_MASS      (col 0) -> gas
         STELLAR_MASS (col 1) -> dark matter
         GAS_MASS     (col 4) -> stars
       BH_MASS (col 5) was already correct.
    """
    out = dict(d)
    out['GAS_MASS']     = d['DM_MASS']
    out['DM_MASS']      = d['STELLAR_MASS']
    out['STELLAR_MASS'] = d['GAS_MASS']
    # PHOTOMETRIC_R was Buser V, not SDSS r.  Rename honestly.
    out['PHOTOMETRIC_V'] = d['PHOTOMETRIC_R']
    out.pop('PHOTOMETRIC_R', None)
    out.pop('COLOUR', None)          # was U-V, not u-r; and definitional anyway
    out.pop('BARYONIC_MASS', None)   # was M_DM + M*, i.e. corrupt AND definitional
    return out


def prepare(name, tag):
    d = load(name)
    if tag == 'TNG50':
        d = fix_tng50(d)
    d.pop('BARYONIC_MASS', None)
    d.pop('COLOUR', None)

    n0 = len(d['STELLAR_MASS'])

    # specific star formation rate; SFR is linear Msun/yr, masses are log10
    sfr = d['SFR']
    with np.errstate(divide='ignore', invalid='ignore'):
        ssfr = np.log10(sfr) - d['STELLAR_MASS']
    d['SSFR'] = ssfr

    seeded      = d['BH_MASS'] > FLOOR
    starforming = sfr > 0
    clean       = seeded & starforming & np.isfinite(ssfr)

    rec = {}
    for k in COMMON:
        v = d[k].copy()
        if k == 'HALFMASS_RAD':
            v = np.log10(np.maximum(v, 1e-10))   # log size, so slopes are unit-free
        if k in ('STAR_METALLICITY','GAS_METALLICITY'):
            v = np.log10(np.maximum(v, 1e-8))    # log metallicity
        rec[k] = v

    full = {k: v.copy() for k, v in rec.items()}
    full['BH_SEEDED']   = seeded.astype(float)
    full['STARFORMING'] = starforming.astype(float)

    cl = {k: v[clean] for k, v in rec.items()}

    print(f'{tag:7s} N_all={n0:6d}  seeded={seeded.sum():6d}  SF={starforming.sum():6d}  clean={clean.sum():6d}')
    for k in COMMON:
        v = cl[k]
        print(f'          {k:18s} med={np.median(v):9.3f}  sd={np.std(v):7.3f}')

    with open(OUT + f'{tag}_clean.pkl','wb') as f:
        pickle.dump(cl, f)
    with open(OUT + f'{tag}_full.pkl','wb') as f:
        pickle.dump(full, f)
    return cl


if __name__ == '__main__':
    print('Common variable set (no algebraic identities):', COMMON)
    print()
    for name, tag in [('tng50_final','TNG50'), ('eagle_final','EAGLE'), ('simba_final','SIMBA')]:
        prepare(name, tag)
        print()
