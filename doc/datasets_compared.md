# Dataset Differences: NSA, TNG50, ALFALFA×NSA

A short comparison of the three datasets, with emphasis on simulation vs observations.

---

## Summary Table

| | NSA | TNG50 | ALFALFA×NSA |
|---|-----|-------|-------------|
| **Type** | Observation | Simulation | Observation |
| **Sample size** | 484,539 | 10,992 | 20,569 |
| **Variables** | 10 | 14 | 13 |
| **Gas info** | No | Yes | Yes (HI) |
| **Redshift** | Yes | No (z=0 snapshot) | Yes |
| **Selection** | Volume-limited optical | Resolution limit | HI-selected |

---

## Simulation vs Observations

### TNG50 (simulation)

- **Intrinsic properties:** DM mass, gas mass, BH mass, stellar/gas metallicity, SFR, etc. are direct outputs.
- **No observational effects:** No Malmquist bias, no flux limits, no dust, no measurement noise.
- **No Sérsic index:** Morphology is not fitted; no light-profile fitting.
- **No redshift:** Single z=0 snapshot; no distance/selection effects.
- **Smallest sample:** ~11k subhalos (resolution limit).
- **Ground truth:** Known physics; used to validate the causal discovery method.

### NSA (observation)

- **Derived quantities:** Stellar mass, metallicity, mass-to-light from SED fitting and templates.
- **Observational effects:** Malmquist bias, flux limits, dust attenuation, measurement errors.
- **Sérsic index:** From light-profile fitting.
- **Redshift:** Included; drives selection and apparent properties.
- **Largest sample:** ~485k galaxies (volume-limited optical).
- **No gas:** Optical-only; no HI or gas mass.

### ALFALFA×NSA (observation)

- **HI-selected:** Cross-match of ALFALFA (21-cm) with NSA (optical).
- **Gas-rich bias:** Favours gas-rich, star-forming, late-type galaxies.
- **Gas variables:** logMH (HI mass), W50 (line width).
- **Same optical systematics as NSA:** Derived masses, Sérsic, Malmquist, etc.
- **Selection:** HI detectability; higher-mass HI sources visible at larger distances.

---

## Variable Overlap and Gaps

| Property | NSA | TNG50 | ALFALFA×NSA |
|---------|-----|-------|-------------|
| Stellar mass | ✓ | ✓ | ✓ |
| Gas mass | ✗ | ✓ | ✓ (HI only) |
| BH mass | ✗ | ✓ | ✗ |
| DM mass | ✗ | ✓ | ✗ |
| Metallicity | ✓ (stellar) | ✓ (stellar + gas) | ✓ (stellar) |
| Sérsic index | ✓ | ✗ | ✓ |
| Redshift | ✓ | ✗ | ✓ |
| Velocity width | ✗ | ✓ (VMAX, VEL_DISP) | ✓ (W50) |
| Size | Half-light radius | Half-mass radius | Half-light radius |

---

## Causal Graph Differences

- **NSA:** Denser graph (19 edges); many bidirected edges; latent confounders (e.g. gas) not observed.
- **TNG50:** Dense graph (35 edges); mostly directed; recovers known physics; no selection effects.
- **ALFALFA×NSA:** Sparse graph (13 edges); highly oriented; logMH as causal root; gas resolves some confounders seen in NSA.
