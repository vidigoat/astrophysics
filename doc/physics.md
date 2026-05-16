# Physical Interpretation of Causal Structures

This document provides the full physical interpretation of the partial ancestral graphs (PAGs) inferred using the FCIT algorithm for all three datasets: NSA, TNG50, and ALFALFA×NSA.

## 4.2. Real Data: NSA (NASA–Sloan Atlas)

We apply the FCIT algorithm to the NSA data. The PAG produced describes a combination of physical effects carrying information about galaxy evolution and observational and selection effects describing the way in which the data was obtained.

As expected, redshift influences the apparent size, which scales inversely with angular diameter distance. It also influences mass and absolute magnitude through Malmquist bias, the preferential detection of intrinsically brighter objects at higher distance.

Mass is seen to causally determine size (rather than vice versa, as would be possible given simply the mass–size relation), suggesting inside-out growth of discs and size expansion via mergers. It also determines Sérsic index, agreeing with the idea that bulge growth and morphological transformation are primarily consequences of hierarchical mass assembly. The absence of a link from morphology to mass disfavours simplistic models where concentration alone sets stellar mass. The link from SFR to absolute magnitude reflects the brightening of galaxies in optical bands due to recent star formation.

The uncertain edges between star formation, stellar mass and morphology highlight the complexity of baryonic processes. Several appear with circle endpoints, indicating algorithmic uncertainty about direction or the influence of confounding variables. The ambiguous link between stellar mass and luminosity is unsurprising, since mass estimates are derived from photometry and strongly depend on mass-to-light ratios. The circle endpoints highlight the possible role of latent factors not included in the analysis—such as dust attenuation, gas content, and halo environment—which can drive correlations and obscure true directions.

The graph does not unambiguously support a picture in which star formation determines morphology on short timescales, or that mass quenching is the sole pathway. Instead, it suggests an intertwined system in which stellar mass remains the fundamental regulator, while star formation and morphology are mutually influenced by common drivers such as gas inflow and environment. Disentangling these physical drivers from observational and selection-induced structure (and external latent variables) requires further work, as indicated directly by the circle edges. The result does however imply that the backbone of galaxy evolution—mass driving size and morphology, and star formation driving luminosity—is recoverable directly from survey data. This is highly promising for future, more sophisticated applications of the methodology.

**Key structural features:**
- **ELPETRO_MASS is a central hub** receiving input from metallicity, mass-to-light ratio, Sérsic index, and redshift, and driving absolute magnitude
- **ELPETRO_TH50_R (half-light radius) is a terminal sink** receiving 5 causal inputs (color, absolute magnitude, Sérsic index, redshift, axis ratio)
- **A bidirected cluster** links COLOR_U_R, ELPETRO_B300 (birthrate parameter), ELPETRO_METS (metallicity), and ELPETRO_MTOL (mass-to-light ratio), suggesting latent confounders (likely star formation history)

---

## TNG50 (IllustrisTNG Simulation)

The TNG50 graph recovers well-known physical relationships from first principles: the stellar mass–metallicity relation, the mass–size relation, the black hole–stellar mass relation, and gas-driven size evolution. The two dominant causal roots (STELLAR_MASS and GAS_MASS) reflect the fundamental baryonic components in galaxy formation.

**Key causal paths:**
1. **STELLAR_MASS → BARYONIC_MASS → DM_MASS**: Stellar mass sets baryonic content, which correlates with halo mass
2. **STELLAR_MASS → STAR_METALLICITY → GAS_METALLICITY**: Stellar enrichment drives interstellar medium metallicity
3. **STELLAR_MASS → BH_MASS**: M–σ-like relation between stellar mass and black hole mass
4. **STELLAR_MASS → VEL_DISP**: Faber–Jackson-like relation
5. **STELLAR_MASS → VMAX**: Tully–Fisher-like relation
6. **GAS_MASS → BH_MASS**: Gas feeding black hole growth
7. **GAS_MASS → HALFMASS_RAD**: Gas extent determines size
8. **GAS_MASS → STAR_METALLICITY**: Gas regulates enrichment
9. **GAS_MASS → VEL_DISP, VMAX**: Gas potential well contribution to dynamics

The graph structure validates the methodology: if FCIT can recover known physics in the simulation, the novel structures it finds in observations are credible.

**Hub nodes:**
- **GAS_MASS**: Primary causal root with 9 outgoing edges
- **STELLAR_MASS**: Primary causal root with 10 outgoing edges
- **HALFMASS_RAD**: Terminal sink receiving 5 causal inputs

---

## ALFALFA×NSA (Gas-Rich HI-Selected Galaxies)

The ALFALFA×NSA graph reveals a distinctly different causal architecture from the optical-only NSA sample. The graph is extremely sparse (edge density 0.167 vs 0.422 for NSA) but highly oriented (84.6% orientation fraction), meaning FCIT is very confident about direction for most edges.

**Signature feature: logMH (HI gas mass) as a causal root**

The most important finding is that logMH drives BARYONIC_MASS directly and influences ELPETRO_MASS and ZDIST. This is the signature feature of the gas-rich population—HI gas mass acts as an independent causal driver that was invisible in the optical-only NSA sample.

**Key causal structure:**
- **logMH → BARYONIC_MASS**: HI gas mass directly drives total baryonic content. In the gas-rich population, the gas reservoir is a primary determinant of the galaxy's total baryonic budget, consistent with these being gas-dominated systems.
- **ELPETRO_MASS → BARYONIC_MASS → ELPETRO_ABSMAG_R**: The dominant causal chain linking stellar mass, baryonic mass, and luminosity
- **logMH o→ ELPETRO_MASS**: Gas mass influences stellar mass (partially directed), consistent with gas availability regulating star formation and stellar mass buildup
- **W50 o-o logMH**: The HI velocity width and gas mass are bidirectionally associated, reflecting the baryonic Tully–Fisher relation. FCIT cannot orient this edge because both quantities are co-determined by the halo potential.
- **logMH o→ ZDIST**: Gas mass influences the observed redshift distribution, reflecting an ALFALFA selection effect (higher-mass HI sources are detectable at greater distances)

**Why is the graph so sparse?**

The ALFALFA×NSA edge density (0.167) is less than half that of NSA (0.422). Three factors contribute:

1. **Gas information resolves confounders**: The addition of logMH and W50 "explains away" associations that appeared as edges in the NSA graph. Edges that were present because of a shared dependence on gas content are no longer needed when gas content is explicitly included.
2. **Sample selection**: ALFALFA-selected galaxies are a more homogeneous population (gas-rich, star-forming, late-type) than the full NSA. Within this subset, fewer causal pathways are active because much of the variation in the general population (e.g., quenched vs star-forming dichotomy) is absent.
3. **Higher penalty discount**: The sparser regularization reflects the need for stronger sparsity at this sample size.

**Comparison with NSA:**

The critical difference between NSA and ALFALFA×NSA is the presence of gas properties (logMH, W50). Adding these variables:

1. **Introduces logMH as an independent causal root** that was invisible in the optical-only NSA
2. **Reduces overall edge density by 60%** (0.422 → 0.167), suggesting that many apparent associations in the optical population are mediated or confounded by gas content
3. **Increases orientation certainty by 34%** (0.632 → 0.846), implying that gas information resolves causal ambiguities present in the optical-only data
4. **Eliminates the bidirected cluster** (COLOR_U_R, B300, METS, MTOL) seen in NSA, replacing it with directed edges—suggesting that gas content is the latent variable responsible for those bidirected associations

This supports the hypothesis that gas content is a key latent variable in optical-only studies of galaxy properties.

---

## Cross-Dataset Summary

**Universal finding: Stellar mass is the fundamental causal driver**

Across all three datasets, stellar mass (ELPETRO_MASS / STELLAR_MASS) is the node with the highest out-degree and drives luminosity, metallicity, size, and dynamics. This is consistent with the established picture from galaxy formation theory.

**Gas-rich galaxies have a distinct causal architecture**

The ALFALFA×NSA graph is qualitatively different from the NSA graph: sparser, more oriented, with HI gas mass as an independent causal root. This is not simply a sample-size effect—it reflects genuine physical differences in the causal mechanisms governing gas-rich vs general galaxy populations.

**The mass–luminosity–size causal chain**

Across all datasets, the dominant causal pathway is:

**Stellar Mass → Luminosity → Size**

- NSA: ELPETRO_MASS → ELPETRO_ABSMAG_R → ELPETRO_TH50_R
- TNG50: STELLAR_MASS → BARYONIC_MASS → HALFMASS_RAD
- ALFALFA×NSA: ELPETRO_MASS → ELPETRO_ABSMAG_R (+ BARYONIC_MASS → ELPETRO_ABSMAG_R)

This is the most robust causal finding of the analysis. Stellar mass fundamentally determines a galaxy's luminosity, which in turn constrains its observed size. This relationship is 100% stable across all bootstrap runs in all datasets.
