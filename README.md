# Causal Structure in Galaxies

Analysis code for causal-discovery experiments on galaxy catalogues and cosmological
simulations, using the FCIT algorithm (Fast Causal Inference with Targeted Testing)
via the `py-tetrad` interface to Tetrad.

Five data sets are processed: two observational catalogues (the NASA--Sloan Atlas and
the gas-selected ALFALFA x NSA matched sample) and three cosmological simulations
(TNG50, EAGLE and SIMBA). Each is reduced to a common set of first-order galaxy
properties, run through FCIT to recover a partial ancestral graph, and compared
across codes on a matched variable set.

## Layout

| Path | Contents |
|---|---|
| `Code/DataPrep/` | Catalogue ingestion and cleaning, one script per data set |
| `Code/FCIT/` | FCIT runs on the per-catalogue variable sets |
| `Code/Analysis/` | Hyperparameter tuning, null tests, robustness checks |
| `Code/Visualizations/` | Corner plots and PAG rendering |
| `reanalysis/` | Current analysis: corrected data extraction, conditional-independence tests, cross-code comparison, figures |
| `reanalysis/results/` | Recovered graphs, FCIT logs and figure outputs |
| `Results/` | Consensus edge lists from the earlier per-catalogue runs |

## Notes on the data

Raw survey and simulation pulls are not tracked (see `.gitignore`); each `DataPrep`
script fetches or reads from its public source. The ALFALFA x NSA cross-match is not
redistributed here and can be rebuilt following Stiskalek et al. (2021).

Two corrections in `reanalysis/` supersede the earlier `Code/` outputs. The
IllustrisTNG `SubhaloMassType` particle-type indices are `0 = gas`, `1 = dark matter`,
`4 = stars`; and `SubhaloStellarPhotometrics` is ordered `U, B, V, K, g, r, i, z`, so
index 2 is Buser `V` rather than SDSS `r`. Black-hole mass is censored in TNG50 and
SIMBA (unseeded objects sit at the catalogue floor) and is restricted to seeded
galaxies throughout.

## Requirements

Python 3.11+, with `py-tetrad` (and a JVM), `numpy`, `scipy`, `pandas`, `matplotlib`
and `networkx`.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Contact

Vidit Patankar — vidit.patankar16@gmail.com
