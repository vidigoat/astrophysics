# Causal Discovery in Galaxy Properties

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive causal discovery analysis comparing gas-rich HI-selected galaxies (ALFALFA × NSA) with the full optical galaxy population (NSA-only), revealing fundamental differences in how physical properties drive galaxy evolution.

## Overview

This repository implements a complete workflow for **causal discovery** in galaxy properties using the **Fast Causal Inference Test (FCIT)** algorithm. By analyzing two complementary galaxy samples, we identify which physical relationships depend on gas content and which hold universally across all galaxy types.

### Key Discovery

**Universal Mass-to-Light/Color Coupling**: A fundamental stellar population relationship (`ELPETRO_MTOL o-o COLOR_U_R`) with 100% bootstrap recovery in both samples, revealing that stellar population age and color are inseparably coupled across all galaxy types.

### Research Highlights

- **Gas-Regulated Regime**: HI-selected galaxies show gas reservoir as the primary organizer of evolution
- **Structure-Driven Regime**: Full optical population reveals structural parameters as causal drivers
- **Morphology Transition**: Morphology is reactive in gas-rich galaxies but causal in the full population
- **Robust Validation**: All relationships validated through comprehensive bootstrap and null model tests

## Results Summary

### ALFALFA × NSA (Gas-Rich Subset)
- **20,569 galaxies** | **13 properties** | **7 edges discovered**
- **6 edges ≥80% bootstrap recovery** (3 at perfect 100%)
- Gas reservoir triad: HI mass, baryonic mass, and stellar mass form inseparable triplet

### NSA-only (Full Optical Population)
- **484,539 galaxies** | **10 properties** | **18 edges discovered**
- **All 18 edges at 100% bootstrap recovery** (perfect stability)
- Structure-driven: Morphology and metallicity act as causal drivers

## Installation

### Prerequisites

- **Python 3.8+**
- **Graphviz** (for visualization)
  - Windows: [Download installer](https://graphviz.org/download/)
  - Linux: `sudo apt-get install graphviz`
  - macOS: `brew install graphviz`

### Dependencies

```bash
pip install numpy pandas astropy matplotlib seaborn scipy pytetrad graphviz
```

### Data Requirements

You need the following FITS files:

1. **ALFALFA × NSA**: `5asfullmatch.fits`
   - Cross-match between ALFALFA HI survey and NSA optical catalogue
   - Available from [ALFALFA survey data releases](http://egg.astro.cornell.edu/alfalfa/data/)

2. **NSA-only**: `nsa_v1_0_1.fits`
   - NASA-Sloan Atlas v1.0.1 optical catalogue
   - Available from [NSA website](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/)

Set environment variables for data paths:
```bash
# Windows
set ALFALFA_FITS_PATH=C:\path\to\5asfullmatch.fits
set NSA_FITS_PATH=C:\path\to\nsa_v1_0_1.fits
set GRAPHVIZ_BIN=C:\path\to\graphviz\bin

# Linux/macOS
export ALFALFA_FITS_PATH=/path/to/5asfullmatch.fits
export NSA_FITS_PATH=/path/to/nsa_v1_0_1.fits
```

## Usage

### 1. Data Preparation

```bash
python Code/DataPrep/ALFALFA_X_NSA_DATAPREP.py
python Code/DataPrep/NSA_Data_Preparation.py
```

### 2. Causal Discovery (FCIT)

```bash
python Code/FCIT/ALFALFA_NSA_FCIT.py
python Code/FCIT/NSA_FCIT.py
```

### 3. Validation

**Bootstrap validation** (tests robustness across subsamples):
```bash
python Code/Validation/ALFALFA_NSA_Bootstrap.py
python Code/Validation/NSA_Bootstrap.py
```

**Null model validation** (confirms edges are genuine):
```bash
python Code/Validation/ALFALFA_NSA_NullModel.py
python Code/Validation/NSA_NullModel.py
```

### 4. Visualizations

```bash
python Code/Visualizations/Alfalfa_Causal_Scatter.py
python Code/Visualizations/NSA_Scatter_plot.py
python Code/Visualizations/NSA_Scatter_plot_Part2.py
python Code/Visualizations/Bootstrap_ALFALFA_NSA_plot.py
python Code/Visualizations/NSA_Validation_Figure.py
python Code/Visualizations/NullModel_Visualization.py
python Code/Visualizations/Property_Distributions.py
```

## Methodology

### FCIT Algorithm

The **Fast Causal Inference Test (FCIT)** is a constraint-based causal discovery algorithm that:
- Tests conditional independence between variable pairs using likelihood ratio tests
- Uses basis function expansions (cubic splines) to capture non-linear relationships
- Applies BIC (Bayesian Information Criterion) scoring to select optimal graph structure
- Outputs a Partial Ancestral Graph (PAG) showing causal relationships with three edge types:
  - **Directed edges** (`→`): Clear causal direction identified
  - **Partially directed edges** (`o→`): Causal direction partially identified
  - **Undirected edges** (`o-o`): Causal direction cannot be determined

### FCIT Parameters

- **Truncation Limit**: 14 (controls maximum order of basis functions)
- **Penalty Discount**: 50 (scales BIC penalty term for graph sparsity)
- **P-value Threshold**: 0.01 (significance level for independence tests)

### Validation

- **Bootstrap resampling**: 50 runs (ALFALFA × NSA), 10 runs (NSA-only), 80% subsamples
- **Null model tests**: Column-wise shuffling confirms all edges are genuine (0 false positives)
- **Recovery threshold**: ≥80% bootstrap recovery for reporting

## Project Structure

```
astrophysics/
├── Code/
│   ├── DataPrep/              # Data preparation scripts
│   │   ├── ALFALFA_X_NSA_DATAPREP.py
│   │   └── NSA_Data_Preparation.py
│   ├── FCIT/                   # Causal discovery scripts
│   │   ├── ALFALFA_NSA_FCIT.py
│   │   └── NSA_FCIT.py
│   ├── Validation/             # Bootstrap and null model validation
│   │   ├── ALFALFA_NSA_Bootstrap.py
│   │   ├── ALFALFA_NSA_NullModel.py
│   │   ├── NSA_Bootstrap.py
│   │   └── NSA_NullModel.py
│   └── Visualizations/         # Figure generation scripts
│       ├── Alfalfa_Causal_Scatter.py
│       ├── NSA_Scatter_plot.py
│       ├── NSA_Scatter_plot_Part2.py
│       ├── Bootstrap_ALFALFA_NSA_plot.py
│       ├── NSA_Validation_Figure.py
│       ├── NullModel_Visualization.py
│       └── Property_Distributions.py
└── README.md
```

## Key Scientific Findings

### 1. Gas Reservoir Triad (ALFALFA × NSA)

The three edges connecting HI mass, baryonic mass, and stellar mass form an inseparable, undirected triplet with 90-100% bootstrap recovery. In HI-selected galaxies, gas availability is the primary constraint on stellar assembly.

### 2. Universal Mass-to-Light/Color Coupling

`ELPETRO_MTOL o-o COLOR_U_R` with 100% bootstrap recovery in both samples. This fundamental stellar evolution relationship transcends gas content.

### 3. Morphology Transition

- **Gas-rich subset**: Morphology is reactive (structural properties disconnected from gas reservoir)
- **Full population**: Morphology is causal (structural parameters drive stellar mass)

### 4. Perfect Bootstrap Stability (NSA-only)

All 18 edges achieve 100% bootstrap recovery, demonstrating exceptional robustness and indicating the causal structure is highly representative of the underlying population.

## Citation

If you use this code or results in your research, please cite:

```bibtex
@software{astrophysics_causal_discovery,
  title = {Causal Discovery in Galaxy Properties: ALFALFA × NSA vs. NSA-only},
  author = {Vidigoat},
  year = {2024},
  url = {https://github.com/vidigoat/astrophysics}
}
```

## References

- **FCIT Algorithm**: Desmond, H., & Ramsey, J. (2025). "The Causal Structure of Galactic Astrophysics." Version November 11, 2025. Institute of Cosmology & Gravitation, University of Portsmouth, Dennis Sciama Building, Portsmouth, PO1 3FX, UK; Carnegie Mellon University Philosophy Department.

- **ALFALFA × NSA Matched Catalogue**: Stiskalek, R., Desmond, H., Holvey, T., & Jones, M. G. "The dependence of subhalo abundance matching on galaxy photometry and selection criteria." Department of Physics, University of Oxford, Denys Wilkinson Building, Keble Road, Oxford OX1 3RH, UK; Fakultät für Physik, Ludwig-Maximilians-Universität München, 80333 München, Germany; School of Physics and Astronomy, University of Glasgow, Glasgow G12 8QQ, UK; Instituto de Astrofísica de Andalucía (IAA-CSIC), Glorieta de la Astronomía s/n, 18008, Granada, Spain; Steward Observatory, University of Arizona, 933 North Cherry Avenue, Rm. N204, Tucson, AZ 85721-0065, USA.

- **ALFALFA Survey**: Haynes, M. P., Giovanelli, R., Kent, B. R., Adams, E. A. K., Balonek, T. J., Craig, D. W., Fertig, D., Finn, R., Giovanardi, C., Hallenbeck, G., Hess, K. M., Hoffman, G. L., et al. "The Arecibo Legacy Fast ALFA Survey: The ALFALFA Extragalactic HI Source Catalog."

- **NSA Catalogue**: Blanton, M. R., Kazin, E., Muna, D., Weaver, B. A., & Price-Whelan, A. (2011). "Improved Background Subtraction for the Sloan Digital Sky Survey Images." *The Astronomical Journal*, 142, 31. Center for Cosmology and Particle Physics, Department of Physics, New York University, 4 Washington Place, New York, NY 10003, USA.

## License

This project is licensed under the MIT License.

## Contact

For questions, collaboration inquiries, or data access requests, please contact:

**Email**: vidit.patankar16@gmail.com

**GitHub**: [@vidigoat](https://github.com/vidigoat)
