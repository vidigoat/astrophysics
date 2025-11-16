# Causal Discovery in Galaxy Properties: ALFALFA × NSA vs. NSA-only

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **A comprehensive causal discovery analysis comparing gas-rich HI-selected galaxies with the full optical galaxy population, revealing fundamental differences in how physical properties drive galaxy evolution.**

## 🌟 Overview

This repository presents a complete, reproducible workflow for **causal discovery** in galaxy properties using the **Fast Causal Inference Test (FCIT)** algorithm. By analyzing two complementary galaxy samples—HI-selected gas-rich galaxies and the full optical population—we identify which physical relationships depend on gas content and which hold universally across all galaxy types.

### 🎯 Key Discovery

**Universal Mass-to-Light/Color Coupling**: We discovered a fundamental stellar population relationship (`ELPETRO_MTOL o-o COLOR_U_R`) with **100% bootstrap recovery** in both samples, revealing that stellar population age and color are inseparably coupled across all galaxy types, regardless of gas content.

### 📊 Research Highlights

- **Gas-Regulated Regime**: HI-selected galaxies show gas reservoir as the primary organizer of evolution
- **Structure-Driven Regime**: Full optical population reveals structural parameters as causal drivers
- **Morphology Transition**: Morphology is reactive in gas-rich galaxies but causal in the full population
- **Robust Validation**: All relationships validated through comprehensive bootstrap and null model tests

---

## 📈 Results Summary

### ALFALFA × NSA (Gas-Rich Subset)
- **20,569 galaxies** | **13 properties** | **7 edges discovered**
- **6 edges ≥80% bootstrap recovery** (3 at perfect 100%)
- **Gas reservoir triad**: HI mass, baryonic mass, and stellar mass form inseparable triplet
- **Morphology isolated**: Structural properties disconnected from gas reservoir

### NSA-only (Full Optical Population)
- **484,539 galaxies** | **10 properties** | **18 edges discovered**
- **All 18 edges at 100% bootstrap recovery** (perfect stability)
- **Structure-driven**: Morphology and metallicity act as causal drivers
- **Clear hierarchy**: Directed relationships show causal flow

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Graphviz** (for visualization)
  - Windows: [Download installer](https://graphviz.org/download/)
  - Linux: `sudo apt-get install graphviz`
  - macOS: `brew install graphviz`

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/vidigoat/astrophysics.git
cd astrophysics
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up data paths** (optional):
```bash
# Windows
set ALFALFA_FITS_PATH=C:\path\to\5asfullmatch.fits
set NSA_FITS_PATH=C:\path\to\nsa_v1_0_1.fits
set GRAPHVIZ_BIN=C:\path\to\graphviz\bin

# Linux/macOS
export ALFALFA_FITS_PATH=/path/to/5asfullmatch.fits
export NSA_FITS_PATH=/path/to/nsa_v1_0_1.fits
```

### Data Requirements

You need to obtain the following FITS files:

1. **ALFALFA × NSA**: `5asfullmatch.fits`
   - Cross-match between ALFALFA HI survey and NSA optical catalogue
   - Available from [ALFALFA survey data releases](http://egg.astro.cornell.edu/alfalfa/data/)

2. **NSA-only**: `nsa_v1_0_1.fits`
   - NASA-Sloan Atlas v1.0.1 optical catalogue
   - Available from [NSA website](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/)

Place these files in a `Data/` directory or set the environment variables above.

---

## 📖 Usage

### 1. Data Preparation

Prepare the ALFALFA × NSA dataset:
```bash
python Code/DataPrep/ALFALFA\ X\ NSA.py
```

Prepare the NSA-only dataset:
```bash
python Code/DataPrep/NSA_Data_Preparation.py
```

### 2. Causal Discovery (FCIT)

Run FCIT on ALFALFA × NSA:
```bash
python Code/FCIT/FCIT.py
```

Run FCIT on NSA-only:
```bash
python Code/FCIT/NSA_FCIT.py
```

### 3. Validation

**Bootstrap validation** (tests robustness across subsamples):
```bash
python Code/Validation/Bootstrap.py        # ALFALFA × NSA (50 runs)
python Code/Validation/NSA_Bootstrap.py    # NSA-only (10 runs)
```

**Null model validation** (confirms edges are genuine):
```bash
python Code/Validation/NullModel.py        # ALFALFA × NSA
python Code/Validation/NSA_NullModel.py    # NSA-only
```

### 4. Visualizations

Generate validation figures:
```bash
python Code/Visualizations/Alfalfa_Validation_Figure.py
python Code/Visualizations/NSA_Validation_Figure.py
```

Generate scatter plots:
```bash
python Code/Visualizations/Alfalfa_Causal_Scatter.py
python Code/Visualizations/NSA_Causal_Scatter.py
python Code/Visualizations/NSA_Causal_Scatter_Part2.py
```

---

## 🔬 Methodology

### FCIT Algorithm

The **Fast Causal Inference Test (FCIT)** is a constraint-based causal discovery algorithm that:
- Tests conditional independence between variable pairs using likelihood ratio tests
- Uses basis function expansions (cubic splines) to capture non-linear relationships
- Applies BIC (Bayesian Information Criterion) scoring to select optimal graph structure
- Outputs a Partial Ancestral Graph (PAG) showing causal relationships with three edge types:
  - **Directed edges** (`→`): Clear causal direction identified
  - **Partially directed edges** (`o→`): Causal direction partially identified
  - **Undirected edges** (`o-o`): Causal direction cannot be determined (bidirectional or confounded)

### FCIT Parameters

We use consistent, carefully chosen parameters across all analyses to ensure reproducibility and comparability:

#### Truncation Limit: 14
- **Purpose**: Controls the maximum order of basis functions in the spline expansion
- **Effect**: Higher values capture more non-linear relationships but require more data
- **Rationale**: 14 is optimal for our sample sizes (~20k–500k galaxies), balancing sensitivity to non-linearities with computational efficiency
- **Range**: Typically 4–20; we tested values 10–18 and found 14 optimal

#### Penalty Discount: 50
- **Purpose**: Scales the BIC penalty term (default is 0.5×log(n)) to control graph sparsity
- **Effect**: Higher values → sparser graphs (fewer edges), lower values → denser graphs (more edges)
- **Rationale**: 50 provides balanced sparsity, preventing overfitting while capturing genuine relationships
- **Impact**: This parameter directly affects the number of edges discovered; we tested 30–70 and found 50 optimal

#### P-value Threshold (Alpha): 0.01
- **Purpose**: Significance level for conditional independence tests
- **Effect**: Lower values → stricter independence criteria → fewer edges, higher values → more lenient → more edges
- **Rationale**: 0.01 is standard for scientific applications, providing a good balance between Type I and Type II errors
- **Interpretation**: Edges are only included if the conditional independence test rejects the null hypothesis at p < 0.01

### Parameter Selection Rationale

These parameters were chosen through systematic testing:
1. **Initial exploration**: Tested truncation limits 10–18 and penalty discounts 30–70
2. **Bootstrap validation**: Selected parameters that produced stable, interpretable graphs
3. **Physical validation**: Ensured discovered relationships align with known astrophysical scaling relations
4. **Consistency**: Same parameters used across both datasets for fair comparison

**Key Insight**: The parameter combination (truncation=14, penalty=50, alpha=0.01) produces graphs that are:
- Physically interpretable (align with known scaling relations)
- Statistically robust (high bootstrap recovery)
- Computationally efficient (reasonable runtime)

### Validation

- **Bootstrap resampling**: 50 runs (ALFALFA × NSA), 10 runs (NSA-only), 80% subsamples
- **Null model tests**: Column-wise shuffling confirms all edges are genuine (0 false positives)
- **Recovery threshold**: ≥80% bootstrap recovery for reporting

---

## 📁 Project Structure

```
astrophysics/
├── Code/
│   ├── DataPrep/          # Data preparation scripts
│   │   ├── ALFALFA X NSA.py
│   │   └── NSA_Data_Preparation.py
│   ├── FCIT/              # Causal discovery scripts
│   │   ├── FCIT.py
│   │   ├── NSA_FCIT.py
│   │   ├── FCIT_NoCuts_ALFALFA.py
│   │   └── FCIT_NoCuts_NSA.py
│   ├── Validation/        # Bootstrap and null model validation
│   │   ├── Bootstrap.py
│   │   ├── NSA_Bootstrap.py
│   │   ├── NullModel.py
│   │   └── NSA_NullModel.py
│   └── Visualizations/    # Figure generation scripts
│       ├── Alfalfa_Validation_Figure.py
│       ├── NSA_Validation_Figure.py
│       ├── Alfalfa_Causal_Scatter.py
│       ├── NSA_Causal_Scatter.py
│       ├── NSA_Causal_Scatter_Part2.py
│       ├── Create_Property_Tables.py
│       └── NullModel_Visualization.py
├── Plots/                 # Generated visualizations
│   ├── CausalStructure/   # Causal graph visualizations
│   ├── BootstrapPlots/    # Bootstrap validation figures
│   ├── ScatterPlots/      # Edge scatter plots
│   ├── Tables/            # Property tables
│   └── NullPlots/         # Null model visualizations
├── Results/               # FCIT outputs and validation results
│   ├── *.txt              # FCIT text outputs
│   ├── *.csv              # Bootstrap validation results
│   └── *.pkl              # Null model results
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── .gitignore            # Git ignore rules
```

---

## 🎓 Key Scientific Findings

### 1. Gas Reservoir Triad (ALFALFA × NSA)

The three edges connecting HI mass, baryonic mass, and stellar mass form an inseparable, undirected triplet with 90-100% bootstrap recovery:

- `BARYONIC_MASS o-o ELPETRO_MASS` (100% recovery)
- `logMH o-o BARYONIC_MASS` (92% recovery)
- `logMH o-o ELPETRO_MASS` (90% recovery)

**Interpretation**: In HI-selected galaxies, gas availability is the primary constraint on stellar assembly. The undirected nature suggests bidirectional coupling: gas enables star formation, while stellar mass affects gas retention.

### 2. Universal Mass-to-Light/Color Coupling

**Discovery**: `ELPETRO_MTOL o-o COLOR_U_R` with 100% bootstrap recovery in both samples.

**Physical Interpretation**: 
- Older stellar populations (high M/L, ~3-10 M☉/L☉) are redder
- Younger stellar populations (low M/L, ~0.1-1 M☉/L☉) are bluer
- This is a fundamental stellar evolution relationship that transcends gas content

### 3. Morphology Transition

- **Gas-rich subset**: Morphology is **reactive** (structural properties disconnected from gas reservoir)
- **Full population**: Morphology is **causal** (structural parameters drive stellar mass)

**Implication**: Morphology's role depends on gas content—it becomes causal when gas is depleted or when considering the full population.

### 4. Perfect Bootstrap Stability (NSA-only)

All 18 edges achieve 100% bootstrap recovery, demonstrating exceptional robustness. This perfect stability is unprecedented and indicates the causal structure is highly representative of the underlying population.

---

## 📊 Output Files

### Causal Graphs
- `Plots/CausalStructure/fcit_final_13props_t14_p50.png` - ALFALFA × NSA PAG
- `Plots/CausalStructure/nsa_fcit_t14_p50.png` - NSA-only PAG

### Validation Figures
- `Plots/BootstrapPlots/alfalfa_bootstrap_validation.png` - ALFALFA × NSA bootstrap
- `Plots/BootstrapPlots/nsa_bootstrap_validation.png` - NSA-only bootstrap
- `Plots/NullPlots/null_model_visualization.png` - Combined null model results

### Scatter Plots
- `Plots/ScatterPlots/alfalfa_causal_scatter.png` - ALFALFA × NSA edge visualizations
- `Plots/ScatterPlots/nsa_causal_scatter.png` - NSA-only edge visualizations (Part 1)
- `Plots/ScatterPlots/nsa_causal_scatter_part2.png` - NSA-only edge visualizations (Part 2)

### Property Tables
- `Plots/Tables/alfalfa_properties_table.png` - ALFALFA × NSA properties
- `Plots/Tables/nsa_properties_table.png` - NSA-only properties

### Results
- `Results/fcit_final_13props_t14_p50.txt` - ALFALFA × NSA FCIT output
- `Results/nsa_fcit_t14_p50.txt` - NSA-only FCIT output
- `Results/bootstrap_validation.csv` - ALFALFA × NSA bootstrap results
- `Results/nsa_bootstrap_validation.csv` - NSA-only bootstrap results

---

## 📚 Datasets

### ALFALFA × NSA Sample
- **Source**: Cross-match between ALFALFA HI survey (Haynes et al. 2018) and NSA optical catalogue (Blanton et al. 2011)
- **Size**: 20,569 galaxies (after quality cuts, 91.5% retention)
- **Properties**: 13 total (including HI mass, baryonic mass, W50 velocity width)
- **Key Feature**: HI-selected sample represents gas-rich subset, probing gas-regulated evolution

### NSA-only Sample
- **Source**: NSA v1.0.1 optical catalogue (Blanton et al. 2011)
- **Size**: 484,539 galaxies (after quality cuts, 75.5% retention)
- **Properties**: 10 total (no HI information)
- **Key Feature**: Full optical population including both gas-rich and gas-poor systems

---

## 🔧 Technical Details

### Dependencies

- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `astropy>=4.0.0` - FITS file handling
- `matplotlib>=3.3.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations
- `scipy>=1.7.0` - Scientific computing
- `pytetrad>=1.0.0` - Causal discovery (FCIT)
- `graphviz>=0.20.0` - Graph visualization

### Environment Variables

Optional environment variables for custom paths:
- `ALFALFA_FITS_PATH` - Path to ALFALFA × NSA FITS file
- `NSA_FITS_PATH` - Path to NSA-only FITS file
- `GRAPHVIZ_BIN` - Path to Graphviz bin directory

---

## 📝 Citation

If you use this code or results in your research, please cite:

```bibtex
@software{astrophysics_causal_discovery,
  title = {Causal Discovery in Galaxy Properties: ALFALFA × NSA vs. NSA-only},
  author = {Vidigoat},
  year = {2024},
  url = {https://github.com/vidigoat/astrophysics},
  note = {Private repository}
}
```

---

## 📖 References

- **FCIT Algorithm**: Ramsey, J. D., et al. (2017). "A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models." *International Journal of Data Science and Analytics*, 3, 121-129.

- **ALFALFA Survey**: Haynes, M. P., et al. (2018). "The Arecibo Legacy Fast ALFA Survey: The ALFALFA Extragalactic H I Source Catalog." *The Astrophysical Journal*, 861, 49.

- **NSA Catalogue**: Blanton, M. R., et al. (2011). "Improved Background Subtraction for the Sloan Digital Sky Survey Images." *The Astronomical Journal*, 142, 31.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Repository Status

**Note**: This is currently a **private repository**. The code and results are not yet publicly available. This repository will be made public upon publication of the associated research paper.

---

## 📋 Understanding Repository Files

### File Status Indicators

When viewing this repository in GitHub or a Git client, you may see colored indicators:
- **Green dots** (●): Files that are tracked by Git and have been committed
- **Modified files**: Files that have been changed since the last commit
- **Untracked files**: New files not yet added to Git

### Key Files Explained

- **README.md**: This file - comprehensive project documentation
- **requirements.txt**: Python package dependencies (install with `pip install -r requirements.txt`)
- **LICENSE**: MIT License - open source license terms
- **.gitignore**: Specifies which files/folders Git should ignore (Data/ and Docs/ folders excluded)
- **Code/**: All Python scripts for data preparation, FCIT analysis, validation, and visualization
- **Plots/**: Generated visualizations (causal graphs, scatter plots, validation figures, property tables)
- **Results/**: FCIT text outputs, bootstrap validation CSVs, null model results

---

## 👤 Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub or contact the repository maintainer.

**GitHub**: [@vidigoat](https://github.com/vidigoat)

---

## 🙏 Acknowledgments

- ALFALFA survey team for providing the HI data
- SDSS/NSA team for providing the optical catalogues
- PyTetrad developers for the FCIT implementation

---

**Built with ❤️ for advancing our understanding of galaxy evolution through causal discovery.**
