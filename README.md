# Causal Structure in Galaxies: A Comparative Analysis of HI-Selected and Optical Populations

This repository contains the code used in the paper "Causal Structure in Galaxies: A Comparative Analysis of HI-Selected and Optical Populations". The code implements causal discovery analysis using the Fast Causal Inference Test (FCIT) algorithm to compare gas-rich HI-selected galaxies (ALFALFA × NSA) with the full optical galaxy population (NSA-only) and cosmological simulations (TNG50), identifying fundamental differences in how physical properties drive galaxy evolution.

**Author:** Vidit Patankar

## Repository Structure

```
astrophysics/
├── Code/
│   ├── DataPrep/          # Data preparation scripts
│   │   ├── ALFALFA_X_NSA_DATAPREP.py
│   │   ├── NSA_Data_Preparation.py
│   │   └── TNG50_Data_Preparation.py
│   ├── FCIT/              # FCIT causal discovery scripts
│   │   ├── ALFALFA_NSA_FCIT.py
│   │   ├── NSA_FCIT.py
│   │   └── TNG50_FCIT.py
│   ├── Analysis/          # Hyperparameter optimization
│   │   ├── ALFALFA_NSA_Hyperparameter_Optimization.py
│   │   ├── Mock_Data_Hyperparameter_Sensitivity.py
│   │   └── create_final_hyperparameter_graphs.py
│   ├── Validation/        # Bootstrap and null model validation
│   │   ├── ALFALFA_NSA_Bootstrap.py
│   │   ├── ALFALFA_NSA_NullModel.py
│   │   ├── NSA_Bootstrap.py
│   │   ├── NSA_NullModel.py
│   │   ├── TNG50_Bootstrap.py
│   │   └── TNG50_NullModel.py
│   └── Visualizations/    # Plotting scripts
│       ├── Bootstrap_ALFALFA_NSA_plot.py
│       ├── Bootstrap_plot_utils.py
│       ├── Combined_Dataset_Distribution.py
│       ├── NullModel_Visualization.py
│       ├── NSA_Bootstrap_plot.py
│       ├── NSA_Scatter_plot.py
│       ├── NSA_Scatter_plot_Part2.py
│       ├── NSA_Validation_Figure.py
│       ├── TNG50_Bootstrap_plot.py
│       ├── TNG50_Scatter_plot.py
│       ├── TNG50_Scatter_plot_Part2.py
│       └── Alfalfa_Causal_Scatter.py
├── Data/                  # Data files (not included in repo - too large)
│   ├── alfalfa_nsa_final_13props.pkl
│   ├── nsa_final_10props.pkl
│   ├── tng50_final.pkl
│   └── MockDatasets/
├── Results/               # FCIT results and validation outputs
│   ├── *.txt              # FCIT causal graph outputs
│   ├── *.csv              # Bootstrap validation results
│   └── *.pkl              # Null model test results
└── Plots/                 # Generated figures
    ├── CausalStructure/
    ├── HyperparameterSensitivity/
    ├── ScatterPlots/
    └── ValidationPlots/
```

## Dependencies

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- pytetrad (for FCIT algorithm)
- graphviz (for graph visualization)
- scipy
- sklearn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/astrophysics.git
cd astrophysics
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn pytetrad scipy scikit-learn graphviz
```

3. Install Graphviz (required for graph visualization):
   - Windows: Download from [Graphviz website](https://graphviz.org/download/)
   - Linux: `sudo apt-get install graphviz`
   - macOS: `brew install graphviz`

## Usage

### 1. Data Preparation

Prepare datasets for analysis:

```bash
python Code/DataPrep/ALFALFA_X_NSA_DATAPREP.py
python Code/DataPrep/NSA_Data_Preparation.py
python Code/DataPrep/TNG50_Data_Preparation.py
```

### 2. Hyperparameter Optimization

Optimize FCIT hyperparameters using mock data:

```bash
python Code/Analysis/Mock_Data_Hyperparameter_Sensitivity.py
python Code/Analysis/ALFALFA_NSA_Hyperparameter_Optimization.py
python Code/Analysis/create_final_hyperparameter_graphs.py
```

### 3. Run FCIT Causal Discovery

Run FCIT on each dataset:

```bash
python Code/FCIT/ALFALFA_NSA_FCIT.py
python Code/FCIT/NSA_FCIT.py
python Code/FCIT/TNG50_FCIT.py
```

### 4. Validation

Run bootstrap and null model validation:

```bash
# Bootstrap validation
python Code/Validation/ALFALFA_NSA_Bootstrap.py
python Code/Validation/NSA_Bootstrap.py
python Code/Validation/TNG50_Bootstrap.py

# Null model tests
python Code/Validation/ALFALFA_NSA_NullModel.py
python Code/Validation/NSA_NullModel.py
python Code/Validation/TNG50_NullModel.py
```

### 5. Generate Visualizations

Create all plots:

```bash
python Code/Visualizations/Bootstrap_ALFALFA_NSA_plot.py
python Code/Visualizations/NSA_Bootstrap_plot.py
python Code/Visualizations/TNG50_Bootstrap_plot.py
python Code/Visualizations/NullModel_Visualization.py
python Code/Visualizations/Combined_Dataset_Distribution.py
# ... and other visualization scripts
```

## Hyperparameters

The optimized hyperparameters for each dataset are:

- **ALFALFA × NSA**: `truncation_limit = 7`, `penalty_discount = 35`
- **NSA-only**: `truncation_limit = 14`, `penalty_discount = 50`
- **TNG50**: `truncation_limit = 7`, `penalty_discount = 15`

These were determined through hyperparameter optimization on mock datasets to maximize F1 score.

## Results

FCIT results are saved in `Results/`:
- `alfalfa_nsa_fcit_t7_p35.txt` - ALFALFA × NSA causal graph
- `nsa_fcit_t14_p50.txt` - NSA-only causal graph
- `tng50_fcit_t7_p15.txt` - TNG50 causal graph

Validation results:
- Bootstrap validation CSVs show edge recovery rates
- Null model test PKLs contain statistical tests for edge significance

## Key Findings

1. **Mass-driven hierarchy**: Stellar and baryonic mass serve as primary causal drivers across all datasets
2. **Gas physics**: Gas mass exhibits strong causal relationships with black hole mass and structural properties
3. **Metallicity enrichment**: Clear enrichment pathways from stellar to gas-phase metallicity
4. **Dataset differences**: HI-selected galaxies show distinct causal structures compared to optical-only populations

## Citation

If you use this code, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please contact Vidit Patankar or open an issue on GitHub.
