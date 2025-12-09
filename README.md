# Causal Structure in Galaxies: Comparing Simulations and Observations

Code for reproducing "Causal Structure in Galaxies: Comparing Simulations and Observations" by Vidit Patankar.

This repository implements causal discovery analysis using the Fast Causal Inference Test (FCIT) algorithm to compare gas-rich HI-selected galaxies (ALFALFA × NSA) with the full optical galaxy population (NSA-only) and cosmological simulations (TNG50). The analysis includes hyperparameter optimization on mock data, comprehensive algorithm comparison (FCIT vs PC vs FCI), bootstrap validation, null model tests, and uncertainty quantification.

## Code Structure

### Data Preparation (`Code/DataPrep/`)

- `ALFALFA_X_NSA_DATAPREP.py`: Prepare ALFALFA × NSA cross-matched dataset
- `NSA_Data_Preparation.py`: Prepare NASA–Sloan Atlas (NSA) dataset
- `TNG50_Data_Preparation.py`: Prepare IllustrisTNG TNG50 simulation data

### FCIT Causal Discovery (`Code/FCIT/`)

- `ALFALFA_NSA_FCIT.py`: Run FCIT on ALFALFA × NSA dataset
- `NSA_FCIT.py`: Run FCIT on NSA-only dataset
- `TNG50_FCIT.py`: Run FCIT on TNG50 simulation dataset

### Analysis (`Code/Analysis/`)

- `Mock_Data_Hyperparameter_Sensitivity.py`: Generate mock datasets and optimize FCIT hyperparameters (truncation limit, penalty discount) using ground truth evaluation metrics (SHD, precision, recall, F1)
- `ALFALFA_NSA_Hyperparameter_Optimization.py`: Hyperparameter optimization for ALFALFA × NSA dataset
- `Comprehensive_Algorithm_Comparison.py`: Compare FCIT, PC, and FCI algorithms across all datasets (edge density, orientation fraction, V-structures, Jaccard similarity)
- `Graph_Metrics_Uncertainty.py`: Compute uncertainty estimates (95% confidence intervals) for graph metrics from bootstrap runs
- `Run_PC_FCI_All_Datasets.py`: Run PC and FCI algorithms on all datasets for comparison
- `TNG50_Noise_Selection_Test.py`: Test robustness to observational noise and ALFALFA-like selection effects
- `NSA_Subsample_Size_Test.py`: Test if edge density differences are due to sample size by subsampling NSA to match TNG50 size

### Validation (`Code/Validation/`)

- `ALFALFA_NSA_Bootstrap.py`: Bootstrap validation for ALFALFA × NSA (50 runs, 80% subsampling)
- `NSA_Bootstrap.py`: Bootstrap validation for NSA (10 runs, 80% subsampling)
- `TNG50_Bootstrap.py`: Bootstrap validation for TNG50 (50 runs, 80% subsampling)
- `ALFALFA_NSA_NullModel.py`: Null model test (shuffled data) for ALFALFA × NSA
- `NSA_NullModel.py`: Null model test for NSA
- `TNG50_NullModel.py`: Null model test for TNG50

## Data Availability

- **NASA–Sloan Atlas (NSA) v1.0.1**: [https://www.sdss4.org/dr17/manga/manga-target-selection/nsa/](https://www.sdss4.org/dr17/manga/manga-target-selection/nsa/)
- **ALFALFA Survey**: [https://egg.astro.cornell.edu/alfalfa/data/](https://egg.astro.cornell.edu/alfalfa/data/)
- **IllustrisTNG (TNG50) Simulation**: [https://www.tng-project.org/data/docs/specifications/](https://www.tng-project.org/data/docs/specifications/)

**Note**: The ALFALFA×NSA cross-match used in this work is not publicly released.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Contact

Vidit Patankar: vidit.patankar16@gmail.com
