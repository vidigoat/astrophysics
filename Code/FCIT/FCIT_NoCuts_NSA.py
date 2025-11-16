"""
Run FCIT on NSA-only dataset with NO physical parameter cuts.
Only removes non-finite values for comparison with cut dataset.
"""
import os
import pickle
import numpy as np
from astropy.io import fits
import pandas as pd
import pytetrad.tools.TetradSearch as ts
import graphviz as gviz

# Add Graphviz to PATH if GRAPHVIZ_BIN environment variable is set
graphviz_bin = os.environ.get('GRAPHVIZ_BIN')
if graphviz_bin:
    os.environ["PATH"] += os.pathsep + graphviz_bin

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Get FITS file path from environment variable or use default
FITS_PATH = os.environ.get('NSA_FITS_PATH', os.path.join(REPO_ROOT, 'Data', 'nsa_v1_0_1.fits'))
OUTPUT_DIR = os.path.join(REPO_ROOT, "Results")
VIZ_DIR = os.path.join(REPO_ROOT, "Plots", "CausalStructure")

print("=" * 70)
print("FCIT ANALYSIS - NSA-ONLY (NO CUTS)")
print("=" * 70)

# Load and prepare data
print("\nLoading FITS file...")
with fits.open(FITS_PATH, memmap=True) as hdul:
    data = hdul[1].data
    n_rows = len(data)
    print(f"Total galaxies: {n_rows:,}")

    sersic_n = np.asarray(data["SERSIC_N"])
    zdist = np.asarray(data["ZDIST"])
    elpetro_b300 = np.asarray(data["ELPETRO_B300"])
    elpetro_mets = np.asarray(data["ELPETRO_METS"])
    elpetro_mtol_all = np.asarray(data["ELPETRO_MTOL"])
    elpetro_ba = np.asarray(data["ELPETRO_BA"])
    elpetro_th50_r = np.asarray(data["ELPETRO_TH50_R"])
    elpetro_mass = np.asarray(data["ELPETRO_MASS"])
    elpetro_absmag = np.asarray(data["ELPETRO_ABSMAG"])

elpetro_mtol_r = elpetro_mtol_all[:, 4]
color_u_r = elpetro_absmag[:, 2] - elpetro_absmag[:, 4]
elpetro_absmag_r = elpetro_absmag[:, 4]

data_dict = {
    "COLOR_U_R": color_u_r,
    "ELPETRO_B300": elpetro_b300,
    "SERSIC_N": sersic_n,
    "ELPETRO_METS": elpetro_mets,
    "ELPETRO_MTOL": elpetro_mtol_r,
    "ELPETRO_BA": elpetro_ba,
    "ELPETRO_TH50_R": elpetro_th50_r,
    "ZDIST": zdist,
    "ELPETRO_MASS": elpetro_mass,
    "ELPETRO_ABSMAG_R": elpetro_absmag_r,
}

# Only remove non-finite values (NO physical cuts)
print("\nRemoving non-finite values only...")
valid_mask = np.ones(n_rows, dtype=bool)
for var_name, var_data in data_dict.items():
    finite_mask = np.isfinite(var_data)
    n_bad = np.count_nonzero(~finite_mask)
    if n_bad > 0:
        print(f"  {var_name}: {n_bad:,} non-finite entries")
    valid_mask &= finite_mask

n_valid = np.count_nonzero(valid_mask)
print(f"\nGalaxies with complete data: {n_valid:,} / {n_rows:,}")
print(f"Removed: {n_rows - n_valid:,} galaxies (non-finite values only)")

for key in data_dict:
    data_dict[key] = data_dict[key][valid_mask]

# Prepare DataFrame
var_names = list(data_dict.keys())
full_data = np.column_stack([data_dict[var] for var in var_names])
df = pd.DataFrame(full_data, columns=var_names)

print(f"\nRunning FCIT on {len(df):,} galaxies, {len(var_names)} properties")
print("Parameters: truncation_limit=14, penalty_discount=50, alpha=0.01")

# Run FCIT
search = ts.TetradSearch(df)
search.set_verbose(False)
search.use_basis_function_lrt(truncation_limit=14, alpha=0.01)
search.use_basis_function_bic(truncation_limit=14, penalty_discount=50)
search.run_fcit()

graph_str = str(search.get_java())
lines = [l for l in graph_str.split("\n") if l.strip() and not l.startswith("Graph")]

# Count edges
edges = [l for l in lines if any(a in l for a in ["-->", "<--", "o-o", "o->"])]
n_edges = len(edges)

print(f"\n{'='*70}")
print(f"RESULTS: {n_edges} edges found")
print(f"{'='*70}")

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

output_file = os.path.join(OUTPUT_DIR, "fcit_nocuts_nsa_t14_p50.txt")
with open(output_file, 'w') as f:
    f.write("FCIT Causal Graph - NSA Dataset (NO CUTS)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Galaxies: {len(df):,}\n")
    f.write("Parameters:\n")
    f.write("  truncation_limit = 14\n")
    f.write("  penalty_discount = 50\n")
    f.write("  alpha            = 0.01\n\n")
    f.write("Variables:\n")
    for i, var in enumerate(var_names, 1):
        f.write(f"  {i:2d}. {var}\n")
    f.write("\n" + "=" * 70 + "\n\n")
    f.write("Graph Nodes:\n\n")
    f.write(";".join(var_names) + "\n\n\n")
    f.write("Graph Edges:\n\n")
    for i, edge in enumerate(edges, 1):
        f.write(f"{i}. {edge}\n")
    f.write("\n")

print(f"Results saved to: {output_file}")
print(f"\nEdges found:\n")
for i, edge in enumerate(edges, 1):
    print(f"  {i}. {edge}")

# Generate graph visualization
print("\n" + "=" * 70)
print("Generating graph visualization...")
print("=" * 70)

dot_graph = search.get_dot()
graph_viz = gviz.Source(dot_graph)
base_name = "fcit_nocuts_nsa_t14_p50"
png_path = graph_viz.render(
    filename=base_name,
    directory=VIZ_DIR,
    format="png",
    cleanup=True,
)

print(f"Graph visualization saved to: {png_path}")

print("\n" + "=" * 70)
print("FCIT analysis complete (no cuts)")
print("=" * 70)

