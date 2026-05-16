"""
Corner plots: lower triangle (slant) only, distribution not dots.
Darker where density is higher; distinct colour per dataset (NSA=blue, TNG50=green, ALFALFA=orange).
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # type: ignore[reportMissingImports]
from matplotlib.ticker import MaxNLocator

# Publication settings
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.edgecolor'] = '0.25'
mpl.rcParams['xtick.color'] = '0.2'
mpl.rcParams['ytick.color'] = '0.2'

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
PLOTS_DIR = REPO_ROOT / "Plots" / "CornerPlots"

CONTOUR_QUANTILES = (0.393, 0.865, 0.989)
OUTER_PARTICLES_QUANTILE = 0.995
# Subsample for KDE (contours) to keep plot fast; contours look the same
KDE_SAMPLE = 2500
CONTOUR_GRID = 60  # 60x60 grid instead of 100x100

# Per-dataset colours: strong contrast so contours and histograms are clearly visible
# fill_1 = innermost (1σ, darkest), fill_3 = outermost (3σ, lightest)
COLORS = {
    'NSA': {
        'hist_fill': '#B8D4F0',
        'hist_edge': '#1A3A5C',
        'scatter': '#5B9BD5',
        'fill_1': '#1A3A5C',
        'fill_2': '#2E5F8C',
        'fill_3': '#7EB0D4',
    },
    'TNG50': {
        'hist_fill': '#B8E0B8',
        'hist_edge': '#1B3D1B',
        'scatter': '#4A9D4E',
        'fill_1': '#1B5E20',
        'fill_2': '#2E7D32',
        'fill_3': '#81C784',
    },
    'ALFALFA': {
        'hist_fill': '#FFE0B2',
        'hist_edge': '#E65100',
        'scatter': '#F39C12',
        'fill_1': '#BF360C',
        'fill_2': '#E65100',
        'fill_3': '#FFB74D',
    }
}

NSA_VARS = ["ZDIST", "ELPETRO_ABSMAG_R", "log_B300", "ELPETRO_MASS", 
            "SERSIC_N", "ELPETRO_BA", "ELPETRO_TH50_R"]
NSA_LABELS = {
    "ZDIST": "ZDIST",
    "ELPETRO_ABSMAG_R": "ABSMAG",
    "log_B300": "log(B300)",
    "ELPETRO_MASS": "log(MASS)",
    "SERSIC_N": "SERSIC_N",
    "ELPETRO_BA": "ELPETRO_BA",
    "ELPETRO_TH50_R": "ELPETRO_TH50_R",
}

TNG50_VARS = ["STELLAR_MASS", "BARYONIC_MASS", "GAS_MASS", "HALFMASS_RAD",
              "SFR", "PHOTOMETRIC_R", "COLOUR"]
TNG50_LABELS = {v: v for v in TNG50_VARS}

ALFALFA_VARS = NSA_VARS
ALFALFA_LABELS = NSA_LABELS


def load_data(dataset_name: str) -> pd.DataFrame:
    dataset = dataset_name.upper()
    if dataset == "NSA":
        path = DATA_DIR / "nsa_final_10props.pkl"
        vars_use = NSA_VARS
    elif dataset == "TNG50":
        path = DATA_DIR / "tng50_final.pkl"
        vars_use = TNG50_VARS
    elif dataset == "ALFALFA":
        path = DATA_DIR / "alfalfa_nsa_final_13props.pkl"
        vars_use = ALFALFA_VARS
    else:
        raise ValueError(f"Unknown: {dataset_name}")
    
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    
    if "log_B300" in vars_use and "log_B300" not in df.columns:
        b300 = df.get("ELPETRO_B300", np.ones(len(df)) * np.nan)
        df["log_B300"] = np.log10(np.maximum(np.asarray(b300, float), 1e-10))
    
    cols = [c for c in vars_use if c in df.columns]
    return df[cols].dropna()


def get_labels(dataset_name: str):
    dataset = dataset_name.upper()
    if dataset == "NSA":
        return NSA_LABELS
    elif dataset == "TNG50":
        return TNG50_LABELS
    else:
        return ALFALFA_LABELS


def kde_levels(zz, quantiles):
    """Get contour levels for quantiles."""
    flat = np.sort(zz.ravel())[::-1]
    cum = np.cumsum(flat)
    if cum[-1] <= 0:
        return [0] * len(quantiles)
    levels = []
    for q in quantiles:
        idx = np.searchsorted(cum, q * cum[-1])
        levels.append(float(flat[min(idx, len(flat)-1)]))
    return levels


def plot_corner(dataset_name: str, max_points: int = 10000, dpi: int = 300):
    """
    Create corner plot matching reference with filled contours.
    
    Parameters
    ----------
    dataset_name : str
        'NSA' (blue), 'TNG50' (green), or 'ALFALFA' (orange)
    max_points : int
        Number of points to plot (20000 for dense look)
    dpi : int
        Resolution
    """
    print(f"\n{'='*60}")
    print(f"Generating {dataset_name.upper()} corner plot...")
    print(f"{'='*60}")
    
    df = load_data(dataset_name)
    print(f"Loaded {len(df)} data points")
    
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
        print(f"Using {max_points} points for plotting")
    
    labels_dict = get_labels(dataset_name)
    vars_list = [c for c in df.columns if c in labels_dict]
    if not vars_list:
        vars_list = list(df.columns)
        labels_dict = {c: c for c in vars_list}
    
    # Get colors for this dataset
    colors = COLORS[dataset_name.upper()]
    print(f"Using color scheme: {dataset_name.upper()}")
    
    n = len(vars_list)
    
    # Create figure
    fig, axes = plt.subplots(n, n, figsize=(2.6*n, 2.6*n))
    if n == 1:
        axes = np.array([[axes]])
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.03, wspace=0.03,
                       left=0.09, right=0.97, bottom=0.09, top=0.94)
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            # Lower triangle only (slant): only plot when i >= j
            if i < j:
                ax.set_visible(False)
                continue

            var_y = vars_list[i]
            var_x = vars_list[j]
            x = np.asarray(df[var_x], dtype=float)
            y = np.asarray(df[var_y], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            for spine in ax.spines.values():
                spine.set_color('0.2')
                spine.set_linewidth(1.0)

            if i == j:
                # Diagonal: filled histogram with clear dark edge (professional contrast)
                n_bins = min(50, max(25, len(x) // 150))
                ax.hist(x, bins=n_bins, density=True, histtype='stepfilled',
                        facecolor=colors['hist_fill'], edgecolor=colors['hist_edge'],
                        linewidth=1.4, alpha=1.0)
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
                ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.6)
                ax.set_axisbelow(True)
            else:
                # OFF-DIAGONAL: clean 3-band distribution, then particles only in outer region
                if len(x) >= 30:
                    try:
                        # Subsample for KDE to keep runtime low; contours look the same
                        n_kde = min(KDE_SAMPLE, len(x))
                        idx_kde = np.random.RandomState(42).choice(len(x), size=n_kde, replace=False)
                        x_kde, y_kde = x[idx_kde], y[idx_kde]
                        kde = scipy_stats.gaussian_kde(np.vstack([x_kde, y_kde]), bw_method='scott')
                        x_min, x_max = x.min(), x.max()
                        y_min, y_max = y.min(), y.max()
                        dx = max((x_max - x_min) * 0.08, 1e-9)
                        dy = max((y_max - y_min) * 0.08, 1e-9)
                        g = CONTOUR_GRID
                        xi = np.linspace(x_min - dx, x_max + dx, g)
                        yi = np.linspace(y_min - dy, y_max + dy, g)
                        xx, yy = np.meshgrid(xi, yi)
                        zz = np.reshape(kde(np.vstack([xx.ravel(), yy.ravel()])), xx.shape)
                        raw_levels = kde_levels(zz, CONTOUR_QUANTILES)
                        levels = sorted(set(l for l in raw_levels if np.isfinite(l)))
                        z_max = np.nanmax(zz)
                        if len(levels) >= 3 and levels[0] < levels[1] < levels[2] and z_max > levels[2]:
                            # Clean 3 bands: inner (1σ), mid (2σ), outer (3σ) with distinct boundaries
                            ax.contourf(xx, yy, zz, levels=[levels[2], z_max],
                                        colors=[colors['fill_1']], alpha=0.92, zorder=2)
                            ax.contourf(xx, yy, zz, levels=[levels[1], levels[2]],
                                        colors=[colors['fill_2']], alpha=0.78, zorder=2)
                            ax.contourf(xx, yy, zz, levels=[levels[0], levels[1]],
                                        colors=[colors['fill_3']], alpha=0.58, zorder=2)
                            ax.contour(xx, yy, zz, levels=levels,
                                       colors=colors['hist_edge'], linewidths=1.15,
                                       alpha=1.0, zorder=3)
                        elif len(levels) >= 2 and levels[0] < levels[1]:
                            ax.contourf(xx, yy, zz, levels=[levels[1], z_max],
                                        colors=[colors['fill_1']], alpha=0.88, zorder=2)
                            ax.contourf(xx, yy, zz, levels=[levels[0], levels[1]],
                                        colors=[colors['fill_2']], alpha=0.65, zorder=2)
                            ax.contour(xx, yy, zz, levels=levels,
                                       colors=colors['hist_edge'], linewidths=1.15, alpha=1.0, zorder=3)
                        # Particles only *not* in the distribution: density below 99.5% contour
                        dens_at_points = kde(np.vstack([x, y]))
                        outer_levels = kde_levels(zz, (OUTER_PARTICLES_QUANTILE,))
                        outer_threshold = outer_levels[0] if outer_levels else levels[0]
                        not_in_distribution = dens_at_points < outer_threshold
                        if np.any(not_in_distribution):
                            ax.scatter(x[not_in_distribution], y[not_in_distribution],
                                       s=1.6, alpha=0.36, color=colors['scatter'],
                                       edgecolors='none', rasterized=True, zorder=4)
                    except Exception:
                        pass

            ax.tick_params(labelsize=10, direction='in', top=True, right=True,
                           length=4, width=0.9, colors='0.2')
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=False))
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=False))
            if i != j:
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.4)
                ax.set_axisbelow(True)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels_dict.get(var_x, var_x), fontsize=11, color='0.15')
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(labels_dict.get(var_y, var_y), fontsize=11, color='0.15')

    # Title and caption
    dlabel = 'ALFALFA×NSA' if dataset_name.upper() == 'ALFALFA' else dataset_name.upper()
    title = f"Distributions and pairwise correlations of the {dlabel} data used as input to the causal discovery algorithm."
    fig.text(0.5, 0.97, title, ha='center', fontsize=12, va='top', fontweight='bold')
    caption = "The contour levels contain 39.3, 86.5 and 98.9 per cent of the points (1, 2 and 3σ)."
    fig.text(0.5, 0.015, caption, ha='center', fontsize=9, style='italic', color='0.35')
    
    # Save PNG only (no PDF unless requested)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOTS_DIR / f"corner_{dataset_name.lower()}_FINAL.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close(fig)
    print(f"\n✓ Saved: {out_png.name}")
    print(f"{'='*60}\n")
    return out_png


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NSA", "TNG50", "ALFALFA"], 
                       default="NSA")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--points", type=int, default=20000,
                       help="Number of points (20000 for dense)")
    args = parser.parse_args()
    
    if args.all:
        for name in ["NSA", "TNG50", "ALFALFA"]:
            color = {'NSA': 'BLUE', 'TNG50': 'GREEN', 'ALFALFA': 'ORANGE'}[name]
            print(f"\n>>> {name} ({color})")
            try:
                plot_corner(name, max_points=args.points, dpi=args.dpi)
            except FileNotFoundError as e:
                print(f"SKIP: {e}")
    else:
        plot_corner(args.dataset, max_points=args.points, dpi=args.dpi)


if __name__ == "__main__":
    main()