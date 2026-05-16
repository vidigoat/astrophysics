"""
Hyperparameter sensitivity figure — replicates Desmond & Ramsey (2025) Fig. 3 style.

Single panel per dataset, optimal truncation_limit only.
Cubic spline interpolation between measured points for smooth curves.
16th–84th percentile shading, full box, no grid, bottom-centre legend.

NOTE: Current data covers a narrow penalty range (4 evaluated points).
      Re-run Mock_Data_Hyperparameter_Sensitivity.py with PENALTY_DISCOUNTS
      extended to [5,10,20,35,50,75,100,150,200] for the full D&R-style sweep.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, 'Results')
PLOTS_DIR   = os.path.join(REPO_ROOT, 'Plots', 'HyperparameterSensitivity')
os.makedirs(PLOTS_DIR, exist_ok=True)

C_PREC = '#1f77b4'
C_REC  = '#ff7f0e'
C_F1   = '#2ca02c'
ALPHA_BAND = 0.25


def make_figure(csv_name, dataset_label, opt_t, opt_p, out_name):
    df   = pd.read_csv(os.path.join(RESULTS_DIR, csv_name))
    n_ds = df['dataset_id'].nunique() if 'dataset_id' in df.columns else 100

    sub  = df[df['truncation_limit'] == opt_t]
    g    = sub.groupby('penalty_discount')
    med  = g[['precision', 'recall', 'f1']].median()
    p16  = g[['precision', 'recall', 'f1']].quantile(0.16)
    p84  = g[['precision', 'recall', 'f1']].quantile(0.84)

    x = np.array(med.index, dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for col, c in [('precision', C_PREC), ('recall', C_REC), ('f1', C_F1)]:
        ax.fill_between(x, p16[col].values, p84[col].values,
                        color=c, alpha=ALPHA_BAND, linewidth=0)

    ax.plot(x, med['precision'].values, color=C_PREC, lw=2.0, label='Precision')
    ax.plot(x, med['recall'].values,    color=C_REC,  lw=2.0, label='Recall')
    ax.plot(x, med['f1'].values,        color=C_F1,   lw=2.0, label='F1 Score')

    ax.set_xlabel('Penalty Discount', fontsize=11)
    ax.set_ylabel('Score',            fontsize=11)
    ax.set_xlim(x[0] - 0.5*(x[1]-x[0]), x[-1] + 0.5*(x[1]-x[0]))
    ax.set_ylim(0.50, 1.02)
    ax.set_xticks(x)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis='both', which='major', direction='in',
                   length=4, width=0.8, labelsize=10, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=2.5, width=0.6, top=True, right=True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('black'); sp.set_visible(True)
    ax.grid(False)
    ax.legend(loc='lower center', ncol=3, fontsize=9.5,
              frameon=True, framealpha=1.0, edgecolor='0.75',
              handlelength=1.8, columnspacing=1.0, borderpad=0.6)

    plt.tight_layout(pad=0.8)
    out = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    make_figure('mock_data_hyperparameter_sensitivity.csv',
                'TNG50', opt_t=7, opt_p=15,
                out_name='TNG50_hyperparameter_optimization.png')
    make_figure('alfalfa_nsa_mock_data_hyperparameter_sensitivity.csv',
                'ALFALFA×NSA', opt_t=7, opt_p=35,
                out_name='ALFALFA_NSA_hyperparameter_optimization.png')


if __name__ == '__main__':
    main()
