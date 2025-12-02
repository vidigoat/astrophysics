"""
Generate hyperparameter optimization visualization plots.

Creates precision, recall, and F1 score plots as a function of penalty discount
for the optimal truncation limit, with uncertainty bands showing 16th-84th percentile ranges.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

RESULTS_DIR = 'Results'
PLOTS_DIR = 'Plots/HyperparameterSensitivity'
os.makedirs(PLOTS_DIR, exist_ok=True)

PRECISION_COLOR = '#1f77b4'
RECALL_COLOR = '#ff7f0e'
F1_COLOR = '#2ca02c'

def interpolate_metrics(penalty_discounts, metrics_dict, n_points=500):
    """Interpolate metrics for smooth plotting."""
    x_min, x_max = 0, max(penalty_discounts) + 25
    penalty_smooth = np.linspace(x_min, x_max, n_points)
    
    if len(penalty_discounts) >= 3:
        kind = 'quadratic'
    elif len(penalty_discounts) == 2:
        kind = 'linear'
    else:
        return penalty_discounts, metrics_dict
    
    interpolated = {}
    for key, values in metrics_dict.items():
        interp_func = interp1d(penalty_discounts, values, kind=kind, 
                               fill_value='extrapolate', bounds_error=False)
        interpolated[key] = interp_func(penalty_smooth)
    
    return penalty_smooth, interpolated

def create_plot(df, dataset_name, optimal_t, optimal_p):
    """Generate precision/recall/F1 plot for optimal truncation limit."""
    results_path = os.path.join(RESULTS_DIR, 
                               'mock_data_hyperparameter_sensitivity.csv' if dataset_name == 'TNG50'
                               else 'alfalfa_nsa_mock_data_hyperparameter_sensitivity.csv')
    results_df = pd.read_csv(results_path)
    filtered_df = results_df[results_df['truncation_limit'] == optimal_t]
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for truncation_limit={optimal_t}")
    
    grouped = filtered_df.groupby('penalty_discount').agg({
        'precision': ['mean', lambda x: x.quantile(0.16), lambda x: x.quantile(0.84)],
        'recall': ['mean', lambda x: x.quantile(0.16), lambda x: x.quantile(0.84)],
        'f1': ['mean', lambda x: x.quantile(0.16), lambda x: x.quantile(0.84)]
    })
    
    grouped.columns = ['_'.join(col).strip() if col[1] != '' else col[0] 
                      for col in grouped.columns.values]
    grouped.columns = [col.replace('<lambda_0>', 'p16').replace('<lambda_1>', 'p84') 
                      for col in grouped.columns]
    
    penalty_discounts = np.array(sorted(grouped.index.values))
    
    metrics = {
        'precision_mean': grouped['precision_mean'].values,
        'precision_p16': grouped['precision_p16'].values,
        'precision_p84': grouped['precision_p84'].values,
        'recall_mean': grouped['recall_mean'].values,
        'recall_p16': grouped['recall_p16'].values,
        'recall_p84': grouped['recall_p84'].values,
        'f1_mean': grouped['f1_mean'].values,
        'f1_p16': grouped['f1_p16'].values,
        'f1_p84': grouped['f1_p84'].values
    }
    
    penalty_smooth, metrics_smooth = interpolate_metrics(penalty_discounts, metrics)
    
    if metrics_smooth:
        x_plot = penalty_smooth
        prec_mean = metrics_smooth['precision_mean']
        prec_p16 = metrics_smooth['precision_p16']
        prec_p84 = metrics_smooth['precision_p84']
        rec_mean = metrics_smooth['recall_mean']
        rec_p16 = metrics_smooth['recall_p16']
        rec_p84 = metrics_smooth['recall_p84']
        f1_mean = metrics_smooth['f1_mean']
        f1_p16 = metrics_smooth['f1_p16']
        f1_p84 = metrics_smooth['f1_p84']
    else:
        x_plot = penalty_discounts
        prec_mean = metrics['precision_mean']
        prec_p16 = metrics['precision_p16']
        prec_p84 = metrics['precision_p84']
        rec_mean = metrics['recall_mean']
        rec_p16 = metrics['recall_p16']
        rec_p84 = metrics['recall_p84']
        f1_mean = metrics['f1_mean']
        f1_p16 = metrics['f1_p16']
        f1_p84 = metrics['f1_p84']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.fill_between(x_plot, prec_p16, prec_p84, alpha=0.25, color=PRECISION_COLOR, zorder=4)
    ax.fill_between(x_plot, rec_p16, rec_p84, alpha=0.25, color=RECALL_COLOR, zorder=4)
    ax.fill_between(x_plot, f1_p16, f1_p84, alpha=0.25, color=F1_COLOR, zorder=4)
    
    ax.plot(x_plot, prec_mean, color=PRECISION_COLOR, linewidth=2.5, label='Precision', zorder=5)
    ax.plot(x_plot, rec_mean, color=RECALL_COLOR, linewidth=2.5, label='Recall', zorder=5)
    ax.plot(x_plot, f1_mean, color=F1_COLOR, linewidth=2.5, label='F1', zorder=5)
    
    n_datasets = results_df['dataset_id'].nunique() if 'dataset_id' in results_df.columns else 100
    ax.set_xlabel('Penalty Discount', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision, recall and F1 statistics across {n_datasets} {dataset_name}-like mock datasets\n'
                 f'as a function of penalty_discount, at truncation_limit = {optimal_t}.',
                 fontsize=12, fontweight='normal', pad=15)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(0, max(penalty_discounts) + 25)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black', fancybox=True,
              title='Solid lines show the mean over datasets,\nbands show 16th-84th percentile range.',
              title_fontsize=9)
    ax.tick_params(direction='in', top=True, right=True, labelsize=11, length=5, width=1.1)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    plt.tight_layout()
    
    return fig

def main():
    tng50_df = pd.read_csv(os.path.join(RESULTS_DIR, 'mock_data_hyperparameter_summary.csv'))
    alfalfa_df = pd.read_csv(os.path.join(RESULTS_DIR, 'alfalfa_nsa_mock_data_hyperparameter_summary.csv'))
    
    fig1 = create_plot(tng50_df, 'TNG50', optimal_t=7, optimal_p=15)
    fig1.savefig(os.path.join(PLOTS_DIR, 'TNG50_hyperparameter_optimization.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    fig2 = create_plot(alfalfa_df, 'ALFALFA×NSA', optimal_t=7, optimal_p=35)
    fig2.savefig(os.path.join(PLOTS_DIR, 'ALFALFA_NSA_hyperparameter_optimization.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

if __name__ == '__main__':
    main()
