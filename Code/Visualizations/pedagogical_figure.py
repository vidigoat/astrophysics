"""
Pedagogical figure: causal discovery via conditional independence testing.

Three causal scenarios for (M★, G_HI, SFR) that look identical under plain
correlation but are distinguishable by the conditional independence test:
  M★ ⊥ SFR | G_HI  (or not).

Mirrors and extends Desmond & Ramsey (2025) Fig. 2 with astrophysical variables
directly relevant to this paper's ALFALFA×NSA results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(REPO_ROOT, 'Plots', 'Pedagogical')
os.makedirs(PLOTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)
N = 900


# ── synthetic causal data ────────────────────────────────────────────────────

def make_s1(n):
    """Chain: M★ → G_HI → SFR.
    G_HI fully mediates M★ → SFR  →  M★ ⊥ SFR | G_HI"""
    M = rng.normal(0, 1, n)
    G = 0.90 * M   + rng.normal(0, 0.44, n)
    S = 0.90 * G   + rng.normal(0, 0.44, n)
    return M, G, S

def make_s2(n):
    """Feedback fork: SFR → M★  and  SFR → G_HI.
    Direct SFR → M★ edge survives conditioning  →  M★ ⊥̸ SFR | G_HI"""
    S = rng.normal(0, 1, n)
    M = 0.85 * S   + rng.normal(0, 0.55, n)
    G = 0.70 * S   + rng.normal(0, 0.70, n)
    return M, G, S

def make_s3(n):
    """Latent environment E → M★  and  E → SFR;  G_HI ← M★.
    E is unobserved, G_HI not on M★–SFR path  →  M★ ⊥̸ SFR | G_HI"""
    E = rng.normal(0, 1, n)
    M = 0.85 * E   + rng.normal(0, 0.55, n)
    S = 0.85 * E   + rng.normal(0, 0.55, n)
    G = 0.45 * M   + rng.normal(0, 0.88, n)   # M★ → G_HI (peripheral)
    return M, G, S

def residuals(y, x):
    c = np.polyfit(x, y, 1)
    return y - np.polyval(c, x)


# ── DAG drawing ──────────────────────────────────────────────────────────────

def _box(ax, xy, label, latent=False):
    kw = dict(ha='center', va='center', fontsize=9.5, fontweight='bold',
              transform=ax.transAxes, zorder=5)
    style = dict(boxstyle='round,pad=0.45', linewidth=1.4,
                 facecolor='#f5f5f5' if latent else 'white',
                 edgecolor='#555555',
                 linestyle='--' if latent else '-')
    ax.text(xy[0], xy[1], label, bbox=style, **kw)

def _arr(ax, src, dst, bidir=False, dashed=False, color='#333333', rad=0.0):
    style = '<->' if bidir else '->'
    ls    = 'dashed' if dashed else 'solid'
    ax.annotate('', xy=dst, xytext=src,
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle=style, color=color, lw=1.6,
                                linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=4)

def dag_s1(ax):
    """M★ → G_HI → SFR"""
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    _box(ax, (0.12, 0.52), 'M★')
    _box(ax, (0.50, 0.52), r'G$_\mathregular{HI}$')
    _box(ax, (0.88, 0.52), 'SFR')
    _arr(ax, (0.24, 0.52), (0.36, 0.52))
    _arr(ax, (0.64, 0.52), (0.76, 0.52))
    ax.text(0.50, 0.12, 'Mass-driven chain\nM★ → G$_{HI}$ → SFR',
            ha='center', fontsize=7.5, color='0.45', style='italic',
            transform=ax.transAxes)

def dag_s2(ax):
    """SFR → M★  and  SFR → G_HI"""
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    _box(ax, (0.50, 0.80), 'SFR')
    _box(ax, (0.15, 0.28), 'M★')
    _box(ax, (0.85, 0.28), r'G$_\mathregular{HI}$')
    _arr(ax, (0.37, 0.70), (0.22, 0.40))
    _arr(ax, (0.63, 0.70), (0.78, 0.40))
    ax.text(0.50, 0.02, 'Feedback fork\nSFR → M★  and  SFR → G$_{HI}$',
            ha='center', fontsize=7.5, color='0.45', style='italic',
            transform=ax.transAxes)

def dag_s3(ax):
    """E (latent) → M★,  E → SFR,  M★ → G_HI"""
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    _box(ax, (0.50, 0.82), 'E\n(halo)', latent=True)
    _box(ax, (0.18, 0.42), 'M★')
    _box(ax, (0.82, 0.42), 'SFR')
    _box(ax, (0.18, 0.10), r'G$_\mathregular{HI}$')
    _arr(ax, (0.38, 0.74), (0.26, 0.54), dashed=True)
    _arr(ax, (0.62, 0.74), (0.74, 0.54), dashed=True)
    _arr(ax, (0.18, 0.32), (0.18, 0.20))
    ax.text(0.50, -0.02, 'Latent halo E (- -) → M★, SFR;  M★ → G$_{HI}$',
            ha='center', fontsize=7.5, color='0.45', style='italic',
            transform=ax.transAxes)


# ── scatter panel ─────────────────────────────────────────────────────────────

def scat(ax, x, y, xlabel, ylabel, title, color, indep):
    ax.scatter(x, y, s=5, alpha=0.28, color=color, rasterized=True, linewidths=0)
    c  = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, np.polyval(c, xx), color=color, lw=1.8, alpha=0.9)

    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title,   fontsize=8.5, pad=5)
    ax.tick_params(labelsize=7.5, length=3, direction='in', top=True, right=True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color('0.55')

    if indep is not None:
        sym = r'$\perp$'  if indep else r'$\not\perp$'
        clr = '#1a7a1a'   if indep else '#cc2222'
        ax.text(0.97, 0.06, sym, transform=ax.transAxes,
                fontsize=15, ha='right', va='bottom', color=clr, fontweight='bold')


# ── assemble ──────────────────────────────────────────────────────────────────

def make_figure():
    M1, G1, S1 = make_s1(N)
    M2, G2, S2 = make_s2(N)
    M3, G3, S3 = make_s3(N)

    COLORS = ['#0173B2', '#DE8F05', '#CC4444']
    DAGS   = [dag_s1, dag_s2, dag_s3]
    DATA   = [(M1,G1,S1), (M2,G2,S2), (M3,G3,S3)]
    INDEP  = [True, False, False]          # only chain is independent

    fig = plt.figure(figsize=(11.5, 9.2))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(3, 3,
                          left=0.07, right=0.97,
                          top=0.88, bottom=0.09,
                          hspace=0.60, wspace=0.38)

    for r, (col, dag_fn, (M, G, S), indep) in enumerate(
            zip(COLORS, DAGS, DATA, INDEP)):

        # DAG
        ax0 = fig.add_subplot(gs[r, 0])
        dag_fn(ax0)

        # raw SFR vs M★ — all 3 scenarios look similar here (no ⊥ symbol)
        ax1 = fig.add_subplot(gs[r, 1])
        scat(ax1, M, S,
             xlabel=r'M$_\star$ (standardised)',
             ylabel=r'SFR (standardised)',
             title=r'SFR vs M$_\star$   (unconditioned)',
             color=col, indep=None)

        # conditioned: residuals after regressing out G_HI
        Mr = residuals(M, G)
        Sr = residuals(S, G)
        ax2 = fig.add_subplot(gs[r, 2])
        scat(ax2, Mr, Sr,
             xlabel=r'M$_\star$ $|$ G$_\mathrm{HI}$',
             ylabel=r'SFR $|$ G$_\mathrm{HI}$',
             title=r'SFR vs M$_\star$   conditioned on G$_\mathrm{HI}$',
             color=col, indep=indep)

    # column headers
    col_x  = [0.165, 0.500, 0.835]
    col_hdr = ['Causal structure', 'Raw correlation',
               r'Conditioned on G$_\mathrm{HI}$']
    for cx, hdr in zip(col_x, col_hdr):
        fig.text(cx, 0.915, hdr, ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color='0.2')

    # row labels
    row_y   = [0.79, 0.505, 0.225]
    row_lbl = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    for ry, lbl in zip(row_y, row_lbl):
        fig.text(0.005, ry, lbl, ha='left', va='center',
                 fontsize=9, fontweight='bold', color='0.4', rotation=90)

    # title
    fig.suptitle(
        'How causal discovery distinguishes physical mechanisms beyond correlation\n'
        r'Three scenarios for (M$_\star$, G$_\mathrm{HI}$, SFR) '
        r'degenerate under plain correlation but distinguishable by '
        r'M$_\star$ $\perp$ SFR $|$ G$_\mathrm{HI}$',
        fontsize=10, y=0.975, color='0.15')

    # legend
    leg = (r'$\perp$ = conditional independence  '
           r'(M$_\star$ $\perp$ SFR $|$ G$_\mathrm{HI}$) — '
           r'indicates G$_\mathrm{HI}$ mediates or causes both'
           '\n'
           r'$\not\perp$ = dependence persists — '
           r'direct causal link or latent confounder not blocked by G$_\mathrm{HI}$')
    fig.text(0.50, 0.005, leg, ha='center', va='bottom', fontsize=8, color='0.35',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#f8f8f8',
                       edgecolor='0.75', linewidth=0.8))
    return fig


if __name__ == '__main__':
    fig = make_figure()
    out = os.path.join(PLOTS_DIR, 'pedagogical_conditional_independence.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')
