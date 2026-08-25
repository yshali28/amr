"""
Shared matplotlib settings and save helper for publication figures.

Example:
    from src.visualization.style import apply_style, SPECIES_COLORS, save_figure

    apply_style()
    fig, ax = plt.subplots()
    ...
    save_figure(fig, 'r5_all_zero_fraction')
"""

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'

SPECIES_COLORS = {
    'E. coli': '#1b6ca8',
    'S. aureus': '#c1440e',
    'K. pneumoniae': '#2e7d32',
    'P. aeruginosa': '#6a3d9a',
}


def apply_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'figure.dpi': 100,
    })


def save_figure(fig, name):
    """
    Save fig as both results/figures/{name}.png (300 dpi) and .pdf (vector).

    Example:
        save_figure(fig, 'r5_all_zero_fraction')
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')
