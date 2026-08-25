"""
M5: Three correlation heatmaps side by side (E. coli, S. aureus,
K. pneumoniae), shared |correlation| colour scale 0-1.

Reads results/metrics/correlation_matrix_{ecoli,saureus,kpneumoniae}.csv.
Saves results/figures/m5_correlation_heatmaps.{png,pdf}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_style, save_figure

METRICS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'metrics'

PANELS = [
    ('ecoli', 'E. coli'),
    ('saureus', 'S. aureus'),
    ('kpneumoniae', 'K. pneumoniae'),
]


def short_label(name, max_len=14):
    return name if len(name) <= max_len else name[:max_len - 1] + '…'


def main():
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im = None

    for ax, (label, title) in zip(axes, PANELS):
        corr = pd.read_csv(METRICS_DIR / f'correlation_matrix_{label}.csv', index_col=0)
        abs_corr = corr.abs()

        im = ax.imshow(abs_corr.values, vmin=0, vmax=1, cmap='viridis')
        ticks = range(len(abs_corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([short_label(c) for c in abs_corr.columns], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([short_label(c) for c in abs_corr.index], fontsize=8)
        ax.set_title(title)

        for i in range(len(abs_corr.index)):
            for j in range(len(abs_corr.columns)):
                ax.text(j, i, f"{abs_corr.values[i, j]:.2f}", ha='center', va='center',
                         color='white' if abs_corr.values[i, j] < 0.6 else 'black', fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.8, label='|correlation|')
    fig.suptitle('Label correlation structure by species (final modeling panels)')

    save_figure(fig, 'm5_correlation_heatmaps')
    plt.close(fig)
    print("saved m5_correlation_heatmaps")


if __name__ == '__main__':
    main()
