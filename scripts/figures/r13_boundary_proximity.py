"""
R13: Boundary-proximity histograms, E. coli vs. K. pneumoniae -- actual
predicted probabilities from the Binary Relevance baseline (temporal
split), not simulated.

Reads results/metrics/baseline_probas_{ecoli,kpneumoniae}_temporal.csv.
Saves results/figures/r13_boundary_proximity.{png,pdf}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import SPECIES_COLORS, apply_style, save_figure

METRICS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'metrics'


def main():
    apply_style()

    ecoli_proba = pd.read_csv(METRICS_DIR / 'baseline_probas_ecoli_temporal.csv')
    kp_proba = pd.read_csv(METRICS_DIR / 'baseline_probas_kpneumoniae_temporal.csv')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, (label, df, color) in zip(axes, [
        ('E. coli', ecoli_proba, SPECIES_COLORS['E. coli']),
        ('K. pneumoniae', kp_proba, SPECIES_COLORS['K. pneumoniae']),
    ]):
        all_proba = df.values.flatten()
        ax.hist(all_proba, bins=40, color=color, alpha=0.85)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1.2, label='decision boundary (0.5)')

        near_boundary = ((all_proba > 0.4) & (all_proba < 0.6)).mean()
        ax.set_title(f'{label}\n{near_boundary:.1%} of predictions in [0.4, 0.6]')
        ax.set_xlabel('predicted probability (pooled across antibiotics)')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('count')
    fig.suptitle('Boundary proximity of baseline predictions (temporal split test set)')

    fig.tight_layout()
    save_figure(fig, 'r13_boundary_proximity')
    plt.close(fig)
    print("saved r13_boundary_proximity")


if __name__ == '__main__':
    main()
