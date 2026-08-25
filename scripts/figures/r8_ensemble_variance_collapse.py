"""
R8: Ensemble variance-collapse strip plot -- individual chain all_zero_frac
values for the random-order ECC vs. the seed-constrained (targeted) ECC.

Reads results/metrics/ensemble_temporal_individual_zero_fracs.csv and
results/metrics/targeted_ensemble_temporal_individual_zero_fracs.csv.
Saves results/figures/r8_ensemble_variance_collapse.{png,pdf}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_style, save_figure

METRICS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'metrics'


def main():
    apply_style()

    random_ecc = pd.read_csv(METRICS_DIR / 'ensemble_temporal_individual_zero_fracs.csv')
    targeted_ecc = pd.read_csv(METRICS_DIR / 'targeted_ensemble_temporal_individual_zero_fracs.csv')

    fig, ax = plt.subplots(figsize=(6, 4))

    groups = [
        ('Random-order\nECC', random_ecc['all_zero_frac'].to_numpy(), '#1b6ca8'),
        ('Seed-constrained\nECC', targeted_ecc['all_zero_frac'].to_numpy(), '#c1440e'),
    ]

    rng = np.random.RandomState(0)
    for i, (label, values, color) in enumerate(groups):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, color=color, alpha=0.8, s=40, zorder=3)
        ax.hlines(values.mean(), i - 0.2, i + 0.2, color=color, linewidth=2, zorder=4)
        ax.text(i, values.max() + 0.02, f"mean={values.mean():.3f}\nsd={values.std():.3f}",
                ha='center', va='bottom', fontsize=9, color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel('individual chain all_zero_frac')
    ax.set_title('Ensemble variance collapse (10 chains each, temporal split)')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_figure(fig, 'r8_ensemble_variance_collapse')
    plt.close(fig)
    print("saved r8_ensemble_variance_collapse")


if __name__ == '__main__':
    main()
