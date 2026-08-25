"""
R5: All-zero fraction vs. true rate, all E. coli chain configurations
(baseline, Option A/D/E, random-order ECC, seed-constrained ECC),
horizontal bar chart with a true-rate reference line, colour-coded by
distance from the line.

Reads results/metrics/{baseline_probas_ecoli_temporal,chains_option_a_temporal,
chains_option_d_temporal,chains_option_e_temporal,ensemble_temporal,
targeted_ensemble_temporal}.csv. The true all-susceptible rate is computed
fresh from the temporal test set's ground-truth labels (cheap: labels only,
no model training).

Note: "Option F" from the original spec does not exist anywhere in this
codebase (only Option A, D, E chain orderings were ever trained) -- omitted.

Saves results/figures/r5_all_zero_fraction.{png,pdf}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_style, save_figure

METRICS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'metrics'


def true_all_zero_rate():
    from src.config import ANTIBIOTICS
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.split import temporal_split

    dataset = load_dataset()
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)
    test_idx = split['test_idx']

    Y_test = np.column_stack([dataset.to_numpy(a)[test_idx] for a in ANTIBIOTICS])
    return (Y_test.sum(axis=1) == 0).mean()


def baseline_all_zero_frac():
    proba_df = pd.read_csv(METRICS_DIR / 'baseline_probas_ecoli_temporal.csv')
    pred = (proba_df >= 0.5).astype(int)
    return (pred.sum(axis=1) == 0).mean()


def chain_all_zero_frac(filename):
    df = pd.read_csv(METRICS_DIR / filename)
    return df['all_zero_frac'].iloc[0]


def main():
    apply_style()

    print("computing true all-susceptible rate on temporal test set...")
    true_rate = true_all_zero_rate()
    print(f"true_rate={true_rate:.4f}")

    configs = [
        ('Baseline (BR)', baseline_all_zero_frac()),
        ('Chain: Option A', chain_all_zero_frac('chains_option_a_temporal.csv')),
        ('Chain: Option D', chain_all_zero_frac('chains_option_d_temporal.csv')),
        ('Chain: Option E', chain_all_zero_frac('chains_option_e_temporal.csv')),
        ('Random-order ECC', chain_all_zero_frac('ensemble_temporal.csv')),
        ('Seed-constrained ECC', chain_all_zero_frac('targeted_ensemble_temporal.csv')),
    ]

    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    distances = [abs(v - true_rate) for v in values]

    cmap = plt.cm.get_cmap('RdYlGn_r')
    max_dist = max(distances) if max(distances) > 0 else 1
    colors = [cmap(d / max_dist) for d in distances]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, zorder=3)
    ax.axvline(true_rate, color='black', linestyle='--', linewidth=1.5, zorder=4,
               label=f'true all-susceptible rate ({true_rate:.3f})')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('fraction of test isolates predicted all-susceptible')
    ax.set_title('All-zero fraction vs. true rate (E. coli, temporal split)')
    ax.legend(loc='lower right', fontsize=9)

    for y, v in zip(y_pos, values):
        ax.text(v, y, f' {v:.3f}', va='center', fontsize=9)

    fig.tight_layout()
    save_figure(fig, 'r5_all_zero_fraction')
    plt.close(fig)
    print("saved r5_all_zero_fraction")


if __name__ == '__main__':
    main()
