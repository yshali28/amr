"""
M7: S. aureus 9-antibiotic beta-lactam duplicate cluster heatmap (10x10
including Penicillin as contrast).

The 9-antibiotic cluster is identified directly from the Phase 0
systematic duplicate-cluster scan (results/metrics/duplicate_clusters.csv,
species=='saureus'): every antibiotic pair with value_agreement >= 0.99 and
missingness_agreement >= 0.97 among Piperacillin-Tazobactam, Meropenem,
Cefepime, Imipenem, Ceftriaxone, Amoxicillin-Clavulanic acid, Cefuroxime,
Oxacillin, Cefazolin -- confirmed as a fully-connected near-duplicate
cluster. Penicillin is added as the 10th column/row as a contrast (weaker
agreement with the cluster, despite being a beta-lactam itself).

Saves results/figures/m7_saureus_betalactam_cluster.{png,pdf}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_style, save_figure

METRICS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'metrics'

CLUSTER = ['Piperacillin-Tazobactam', 'Meropenem', 'Cefepime', 'Imipenem', 'Ceftriaxone',
           'Amoxicillin-Clavulanic acid', 'Cefuroxime', 'Oxacillin', 'Cefazolin']
CONTRAST = 'Penicillin'
ANTIBIOTICS = CLUSTER + [CONTRAST]


def build_matrix(df, metric):
    matrix = pd.DataFrame(np.nan, index=ANTIBIOTICS, columns=ANTIBIOTICS)
    for a in ANTIBIOTICS:
        matrix.loc[a, a] = 1.0

    for _, row in df.iterrows():
        a, b = row['antibiotic_a'], row['antibiotic_b']
        if a in ANTIBIOTICS and b in ANTIBIOTICS:
            matrix.loc[a, b] = row[metric]
            matrix.loc[b, a] = row[metric]
    return matrix


def main():
    apply_style()

    df = pd.read_csv(METRICS_DIR / 'duplicate_clusters.csv')
    df = df[df['species'] == 'saureus']

    value_matrix = build_matrix(df, 'value_agreement')

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(value_matrix.values, vmin=0.9, vmax=1.0, cmap='viridis')

    ticks = range(len(ANTIBIOTICS))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ANTIBIOTICS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ANTIBIOTICS, fontsize=8)

    # outline the 9-antibiotic cluster vs. the Penicillin contrast row/column
    n_cluster = len(CLUSTER)
    ax.axhline(n_cluster - 0.5, color='red', linewidth=2)
    ax.axvline(n_cluster - 0.5, color='red', linewidth=2)

    for i in range(len(ANTIBIOTICS)):
        for j in range(len(ANTIBIOTICS)):
            val = value_matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                         color='white' if val < 0.96 else 'black', fontsize=7)

    fig.colorbar(im, ax=ax, label='value agreement')
    ax.set_title('S. aureus beta-lactam duplicate cluster (9 antibiotics) + Penicillin contrast')

    fig.tight_layout()
    save_figure(fig, 'm7_saureus_betalactam_cluster')
    plt.close(fig)
    print("saved m7_saureus_betalactam_cluster")
    print("\nPenicillin vs. cluster agreement:")
    print(value_matrix.loc[CONTRAST, CLUSTER])


if __name__ == '__main__':
    main()
