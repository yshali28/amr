"""
R26: P. aeruginosa selection effect -- full-cohort vs. complete-case
resistance rate per antibiotic.

Reads results/metrics/selection_effect_paeruginosa.csv.
Saves results/figures/r26_paeruginosa_selection_effect.{png,pdf}.
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

    df = pd.read_csv(METRICS_DIR / 'selection_effect_paeruginosa.csv')

    y_pos = np.arange(len(df))
    height = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y_pos - height / 2, df['full_cohort_resistance_rate'], height=height,
            label='full cohort', color='#6a3d9a', alpha=0.85)
    ax.barh(y_pos + height / 2, df['complete_case_resistance_rate'], height=height,
            label='complete-case panel', color='#c1440e', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['antibiotic'])
    ax.invert_yaxis()
    ax.set_xlabel('resistance rate')
    ax.set_title('P. aeruginosa: selection effect (full cohort vs. complete-case panel)')
    ax.legend(loc='lower right', fontsize=9)

    for y, (full, cc) in enumerate(zip(df['full_cohort_resistance_rate'], df['complete_case_resistance_rate'])):
        ax.text(full, y - height / 2, f' {full:.3f}', va='center', fontsize=8)
        ax.text(cc, y + height / 2, f' {cc:.3f}', va='center', fontsize=8)

    fig.tight_layout()
    save_figure(fig, 'r26_paeruginosa_selection_effect')
    plt.close(fig)
    print("saved r26_paeruginosa_selection_effect")


if __name__ == '__main__':
    main()
