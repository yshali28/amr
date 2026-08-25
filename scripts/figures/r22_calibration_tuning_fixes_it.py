"""
R22: Calibration-tuning fixes it -- E. coli, F1@0.5 vs. F1@calibration-tuned
per antibiotic.

Reads results/metrics/threshold_tuning_ecoli.csv (method=='calibration' rows).
Saves results/figures/r22_calibration_tuning_fixes_it.{png,pdf}.
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

    df = pd.read_csv(METRICS_DIR / 'threshold_tuning_ecoli.csv')
    calib = df[df['method'] == 'calibration'].reset_index(drop=True)

    y_pos = np.arange(len(calib))
    height = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(y_pos - height / 2, calib['f1_at_0.5'], height=height, label='F1 @ 0.5', color='#c1440e', alpha=0.85)
    ax.barh(y_pos + height / 2, calib['f1_at_tuned'], height=height, label='F1 @ calibration-tuned', color='#1b6ca8', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(calib['antibiotic'])
    ax.invert_yaxis()
    ax.set_xlabel('F1 (test set, 2018)')
    ax.set_title('E. coli: calibration-tuning recovers F1 lost to threshold degeneracy\n(train=2015-16, tune=2017, test=2018)')
    ax.legend(loc='lower right', fontsize=9)

    for y, (v05, vt) in enumerate(zip(calib['f1_at_0.5'], calib['f1_at_tuned'])):
        ax.text(v05, y - height / 2, f' {v05:.3f}', va='center', fontsize=8)
        ax.text(vt, y + height / 2, f' {vt:.3f}', va='center', fontsize=8)

    fig.tight_layout()
    save_figure(fig, 'r22_calibration_tuning_fixes_it')
    plt.close(fig)
    print("saved r22_calibration_tuning_fixes_it")


if __name__ == '__main__':
    main()
