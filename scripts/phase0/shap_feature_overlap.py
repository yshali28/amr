"""
Phase 0: top-15 SHAP feature overlap, Option A vs Option G, E. coli, all 5
antibiotics -- extracted from the already-saved
results/metrics/shap_chain_option_{a,g}_temporal_full.npz (no recomputation).

Saves results/metrics/shap_feature_overlap.csv with columns:
[antibiotic, top15_overlap_count, top15_overlap_fraction].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR

TOP_K = 15


def top_k_indices(arr, k=TOP_K):
    return set(np.argsort(arr)[::-1][:k].tolist())


def main():
    option_a = np.load(METRICS_DIR / 'shap_chain_option_a_temporal_full.npz')
    option_g = np.load(METRICS_DIR / 'shap_chain_option_g_temporal_full.npz')

    antibiotics = sorted(set(option_a.files) & set(option_g.files))

    rows = []
    for antibiotic in antibiotics:
        top_a = top_k_indices(option_a[antibiotic])
        top_g = top_k_indices(option_g[antibiotic])

        overlap = top_a & top_g
        rows.append({
            'antibiotic': antibiotic,
            'top15_overlap_count': len(overlap),
            'top15_overlap_fraction': len(overlap) / TOP_K,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'shap_feature_overlap.csv', index=False)
    print(result_df)


if __name__ == '__main__':
    main()
