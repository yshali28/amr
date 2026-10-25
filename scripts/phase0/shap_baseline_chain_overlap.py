"""
Phase 0: top-15 SHAP feature overlap between the Binary Relevance baseline and
the Option A chain, E. coli, all 5 antibiotics -- read from the already-saved
results/metrics/shap_{baseline,chain_option_a}_temporal_full.npz (no retraining).

This is the comparison the write-up describes but that
scripts/phase0/shap_feature_overlap.py does not compute: that script compares
Option A against Option G (chain vs. chain, the positional control). Both are
useful, they answer different questions, and they are kept separate.

Only the first N_SPECTRAL columns are compared. A chain model at position k has
the 6000 spectral features plus k prior-prediction features appended, so the
arrays have different widths and the extra columns are not spectral bins.

Saves results/metrics/shap_baseline_chain_overlap.csv with columns:
[antibiotic, chain_position, top15_overlap_count, top15_overlap_fraction].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR

TOP_K = 15
N_SPECTRAL = 6000

# Option A ordering, so we can report which chain position each antibiotic sat at.
OPTION_A = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
            'Ciprofloxacin', 'Cotrimoxazole']


def top_k_spectral_indices(arr, k=TOP_K):
    """Indices of the k largest values, restricted to the spectral columns."""
    spectral = arr[:N_SPECTRAL]
    return set(np.argsort(spectral)[::-1][:k].tolist())


def main():
    baseline = np.load(METRICS_DIR / 'shap_baseline_temporal_full.npz')
    chain = np.load(METRICS_DIR / 'shap_chain_option_a_temporal_full.npz')

    antibiotics = sorted(set(baseline.files) & set(chain.files))

    rows = []
    for antibiotic in antibiotics:
        top_baseline = top_k_spectral_indices(baseline[antibiotic])
        top_chain = top_k_spectral_indices(chain[antibiotic])

        overlap = top_baseline & top_chain
        rows.append({
            'antibiotic': antibiotic,
            'chain_position': OPTION_A.index(antibiotic) if antibiotic in OPTION_A else None,
            'top15_overlap_count': len(overlap),
            'top15_overlap_fraction': len(overlap) / TOP_K,
        })

    result_df = pd.DataFrame(rows).sort_values('chain_position')
    result_df.to_csv(METRICS_DIR / 'shap_baseline_chain_overlap.csv', index=False)
    print(result_df.to_string(index=False))

    # Position 0 is the chain seed: its model sees no prior predictions, so it is
    # the same model as the baseline and should come out at 15/15. If it does not,
    # something differs between the two runs and the comparison is not clean.
    seed_row = result_df[result_df['chain_position'] == 0]
    if not seed_row.empty:
        seed_overlap = seed_row['top15_overlap_count'].iloc[0]
        print(f"\nsanity check -- chain seed overlap with baseline: {seed_overlap}/15")
        if seed_overlap != TOP_K:
            print("WARNING: expected 15/15 for the seed antibiotic.")


if __name__ == '__main__':
    main()
