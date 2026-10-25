"""
Phase 0: true all-susceptible rate on the temporal test set, for all 4 species.

Every calibration claim in the write-up is read against these rates (an all-zero
prediction fraction is "well calibrated" when it is close to the true rate, not
when it is low), but until now they were recomputed inside
scripts/figures/r5_all_zero_fraction.py and never stored. This persists them.

Cheap: labels only, no model training.

Saves results/metrics/true_all_zero_rates.csv with columns:
[species, n_test, true_all_zero_rate, n_all_zero].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from maldi_learn.driams import load_driams_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.data.load import load_metadata_with_years
from src.data.split import temporal_split

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
    ('paeruginosa', 'src.config_paeruginosa'),
]


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_with_years()

    rows = []
    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS'])
        DATA_ROOT, SPECIES, ANTIBIOTICS = module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS

        print(f"[true_all_zero_rates] {label}: loading cohort...")
        dataset = load_driams_dataset(
            root=str(DATA_ROOT.parent), site='DRIAMS-A',
            years=['2015', '2016', '2017', '2018'],
            species=SPECIES, antibiotics=ANTIBIOTICS,
            handle_missing_resistance_measurements='remove_if_any_missing',
            spectra_type='binned_6000',
        )

        split = temporal_split(dataset, metadata_df)
        test_idx = split['test_idx']

        # Same accessor pattern as scripts/figures/r5_all_zero_fraction.py and
        # src/models/chains.py, both of which treat to_numpy() as returning the
        # label array directly for the given antibiotic.
        Y_test = np.column_stack([dataset.to_numpy(a)[test_idx] for a in ANTIBIOTICS])
        all_zero = (Y_test.sum(axis=1) == 0)

        rows.append({
            'species': label,
            'n_test': len(test_idx),
            'n_all_zero': int(all_zero.sum()),
            'true_all_zero_rate': all_zero.mean(),
        })
        print(f"  {label}: n_test={len(test_idx)}, true_all_zero_rate={all_zero.mean():.4f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'true_all_zero_rates.csv', index=False)
    print()
    print(result_df.to_string(index=False))


if __name__ == '__main__':
    main()
