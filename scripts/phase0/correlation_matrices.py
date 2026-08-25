"""
Phase 0: label correlation matrices for each species' final modeling panel,
computed on the spectrum-validated cohort (via load_driams_dataset), matching
the methodology already used to produce the mean |corr| ranges documented in
each species' config file.

Saves results/metrics/correlation_matrix_{species}.csv (one per species).
"""

import sys
from pathlib import Path

import numpy as np
from maldi_learn.driams import load_driams_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
    ('paeruginosa', 'src.config_paeruginosa'),
]


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS'])
        DATA_ROOT, SPECIES, ANTIBIOTICS = module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS

        print(f"[correlation_matrices] {label}: loading dataset...")
        dataset = load_driams_dataset(
            root=str(DATA_ROOT.parent),
            site='DRIAMS-A',
            years=['2015', '2016', '2017', '2018'],
            species=SPECIES,
            antibiotics=ANTIBIOTICS,
            handle_missing_resistance_measurements='remove_if_any_missing',
            spectra_type='binned_6000',
        )
        print(f"[correlation_matrices] {label}: n_samples={dataset.n_samples}")

        corr = dataset.y[ANTIBIOTICS].corr()
        corr.to_csv(METRICS_DIR / f'correlation_matrix_{label}.csv')

        off_diag = corr.values[~np.eye(len(ANTIBIOTICS), dtype=bool)]
        print(f"[correlation_matrices] {label}: mean |corr| off-diag = {abs(off_diag).mean():.4f}, "
              f"range = [{abs(off_diag).min():.4f}, {abs(off_diag).max():.4f}]")


if __name__ == '__main__':
    main()
