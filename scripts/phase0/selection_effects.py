"""
Phase 0: selection-effect check -- per-antibiotic resistance rate in the full
species cohort (every isolate with a non-missing label for that antibiotic,
metadata-level) vs. the resistance rate within the final complete-case
modeling panel.

The complete-case side uses the spectrum-validated cohort (via
load_driams_dataset), matching the documented convention in every species'
config.py ("use the spectrum-validated count in all reported results") --
NOT the metadata-only complete-case mask, which admits isolates lacking an
actual matched spectrum and understates the true selection effect.

Saves results/metrics/selection_effect_{species}.csv (one per species), with
columns: antibiotic, full_cohort_n, full_cohort_resistance_rate,
complete_case_n, complete_case_resistance_rate.
"""

import sys
from pathlib import Path

import pandas as pd
from maldi_learn.driams import load_driams_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR, collapse_label, load_raw_metadata
from src.config import ID_FOLDER, MISSING_TOKEN

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
    ('paeruginosa', 'src.config_paeruginosa'),
]


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[selection_effects] loading raw metadata (for full-cohort rates)...")
    metadata = load_raw_metadata(ID_FOLDER)

    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS'])
        DATA_ROOT, SPECIES, ANTIBIOTICS = module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS

        species_df = metadata[metadata['species'] == SPECIES]

        print(f"[selection_effects] {label}: loading spectrum-validated complete-case cohort...")
        dataset = load_driams_dataset(
            root=str(DATA_ROOT.parent),
            site='DRIAMS-A',
            years=['2015', '2016', '2017', '2018'],
            species=SPECIES,
            antibiotics=ANTIBIOTICS,
            handle_missing_resistance_measurements='remove_if_any_missing',
            spectra_type='binned_6000',
        )
        print(f"[selection_effects] {label}: n_samples={dataset.n_samples}")

        rows = []
        for antibiotic in ANTIBIOTICS:
            full_labels = species_df[antibiotic].apply(lambda v: collapse_label(v, MISSING_TOKEN)).dropna()
            complete_labels = dataset.to_numpy(antibiotic)

            rows.append({
                'antibiotic': antibiotic,
                'full_cohort_n': len(full_labels),
                'full_cohort_resistance_rate': full_labels.mean(),
                'complete_case_n': len(complete_labels),
                'complete_case_resistance_rate': complete_labels.mean(),
            })

        result_df = pd.DataFrame(rows)
        result_df['rate_diff'] = result_df['complete_case_resistance_rate'] - result_df['full_cohort_resistance_rate']
        result_df.to_csv(METRICS_DIR / f'selection_effect_{label}.csv', index=False)

        print(f"[selection_effects] {label}:")
        print(result_df)


if __name__ == '__main__':
    main()
