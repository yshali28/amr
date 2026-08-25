"""
Phase 0: selection-effect check -- per-antibiotic resistance rate in the full
species cohort (every isolate with a non-missing label for that antibiotic)
vs. the resistance rate within the final complete-case modeling panel.

Saves results/metrics/selection_effect_{species}.csv (one per species), with
columns: antibiotic, full_cohort_n, full_cohort_resistance_rate,
complete_case_n, complete_case_resistance_rate.
"""

import sys
from pathlib import Path

import pandas as pd

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

    print("[selection_effects] loading raw metadata...")
    metadata = load_raw_metadata(ID_FOLDER)

    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['SPECIES', 'ANTIBIOTICS'])
        SPECIES, ANTIBIOTICS = module.SPECIES, module.ANTIBIOTICS

        species_df = metadata[metadata['species'] == SPECIES]

        complete_mask = pd.Series(True, index=species_df.index)
        for antibiotic in ANTIBIOTICS:
            column = species_df[antibiotic]
            complete_mask &= column.notna() & (column != MISSING_TOKEN)
        complete_df = species_df[complete_mask]

        rows = []
        for antibiotic in ANTIBIOTICS:
            full_labels = species_df[antibiotic].apply(lambda v: collapse_label(v, MISSING_TOKEN)).dropna()
            complete_labels = complete_df[antibiotic].apply(lambda v: collapse_label(v, MISSING_TOKEN)).dropna()

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
