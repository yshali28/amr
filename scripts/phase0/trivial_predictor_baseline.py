"""
Phase 0: trivial all-susceptible predictor on the temporal test set, all 4 species.

The write-up argues that Hamming loss stops discriminating on heavily skewed
panels, and supported it with "a trivial all-susceptible predictor achieves
Hamming 0.1019 against the model's 0.0933" for K. pneumoniae. That model figure
came from the superseded Meropenem panel (the final panel gives 0.130), so the
pair of numbers no longer matches and the comparison has to be recomputed on the
current panels.

This predicts 0 for every antibiotic on every test isolate and scores it with the
same multilabel_metrics() used everywhere else, so the comparison is like-for-like.

Cheap: labels only, no model training.

Saves results/metrics/trivial_predictor_baseline.csv with columns:
[species, n_test, hamming_loss, jaccard_score, jaccard_score_zd1,
 exact_match_ratio, restricted_jaccard].
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
from src.evaluation.metrics import multilabel_metrics

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
    ('paeruginosa', 'src.config_paeruginosa'),
]

# For reference in the printout: the trained Binary Relevance baseline's Hamming
# loss on the same split, from results/metrics/baseline_*_temporal.csv.
BASELINE_HAMMING = {
    'ecoli': 0.2591,
    'saureus': 0.1597,
    'kpneumoniae': 0.1300,
    'paeruginosa': 0.1070,
}


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_with_years()

    rows = []
    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS'])
        DATA_ROOT, SPECIES, ANTIBIOTICS = module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS

        print(f"[trivial_predictor] {label}: loading cohort...")
        dataset = load_driams_dataset(
            root=str(DATA_ROOT.parent), site='DRIAMS-A',
            years=['2015', '2016', '2017', '2018'],
            species=SPECIES, antibiotics=ANTIBIOTICS,
            handle_missing_resistance_measurements='remove_if_any_missing',
            spectra_type='binned_6000',
        )

        split = temporal_split(dataset, metadata_df)
        test_idx = split['test_idx']

        Y_test = np.column_stack([dataset.to_numpy(a)[test_idx] for a in ANTIBIOTICS])
        Y_trivial = np.zeros_like(Y_test)

        metrics = multilabel_metrics(Y_test, Y_trivial)
        metrics['species'] = label
        metrics['n_test'] = len(test_idx)
        rows.append(metrics)

        trained = BASELINE_HAMMING.get(label)
        print(f"  {label}: trivial Hamming={metrics['hamming_loss']:.4f}, "
              f"trained baseline Hamming={trained}")
        if trained is not None and metrics['hamming_loss'] < trained:
            print("    note: the trivial predictor BEATS the trained baseline on Hamming here")

    column_order = ['species', 'n_test', 'hamming_loss', 'jaccard_score',
                    'jaccard_score_zd1', 'exact_match_ratio', 'restricted_jaccard']
    result_df = pd.DataFrame(rows)[column_order]
    result_df.to_csv(METRICS_DIR / 'trivial_predictor_baseline.csv', index=False)
    print()
    print(result_df.to_string(index=False))


if __name__ == '__main__':
    main()
