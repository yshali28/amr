"""
Phase 0: raw predicted-probability arrays from the Binary Relevance baseline
(temporal split, test set), for the boundary-proximity check -- E. coli and
K. pneumoniae.

Saves results/metrics/baseline_probas_{species}_temporal.csv, one column
per antibiotic, one row per test-set isolate.
"""

import sys
from pathlib import Path

import pandas as pd
from maldi_learn.driams import load_driams_dataset
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.data.features import to_feature_matrix
from src.data.load import load_metadata_with_years
from src.data.split import temporal_split

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
]


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_df = load_metadata_with_years()

    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS', 'RANDOM_SEED'])
        DATA_ROOT, SPECIES, ANTIBIOTICS, RANDOM_SEED = (
            module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS, module.RANDOM_SEED
        )

        print(f"[boundary_probas] {label}: loading dataset...")
        dataset = load_driams_dataset(
            root=str(DATA_ROOT.parent),
            site='DRIAMS-A',
            years=['2015', '2016', '2017', '2018'],
            species=SPECIES,
            antibiotics=ANTIBIOTICS,
            handle_missing_resistance_measurements='remove_if_any_missing',
            spectra_type='binned_6000',
        )
        X = to_feature_matrix(dataset)
        split = temporal_split(dataset, metadata_df)
        train_idx, test_idx = split['train_idx'], split['test_idx']
        X_train, X_test = X[train_idx], X[test_idx]

        proba_cols = {}
        for antibiotic in ANTIBIOTICS:
            print(f"[boundary_probas] {label}/{antibiotic}: training...")
            y = dataset.to_numpy(antibiotic)
            model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
            model.fit(X_train, y[train_idx])
            proba_cols[antibiotic] = model.predict_proba(X_test)[:, 1]

        proba_df = pd.DataFrame(proba_cols)
        proba_df.to_csv(METRICS_DIR / f'baseline_probas_{label}_temporal.csv', index=False)
        print(f"[boundary_probas] {label}: saved {proba_df.shape}")


if __name__ == '__main__':
    main()
