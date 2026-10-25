"""
Phase 0: covariate-shift test -- can collection year be predicted from the
spectrum alone? All 4 species.

Why this matters. Performance degrades consistently under temporal validation and
the write-up currently reports the cause as unresolved: resistance prevalence was
checked across years and found stable, so class-balance shift is ruled out, but
nothing has tested whether the SPECTRA themselves shifted.

This trains a classifier to separate 2018 spectra from 2015-2017 spectra using no
labels at all. If it performs clearly above chance, the input distribution moved
between training and test periods, which is a demonstrated cause for the
degradation rather than an absence of one. If it sits near 0.5, covariate shift is
ruled out too and the cause remains open -- which is also a reportable result, and
a cleaner limitation than not having looked.

Scored by 5-fold stratified cross-validation AUROC over the whole cohort, so the
figure does not depend on an arbitrary split of the shift-detection task itself.

Cost: one XGBoost fit per fold per species.

Saves results/metrics/covariate_shift_test.csv with columns:
[species, n_total, n_train_years, n_test_year, cv_auroc_mean, cv_auroc_std].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from maldi_learn.driams import load_driams_dataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.config import RANDOM_SEED
from src.data.features import to_feature_matrix
from src.data.load import load_metadata_with_years
from src.data.split import temporal_split

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
    ('paeruginosa', 'src.config_paeruginosa'),
]

N_FOLDS = 5


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_with_years()

    rows = []
    for label, module_name in SPECIES_CONFIGS:
        module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS'])
        DATA_ROOT, SPECIES, ANTIBIOTICS = module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS

        print(f"[covariate_shift] {label}: loading cohort...")
        dataset = load_driams_dataset(
            root=str(DATA_ROOT.parent), site='DRIAMS-A',
            years=['2015', '2016', '2017', '2018'],
            species=SPECIES, antibiotics=ANTIBIOTICS,
            handle_missing_resistance_measurements='remove_if_any_missing',
            spectra_type='binned_6000',
        )

        X = to_feature_matrix(dataset)
        split = temporal_split(dataset, metadata_df)

        # The target here is the PERIOD an isolate came from, not its resistance.
        is_test_year = np.zeros(len(X), dtype=int)
        is_test_year[split['test_idx']] = 1

        n_train_years = int((is_test_year == 0).sum())
        n_test_year = int((is_test_year == 1).sum())
        print(f"  2015-2017: {n_train_years}, 2018: {n_test_year}")

        model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X, is_test_year, cv=cv, scoring='roc_auc')

        rows.append({
            'species': label,
            'n_total': len(X),
            'n_train_years': n_train_years,
            'n_test_year': n_test_year,
            'cv_auroc_mean': scores.mean(),
            'cv_auroc_std': scores.std(),
        })
        print(f"  year-prediction AUROC: {scores.mean():.4f} +/- {scores.std():.4f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'covariate_shift_test.csv', index=False)
    print()
    print(result_df.to_string(index=False))

    print("\nReading the result:")
    print("  ~0.5          -> no detectable spectral shift; cause stays unresolved")
    print("  0.6 to 0.8    -> moderate shift, a plausible contributor")
    print("  >0.8          -> strong shift; the spectra clearly moved between periods")


if __name__ == '__main__':
    main()
