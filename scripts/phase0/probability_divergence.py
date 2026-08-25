"""
Phase 0: probability-space divergence by chain position -- reconstructs
previously-validated methodology from a prior working session (not new
experimental design), per exact spec:

  For a fitted ClassifierChain (Option A ordering), at each step compute the
  chain estimator's predicted probability vs. an independently-trained
  baseline model's predicted probability (same antibiotic, same test set),
  take mean absolute difference. Do this for both E. coli and
  K. pneumoniae, Option A ordering, temporal split.

Saves results/metrics/probability_divergence.csv with columns:
[species, chain_position, antibiotic, mean_abs_proba_diff].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.data.features import to_feature_matrix
from src.data.load import load_metadata_with_years
from src.data.split import temporal_split


def run_for_species(label, dataset, X, order, random_seed, metadata_df):
    split = temporal_split(dataset, metadata_df)
    train_idx, test_idx = split['train_idx'], split['test_idx']

    Y = np.column_stack([dataset.to_numpy(antibiotic) for antibiotic in order])
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train = Y[train_idx]

    print(f"[probability_divergence] {label}: fitting Option A chain...")
    chain = ClassifierChain(
        estimator=XGBClassifier(random_state=random_seed, eval_metric='logloss'),
        order=list(range(len(order))),
        random_state=random_seed,
    )
    chain.fit(X_train, Y_train)
    chain_proba = chain.predict_proba(X_test)

    rows = []
    for i, antibiotic in enumerate(order):
        print(f"[probability_divergence] {label}/{antibiotic}: training independent baseline...")
        baseline_model = XGBClassifier(random_state=random_seed, eval_metric='logloss')
        baseline_model.fit(X_train, Y_train[:, i])
        baseline_proba = baseline_model.predict_proba(X_test)[:, 1]

        mean_abs_diff = np.abs(chain_proba[:, i] - baseline_proba).mean()
        print(f"[probability_divergence] {label}/{antibiotic}: mean_abs_proba_diff={mean_abs_diff:.4f}")

        rows.append({
            'species': label,
            'chain_position': i,
            'antibiotic': antibiotic,
            'mean_abs_proba_diff': mean_abs_diff,
        })

    return rows


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_df = load_metadata_with_years()

    all_rows = []

    from src.config import RANDOM_SEED as ECOLI_SEED
    from src.data.load import load_dataset

    ecoli_order = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
                   'Ciprofloxacin', 'Cotrimoxazole']
    print("[probability_divergence] ecoli: loading dataset...")
    ecoli_dataset = load_dataset()
    ecoli_X = to_feature_matrix(ecoli_dataset)
    all_rows += run_for_species('ecoli', ecoli_dataset, ecoli_X, ecoli_order, ECOLI_SEED, metadata_df)

    from maldi_learn.driams import load_driams_dataset

    from src.config_kpneumoniae import ANTIBIOTICS as KP_ANTIBIOTICS
    from src.config_kpneumoniae import DATA_ROOT as KP_DATA_ROOT
    from src.config_kpneumoniae import OPTION_A_KPNEUMONIAE, RANDOM_SEED as KP_SEED
    from src.config_kpneumoniae import SPECIES as KP_SPECIES

    print("[probability_divergence] kpneumoniae: loading dataset...")
    kp_dataset = load_driams_dataset(
        root=str(KP_DATA_ROOT.parent),
        site='DRIAMS-A',
        years=['2015', '2016', '2017', '2018'],
        species=KP_SPECIES,
        antibiotics=KP_ANTIBIOTICS,
        handle_missing_resistance_measurements='remove_if_any_missing',
        spectra_type='binned_6000',
    )
    kp_X = to_feature_matrix(kp_dataset)
    all_rows += run_for_species('kpneumoniae', kp_dataset, kp_X, OPTION_A_KPNEUMONIAE, KP_SEED, metadata_df)

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(METRICS_DIR / 'probability_divergence.csv', index=False)
    print(result_df)


if __name__ == '__main__':
    main()
