"""
Phase 0: combined SHAP prior-attribution summary.

E. coli (Option A, both splits): extracted directly from the already-saved
results/metrics/shap_chain_option_a_{temporal,random}.csv (no recomputation).

K. pneumoniae (Option A, chain step 1 only): freshly computed -- fits the
Option A chain (temporal split), then runs shap.TreeExplainer on just the
step-1 estimator's augmented input (spectral features + step 0's own
predictions on the same subsample), matching shap_analysis.py's shap_chain
methodology but restricted to a single step to keep this bounded.

Saves results/metrics/shap_prior_attribution.csv with columns:
[species, order_name, split, antibiotic, chain_position, fraction_prior_attribution].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.config_kpneumoniae import ANTIBIOTICS, DATA_ROOT, OPTION_A_KPNEUMONIAE, RANDOM_SEED, SPECIES
from src.data.features import to_feature_matrix
from src.data.load import load_metadata_with_years
from src.data.split import temporal_split

N_SPECTRAL_FEATURES = 6000
SUBSAMPLE_N = 750


def ecoli_rows_from_existing():
    rows = []
    for split_name in ['temporal', 'random']:
        df = pd.read_csv(METRICS_DIR / f'shap_chain_option_a_{split_name}.csv')
        for _, row in df.iterrows():
            rows.append({
                'species': 'ecoli',
                'order_name': row['order_name'],
                'split': row['split'],
                'antibiotic': row['antibiotic'],
                'chain_position': row['chain_position'],
                'fraction_prior_attribution': row['fraction_prior_attribution'],
            })
    return rows


def kpneumoniae_step1_row():
    from maldi_learn.driams import load_driams_dataset

    print("[shap_prior_attribution] kpneumoniae: loading dataset...")
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
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)
    train_idx, test_idx = split['train_idx'], split['test_idx']

    order = OPTION_A_KPNEUMONIAE
    Y = np.column_stack([dataset.to_numpy(antibiotic) for antibiotic in order])
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train = Y[train_idx]

    print("[shap_prior_attribution] kpneumoniae: fitting Option A chain...")
    chain = ClassifierChain(
        estimator=XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'),
        order=list(range(len(order))),
        random_state=RANDOM_SEED,
    )
    chain.fit(X_train, Y_train)

    rng = np.random.RandomState(RANDOM_SEED)
    if len(X_test) > SUBSAMPLE_N:
        idx = rng.choice(len(X_test), SUBSAMPLE_N, replace=False)
    else:
        idx = np.arange(len(X_test))
    X_sub = X_test[idx]

    print("[shap_prior_attribution] kpneumoniae: step 0 predictions for augmentation...")
    step0_pred = chain.estimators_[0].predict(X_sub).reshape(-1, 1)
    step1_input = np.hstack([X_sub, step0_pred])

    print(f"[shap_prior_attribution] kpneumoniae: running TreeExplainer on step 1 ({order[1]}), shape={step1_input.shape}...")
    explainer = shap.TreeExplainer(chain.estimators_[1])
    shap_values = explainer.shap_values(step1_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    prior_shap = shap_values[:, N_SPECTRAL_FEATURES:]
    fraction_prior = np.abs(prior_shap).sum() / np.abs(shap_values).sum()

    print(f"[shap_prior_attribution] kpneumoniae step 1 ({order[1]}): fraction_prior_attribution={fraction_prior:.4f}")

    return {
        'species': 'kpneumoniae',
        'order_name': 'option_a',
        'split': 'temporal',
        'antibiotic': order[1],
        'chain_position': 1,
        'fraction_prior_attribution': fraction_prior,
    }


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = ecoli_rows_from_existing()
    rows.append(kpneumoniae_step1_row())

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'shap_prior_attribution.csv', index=False)
    print(result_df)


if __name__ == '__main__':
    main()
