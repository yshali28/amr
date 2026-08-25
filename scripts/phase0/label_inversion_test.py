"""
Phase 0: label-inversion test -- reconstructs previously-validated
methodology from a prior working session (not new experimental design), per
exact spec:

  Using Option E's ordering (Ciprofloxacin seeded first), invert
  Ciprofloxacin's binary labels (flip 0<->1) before building the chain, keep
  all other antibiotics' labels unchanged. Retrain the chain on this
  modified label set, evaluate all_zero_frac and Jaccard on the temporal
  test set (against the same modified label set, since that is what the
  chain was trained to predict).

Saves results/metrics/label_inversion_test.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.config import RANDOM_SEED
from src.data.features import to_feature_matrix
from src.data.load import load_dataset, load_metadata_with_years
from src.data.split import temporal_split
from src.evaluation.metrics import multilabel_metrics

OPTION_E = ['Ciprofloxacin', 'Ceftriaxone', 'Ampicillin-Amoxicillin', 'Cotrimoxazole', 'Amoxicillin-Clavulanic acid']


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[label_inversion_test] loading dataset...")
    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)
    train_idx, test_idx = split['train_idx'], split['test_idx']

    Y = np.column_stack([dataset.to_numpy(antibiotic) for antibiotic in OPTION_E])
    Y[:, 0] = 1 - Y[:, 0]  # invert Ciprofloxacin (seed position)

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print("[label_inversion_test] fitting chain on inverted-seed labels...")
    chain = ClassifierChain(
        estimator=XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'),
        order=list(range(len(OPTION_E))),
        random_state=RANDOM_SEED,
    )
    chain.fit(X_train, Y_train)

    Y_pred = chain.predict(X_test)

    all_zero_frac = (Y_pred.sum(axis=1) == 0).mean()
    joint_metrics = multilabel_metrics(Y_test, Y_pred)

    print(f"[label_inversion_test] all_zero_frac={all_zero_frac:.4f}, "
          f"jaccard_score={joint_metrics['jaccard_score']:.4f}")

    result_df = pd.DataFrame([{
        'order_name': 'option_e_ciprofloxacin_inverted',
        'seed_antibiotic': 'Ciprofloxacin',
        'seed_label_inverted': True,
        'all_zero_frac': all_zero_frac,
        'jaccard_score': joint_metrics['jaccard_score'],
        'hamming_loss': joint_metrics['hamming_loss'],
        'exact_match_ratio': joint_metrics['exact_match_ratio'],
    }])
    result_df.to_csv(METRICS_DIR / 'label_inversion_test.csv', index=False)


if __name__ == '__main__':
    main()
