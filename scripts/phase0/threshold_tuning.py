"""
Phase 0: threshold-tuning demonstration -- reconstructs previously-validated
methodology from a prior working session (not new experimental design), per
exact spec:

  train=2015-2016, tune=2017 (holdout), test=2018. Fit XGBoost on train only.
  F1-tuning: threshold on 2017 that maximizes F1.
  Calibration-tuning: threshold on 2017 whose predicted-positive rate most
  closely matches the observed positive rate on 2017.
  Apply both thresholds to 2018 test predictions. Report AUROC, F1,
  precision, recall, true_rate, pred_rate for both methods, per antibiotic.

Run for E. coli, S. aureus, K. pneumoniae.
Saves results/metrics/threshold_tuning_{species}.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from maldi_learn.driams import load_driams_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.data.features import to_feature_matrix
from src.data.load import load_metadata_with_years
from src.data.split import _attach_year

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
]


def tune_threshold_f1(y_true, y_proba):
    thresholds = np.unique(y_proba)
    best_threshold, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold


def tune_threshold_calibration(y_true, y_proba):
    observed_rate = y_true.mean()
    thresholds = np.unique(y_proba)
    best_threshold, best_gap = 0.5, np.inf
    for t in thresholds:
        pred_rate = (y_proba >= t).mean()
        gap = abs(pred_rate - observed_rate)
        if gap < best_gap:
            best_gap, best_threshold = gap, t
    return best_threshold


def evaluate(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'auroc': roc_auc_score(y_true, y_proba),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'true_rate': y_true.mean(),
        'pred_rate': y_pred.mean(),
    }


def run_for_species(label, module_name):
    module = __import__(module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS'])
    DATA_ROOT, SPECIES, ANTIBIOTICS = module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS

    print(f"[threshold_tuning] {label}: loading dataset...")
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

    merged = _attach_year(dataset, metadata_df)
    years = merged['year_folder'].astype(int)
    train_idx = merged.loc[years.isin([2015, 2016]), '_pos'].to_numpy()
    tune_idx = merged.loc[years == 2017, '_pos'].to_numpy()
    test_idx = merged.loc[years == 2018, '_pos'].to_numpy()
    print(f"[threshold_tuning] {label}: train={len(train_idx)}, tune={len(tune_idx)}, test={len(test_idx)}")

    X_train, X_tune, X_test = X[train_idx], X[tune_idx], X[test_idx]

    rows = []
    for antibiotic in ANTIBIOTICS:
        y = dataset.to_numpy(antibiotic)
        y_train, y_tune, y_test = y[train_idx], y[tune_idx], y[test_idx]

        model = XGBClassifier(random_state=module.RANDOM_SEED, eval_metric='logloss')
        model.fit(X_train, y_train)

        proba_tune = model.predict_proba(X_tune)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]

        f1_threshold = tune_threshold_f1(y_tune, proba_tune)
        calib_threshold = tune_threshold_calibration(y_tune, proba_tune)

        naive_result = evaluate(y_test, proba_test, f1_threshold)
        calib_result = evaluate(y_test, proba_test, calib_threshold)
        baseline_result = evaluate(y_test, proba_test, 0.5)

        print(f"[threshold_tuning] {label}/{antibiotic}: "
              f"F1@0.5={baseline_result['f1']:.4f}, F1@f1-tuned={naive_result['f1']:.4f} (t={f1_threshold:.3f}), "
              f"F1@calib-tuned={calib_result['f1']:.4f} (t={calib_threshold:.3f})")

        rows.append({
            'antibiotic': antibiotic, 'method': 'naive_f1', 'threshold': f1_threshold,
            'auroc': naive_result['auroc'], 'f1_at_0.5': baseline_result['f1'], 'f1_at_tuned': naive_result['f1'],
            'precision': naive_result['precision'], 'recall': naive_result['recall'],
            'true_rate': naive_result['true_rate'], 'pred_rate': naive_result['pred_rate'],
        })
        rows.append({
            'antibiotic': antibiotic, 'method': 'calibration', 'threshold': calib_threshold,
            'auroc': calib_result['auroc'], 'f1_at_0.5': baseline_result['f1'], 'f1_at_tuned': calib_result['f1'],
            'precision': calib_result['precision'], 'recall': calib_result['recall'],
            'true_rate': calib_result['true_rate'], 'pred_rate': calib_result['pred_rate'],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / f'threshold_tuning_{label}.csv', index=False)
    return result_df


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for label, module_name in SPECIES_CONFIGS:
        run_for_species(label, module_name)


if __name__ == '__main__':
    main()
