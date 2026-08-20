"""
SHAP interpretability for the Binary Relevance baseline and Classifier Chains.

Example:
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.features import to_feature_matrix
    from src.data.split import temporal_split
    from src.interpretability.shap_analysis import shap_baseline, shap_chain

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    baseline_results = shap_baseline(X, dataset, split, 'temporal')

    OPTION_A = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
                'Ciprofloxacin', 'Cotrimoxazole']
    chain_results = shap_chain(X, dataset, split, 'temporal', OPTION_A, 'option_a')
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

from src.config import ANTIBIOTICS, RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_SPECTRAL_FEATURES = 6000
TOP_K = 15


def _get_shap_values(explainer, X_sub):
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    return shap_values


def _subsample(X_test, subsample_n):
    rng = np.random.RandomState(RANDOM_SEED)
    if len(X_test) > subsample_n:
        idx = rng.choice(len(X_test), subsample_n, replace=False)
    else:
        idx = np.arange(len(X_test))
    return X_test[idx]


def shap_baseline(X, dataset, split, split_name, subsample_n=750):
    """
    Compute per-antibiotic SHAP feature importances for the Binary
    Relevance baseline (one XGBClassifier per antibiotic).

    Example:
        results = shap_baseline(X, dataset, split, 'temporal')
        results['Ciprofloxacin']['top15_feature_indices']
    """
    train_idx = split['train_idx']
    test_idx = split['test_idx']

    X_train, X_test = X[train_idx], X[test_idx]
    X_sub = _subsample(X_test, subsample_n)

    results = {}
    full_arrays = {}
    rows = []

    for antibiotic in ANTIBIOTICS:
        print(f"[shap_baseline] {antibiotic}: training model...")
        y = dataset.to_numpy(antibiotic)
        y_train = y[train_idx]

        model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
        model.fit(X_train, y_train)

        print(f"[shap_baseline] {antibiotic}: running TreeExplainer on {len(X_sub)} rows...")
        explainer = shap.TreeExplainer(model)
        shap_values = _get_shap_values(explainer, X_sub)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top15_idx = np.argsort(mean_abs_shap)[::-1][:TOP_K]
        top15_val = mean_abs_shap[top15_idx]

        print(f"[shap_baseline] {antibiotic}: done, top feature idx={top15_idx[0]} (mean|SHAP|={top15_val[0]:.5f})")

        results[antibiotic] = {
            'antibiotic': antibiotic,
            'top15_feature_indices': top15_idx,
            'top15_mean_abs_shap': top15_val,
            'mean_abs_shap_all_features': mean_abs_shap,
        }
        full_arrays[antibiotic] = mean_abs_shap

        row = {'antibiotic': antibiotic, 'split': split_name}
        for k in range(TOP_K):
            row[f'top15_idx_{k}'] = top15_idx[k]
            row[f'top15_val_{k}'] = top15_val[k]
        rows.append(row)

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / f'shap_baseline_{split_name}.csv', index=False)
    np.savez(output_dir / f'shap_baseline_{split_name}_full.npz', **full_arrays)

    return results


def shap_chain(X, dataset, split, split_name, order, order_name, subsample_n=750):
    """
    Compute per-step SHAP feature importances for a Classifier Chain,
    including how much attribution comes from prior-antibiotic predictions
    versus the raw spectrum.

    Example:
        OPTION_A = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
                    'Ciprofloxacin', 'Cotrimoxazole']
        results = shap_chain(X, dataset, split, 'temporal', OPTION_A, 'option_a')
        results['Ceftriaxone']['fraction_prior_attribution']
    """
    train_idx = split['train_idx']
    test_idx = split['test_idx']

    Y = np.column_stack([dataset.to_numpy(antibiotic) for antibiotic in order])

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train = Y[train_idx]

    print(f"[shap_chain:{order_name}] fitting chain...")
    chain = ClassifierChain(
        estimator=XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'),
        order=list(range(len(order))),
        random_state=RANDOM_SEED,
    )
    chain.fit(X_train, Y_train)

    X_sub = _subsample(X_test, subsample_n)

    results = {}
    full_arrays = {}
    rows = []
    prior_pred_columns = []

    for i, antibiotic in enumerate(order):
        print(f"[shap_chain:{order_name}] step {i} ({antibiotic}): building augmented input...")
        if prior_pred_columns:
            aug_sub = np.hstack([X_sub] + prior_pred_columns)
        else:
            aug_sub = X_sub

        print(f"[shap_chain:{order_name}] step {i} ({antibiotic}): running TreeExplainer on {aug_sub.shape}...")
        explainer = shap.TreeExplainer(chain.estimators_[i])
        shap_values = _get_shap_values(explainer, aug_sub)

        spectral_shap = shap_values[:, :N_SPECTRAL_FEATURES]
        prior_shap = shap_values[:, N_SPECTRAL_FEATURES:]

        if i == 0:
            fraction_prior = 0.0
        else:
            fraction_prior = np.abs(prior_shap).sum() / np.abs(shap_values).sum()

        print(f"[shap_chain:{order_name}] step {i} ({antibiotic}): fraction_prior_attribution={fraction_prior:.4f}")

        mean_abs_shap_spectral = np.abs(spectral_shap).mean(axis=0)
        top15_idx = np.argsort(mean_abs_shap_spectral)[::-1][:TOP_K]
        top15_val = mean_abs_shap_spectral[top15_idx]

        results[antibiotic] = {
            'antibiotic': antibiotic,
            'chain_position': i,
            'fraction_prior_attribution': fraction_prior,
            'top15_spectral_feature_indices': top15_idx,
            'top15_spectral_mean_abs_shap': top15_val,
            'mean_abs_shap_spectral_features': mean_abs_shap_spectral,
        }
        full_arrays[antibiotic] = mean_abs_shap_spectral

        rows.append({
            'antibiotic': antibiotic,
            'chain_position': i,
            'fraction_prior_attribution': fraction_prior,
            'split': split_name,
            'order_name': order_name,
        })

        # Build this step's own binary predictions on the augmented subsample,
        # for use as an input column at subsequent chain steps.
        pred_i = chain.estimators_[i].predict(aug_sub)
        prior_pred_columns.append(pred_i.reshape(-1, 1))

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows, columns=['antibiotic', 'chain_position', 'fraction_prior_attribution', 'split', 'order_name'])
    summary_df.to_csv(output_dir / f'shap_chain_{order_name}_{split_name}.csv', index=False)
    np.savez(output_dir / f'shap_chain_{order_name}_{split_name}_full.npz', **full_arrays)

    return results


if __name__ == '__main__':
    from src.data.features import to_feature_matrix
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.split import random_split, temporal_split

    OPTION_A = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
                'Ciprofloxacin', 'Cotrimoxazole']

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()

    splits = {
        'temporal': temporal_split(dataset, metadata_df),
        'random': random_split(dataset, test_size=1313 / dataset.n_samples),
    }

    comparison_rows = []
    for split_name, split in splits.items():
        print(f"\n=== split: {split_name} ===")

        shap_baseline(X, dataset, split, split_name, subsample_n=750)
        chain_results = shap_chain(X, dataset, split, split_name, OPTION_A, 'option_a', subsample_n=750)

        for antibiotic, result in chain_results.items():
            comparison_rows.append({
                'split': split_name,
                'antibiotic': antibiotic,
                'chain_position': result['chain_position'],
                'fraction_prior_attribution': result['fraction_prior_attribution'],
            })

    comparison_df = pd.DataFrame(comparison_rows).sort_values(['chain_position', 'split']).reset_index(drop=True)
    print("\nfraction_prior_attribution comparison (option_a order):")
    print(comparison_df)
