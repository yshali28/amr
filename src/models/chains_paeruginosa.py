"""
Classifier Chains for P. aeruginosa: XGBoost base estimator, modeling label dependencies.

Example:
    from maldi_learn.driams import load_driams_dataset
    from src.data.features import to_feature_matrix
    from src.data.load import load_metadata_with_years
    from src.data.split import temporal_split
    from src.config_paeruginosa import DATA_ROOT, SPECIES, ANTIBIOTICS, OPTION_A_PAERUGINOSA
    from src.models.chains_paeruginosa import train_chain_paeruginosa

    dataset = load_driams_dataset(
        root=str(DATA_ROOT.parent), site='DRIAMS-A',
        years=['2015', '2016', '2017', '2018'], species=SPECIES, antibiotics=ANTIBIOTICS,
        handle_missing_resistance_measurements='remove_if_any_missing', spectra_type='binned_6000',
    )
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)
    results_df = train_chain_paeruginosa(X, dataset, split, 'temporal', OPTION_A_PAERUGINOSA, 'option_a')
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

from src.config_paeruginosa import (
    ANTIBIOTICS,
    DATA_ROOT,
    OPTION_A_PAERUGINOSA,
    RANDOM_SEED,
    REVERSE_ORDER_PAERUGINOSA,
    SPECIES,
)
from src.evaluation.metrics import multilabel_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JOINT_METRIC_COLUMNS = ['hamming_loss', 'jaccard_score', 'jaccard_score_zd1', 'exact_match_ratio', 'restricted_jaccard']


def train_chain_paeruginosa(X, dataset, split, split_name, order, order_name):
    """
    Train and evaluate a Classifier Chain over a given P. aeruginosa antibiotic order.

    Example:
        results_df = train_chain_paeruginosa(X, dataset, split, 'temporal', OPTION_A_PAERUGINOSA, 'option_a')
    """
    train_idx = split['train_idx']
    test_idx = split['test_idx']

    Y = np.column_stack([dataset.to_numpy(antibiotic) for antibiotic in order])

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    chain = ClassifierChain(
        estimator=XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'),
        order=list(range(len(order))),
        random_state=RANDOM_SEED,
    )
    chain.fit(X_train, Y_train)

    Y_pred = chain.predict(X_test)
    Y_proba = chain.predict_proba(X_test)

    rows = []
    for i, antibiotic in enumerate(order):
        auroc = roc_auc_score(Y_test[:, i], Y_proba[:, i])
        f1 = f1_score(Y_test[:, i], Y_pred[:, i])

        print(f"{antibiotic}: AUROC={auroc:.4f}, F1={f1:.4f}")

        rows.append({'antibiotic': antibiotic, 'auroc': auroc, 'f1': f1, 'split': split_name, 'order_name': order_name})

    joint_metrics = multilabel_metrics(Y_test, Y_pred)
    print(f"joint metrics: {joint_metrics}")

    all_zero_frac = (Y_pred.sum(axis=1) == 0).mean()
    print(f"all_zero_frac: {all_zero_frac:.4f}")

    # Chain-baseline agreement: fraction of test isolates where the chain's
    # full predicted label vector exactly matches a freshly-trained
    # independent baseline's, for the same split and column order.
    baseline_pred = np.zeros_like(Y_pred)
    for i, antibiotic in enumerate(order):
        baseline_model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
        baseline_model.fit(X_train, Y_train[:, i])
        baseline_pred[:, i] = baseline_model.predict(X_test)

    chain_baseline_agreement = (Y_pred == baseline_pred).all(axis=1).mean()
    print(f"chain-baseline agreement: {chain_baseline_agreement:.4f}")

    results_df = pd.DataFrame(rows, columns=['antibiotic', 'auroc', 'f1', 'split', 'order_name'])
    for col in JOINT_METRIC_COLUMNS:
        results_df[col] = joint_metrics[col]
    results_df['all_zero_frac'] = all_zero_frac
    results_df['chain_baseline_agreement'] = chain_baseline_agreement

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f'chains_paeruginosa_{order_name}_{split_name}.csv', index=False)

    return results_df


if __name__ == '__main__':
    from maldi_learn.driams import load_driams_dataset

    from src.data.features import to_feature_matrix
    from src.data.load import load_metadata_with_years
    from src.data.split import random_split, temporal_split

    ORDERS = [
        (OPTION_A_PAERUGINOSA, 'option_a'),
        (REVERSE_ORDER_PAERUGINOSA, 'reverse_order'),
    ]

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

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # temporal split
    temporal_split_result = temporal_split(dataset, metadata_df)

    temporal_results_dfs = []
    for order, order_name in ORDERS:
        results_df = train_chain_paeruginosa(X, dataset, temporal_split_result, 'temporal', order, order_name)
        print(f"\n{order_name} temporal split summary:")
        print(results_df)
        print(f"{order_name} chain-baseline agreement (temporal): {results_df['chain_baseline_agreement'].iloc[0]:.4f}")
        temporal_results_dfs.append(results_df)

    # random split, repeated over 10 seeds, matched to temporal's test set size
    test_size = len(temporal_split_result['test_idx']) / dataset.n_samples

    random_results_dfs = []
    for seed in range(10):
        split = random_split(dataset, test_size=test_size, seed=seed)
        for order, order_name in ORDERS:
            results_df = train_chain_paeruginosa(X, dataset, split, f'random_seed{seed}', order, order_name)
            results_df['seed'] = seed
            random_results_dfs.append(results_df)

    random_results_df = pd.concat(random_results_dfs, ignore_index=True)
    random_results_df.to_csv(output_dir / 'chains_paeruginosa_random_seeds.csv', index=False)

    per_antibiotic_summary = (
        random_results_df.groupby(['order_name', 'antibiotic'])[['auroc', 'f1']]
        .agg(['mean', 'std'])
    )
    per_antibiotic_summary.columns = ['_'.join(col) for col in per_antibiotic_summary.columns]
    per_antibiotic_summary = per_antibiotic_summary.reset_index()

    joint_cols = JOINT_METRIC_COLUMNS + ['all_zero_frac', 'chain_baseline_agreement']
    per_seed_joint = random_results_df.drop_duplicates(subset=['order_name', 'seed'])[['order_name'] + joint_cols]
    joint_summary = (
        per_seed_joint.groupby('order_name')[joint_cols]
        .agg(['mean', 'std'])
    )
    joint_summary.columns = ['_'.join(col) for col in joint_summary.columns]
    joint_summary = joint_summary.reset_index()

    summary_df = per_antibiotic_summary.merge(joint_summary, on='order_name')

    print("\nrandom split summary (per antibiotic, across 10 seeds):")
    print(per_antibiotic_summary)
    print("\nrandom split summary (joint metrics, across 10 seeds):")
    print(joint_summary)

    summary_df.to_csv(output_dir / 'chains_paeruginosa_random_summary.csv', index=False)
