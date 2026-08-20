"""
Classifier Chains: XGBoost base estimator, modeling label dependencies.

Example:
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.features import to_feature_matrix
    from src.data.split import temporal_split
    from src.models.chains import train_chain

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    order = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
             'Ciprofloxacin', 'Cotrimoxazole']
    results_df = train_chain(X, dataset, split, 'temporal', order, 'option_a')
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

from src.config import ANTIBIOTICS, RANDOM_SEED
from src.evaluation.metrics import multilabel_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_chain(X, dataset, split, split_name, order, order_name):
    """
    Train and evaluate a Classifier Chain over a given antibiotic order.

    Example:
        results_df = train_chain(X, dataset, split, 'temporal', OPTION_A, 'option_a')
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

    results_df = pd.DataFrame(rows, columns=['antibiotic', 'auroc', 'f1', 'split', 'order_name'])
    results_df['hamming_loss'] = joint_metrics['hamming_loss']
    results_df['jaccard_score'] = joint_metrics['jaccard_score']
    results_df['all_zero_frac'] = all_zero_frac

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f'chains_{order_name}_{split_name}.csv', index=False)

    return results_df


if __name__ == '__main__':
    from src.data.features import to_feature_matrix
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.split import random_split, temporal_split

    OPTION_A = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
                'Ciprofloxacin', 'Cotrimoxazole']
    OPTION_D = ['Ceftriaxone', 'Ciprofloxacin', 'Ampicillin-Amoxicillin',
                'Cotrimoxazole', 'Amoxicillin-Clavulanic acid']
    OPTION_E = ['Ciprofloxacin', 'Ceftriaxone', 'Ampicillin-Amoxicillin',
                'Cotrimoxazole', 'Amoxicillin-Clavulanic acid']
    ORDERS = [(OPTION_A, 'option_a'), (OPTION_D, 'option_d'), (OPTION_E, 'option_e')]

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # temporal split
    split = temporal_split(dataset, metadata_df)

    for order, order_name in ORDERS:
        results_df = train_chain(X, dataset, split, 'temporal', order, order_name)
        print(f"\n{order_name} temporal split summary:")
        print(results_df)

    # random split, repeated over 10 seeds, matched to temporal's test set size
    test_size = 1313 / dataset.n_samples

    random_results_dfs = []
    for seed in range(10):
        split = random_split(dataset, test_size=test_size, seed=seed)
        for order, order_name in ORDERS:
            results_df = train_chain(X, dataset, split, f'random_seed{seed}', order, order_name)
            results_df['seed'] = seed
            random_results_dfs.append(results_df)

    random_results_df = pd.concat(random_results_dfs, ignore_index=True)
    random_results_df.to_csv(output_dir / 'chains_random_seeds.csv', index=False)

    per_antibiotic_summary = (
        random_results_df.groupby(['order_name', 'antibiotic'])[['auroc', 'f1']]
        .agg(['mean', 'std'])
    )
    per_antibiotic_summary.columns = ['_'.join(col) for col in per_antibiotic_summary.columns]
    per_antibiotic_summary = per_antibiotic_summary.reset_index()

    per_seed_joint = random_results_df.drop_duplicates(subset=['order_name', 'seed'])[
        ['order_name', 'hamming_loss', 'jaccard_score']
    ]
    joint_summary = (
        per_seed_joint.groupby('order_name')[['hamming_loss', 'jaccard_score']]
        .agg(['mean', 'std'])
    )
    joint_summary.columns = ['_'.join(col) for col in joint_summary.columns]
    joint_summary = joint_summary.reset_index()

    summary_df = per_antibiotic_summary.merge(joint_summary, on='order_name')

    print("\nrandom split summary (per antibiotic, across 10 seeds):")
    print(per_antibiotic_summary)
    print("\nrandom split summary (joint metrics, across 10 seeds):")
    print(joint_summary)

    summary_df.to_csv(output_dir / 'chains_random_summary.csv', index=False)
