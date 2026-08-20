"""
Binary Relevance baseline: one independent XGBoost model per antibiotic.

Example:
    from src.data.load import load_dataset
    from src.data.features import to_feature_matrix
    from src.data.split import random_split
    from src.models.baseline import train_baseline

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    split = random_split(dataset)
    results_df = train_baseline(X, dataset, split, 'random')
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import ANTIBIOTICS, RANDOM_SEED
from src.evaluation.metrics import multilabel_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_baseline(X, dataset, split, split_name):
    """
    Train and evaluate a per-antibiotic XGBoost baseline.

    Example:
        results_df = train_baseline(X, dataset, split, 'temporal')
    """
    train_idx = split['train_idx']
    test_idx = split['test_idx']

    X_train, X_test = X[train_idx], X[test_idx]

    true_matrix = np.zeros((len(test_idx), len(ANTIBIOTICS)), dtype=int)
    pred_matrix = np.zeros((len(test_idx), len(ANTIBIOTICS)), dtype=int)

    rows = []
    for i, antibiotic in enumerate(ANTIBIOTICS):
        y = dataset.to_numpy(antibiotic)
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auroc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        print(f"{antibiotic}: AUROC={auroc:.4f}, F1={f1:.4f}")

        rows.append({'antibiotic': antibiotic, 'auroc': auroc, 'f1': f1, 'split': split_name})

        true_matrix[:, i] = y_test
        pred_matrix[:, i] = y_pred

    joint_metrics = multilabel_metrics(true_matrix, pred_matrix)
    print(f"joint metrics: {joint_metrics}")

    results_df = pd.DataFrame(rows, columns=['antibiotic', 'auroc', 'f1', 'split'])
    results_df['hamming_loss'] = joint_metrics['hamming_loss']
    results_df['jaccard_score'] = joint_metrics['jaccard_score']

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f'baseline_{split_name}.csv', index=False)

    return results_df


if __name__ == '__main__':
    from src.data.features import to_feature_matrix
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.split import random_split, temporal_split

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # temporal split
    temporal_results_df = train_baseline(X, dataset, temporal_split(dataset, metadata_df), 'temporal')
    print("\ntemporal split summary:")
    print(temporal_results_df)

    # random split, repeated over 10 seeds, matched to temporal's test set size
    test_size = 1313 / dataset.n_samples

    random_results_dfs = []
    for seed in range(10):
        split = random_split(dataset, test_size=test_size, seed=seed)
        results_df = train_baseline(X, dataset, split, f'random_seed{seed}')
        results_df['seed'] = seed
        random_results_dfs.append(results_df)

    random_results_df = pd.concat(random_results_dfs, ignore_index=True)
    random_results_df.to_csv(output_dir / 'baseline_random_seeds.csv', index=False)

    per_antibiotic_summary = (
        random_results_df.groupby('antibiotic')[['auroc', 'f1']]
        .agg(['mean', 'std'])
    )
    per_antibiotic_summary.columns = ['_'.join(col) for col in per_antibiotic_summary.columns]
    per_antibiotic_summary = per_antibiotic_summary.reset_index()

    per_seed_joint = random_results_df.drop_duplicates(subset='seed')[['hamming_loss', 'jaccard_score']]
    joint_summary = per_seed_joint.agg(['mean', 'std'])

    print("\nrandom split summary (per antibiotic, across 10 seeds):")
    print(per_antibiotic_summary)
    print("\nrandom split summary (joint metrics, across 10 seeds):")
    print(joint_summary)

    per_antibiotic_summary['hamming_loss_mean'] = joint_summary.loc['mean', 'hamming_loss']
    per_antibiotic_summary['hamming_loss_std'] = joint_summary.loc['std', 'hamming_loss']
    per_antibiotic_summary['jaccard_score_mean'] = joint_summary.loc['mean', 'jaccard_score']
    per_antibiotic_summary['jaccard_score_std'] = joint_summary.loc['std', 'jaccard_score']

    per_antibiotic_summary.to_csv(output_dir / 'baseline_random_summary.csv', index=False)
