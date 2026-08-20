"""
Ensembles of Classifier Chains (ECC): average predictions across chains
fit with different antibiotic orderings.

Example:
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.features import to_feature_matrix
    from src.data.split import temporal_split
    from src.models.ensemble import train_ensemble, train_targeted_ensemble

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    result = train_ensemble(X, dataset, split, 'temporal')
    result['results_df']
    result['individual_all_zero_fracs']

    result = train_targeted_ensemble(X, dataset, split, 'temporal', 'Ampicillin-Amoxicillin')
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


def _train_chain_ensemble(X, dataset, split, split_name, order_fn, output_prefix, n_chains):
    """
    Shared ECC training loop: fit n_chains ClassifierChains (order chosen
    per-chain by order_fn(i)), average their probabilities, and evaluate.
    """
    train_idx = split['train_idx']
    test_idx = split['test_idx']

    Y = np.column_stack([dataset.to_numpy(antibiotic) for antibiotic in ANTIBIOTICS])

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    n_antibiotics = len(ANTIBIOTICS)
    individual_all_zero_fracs = []
    proba_sum = np.zeros((len(test_idx), n_antibiotics))

    for i in range(n_chains):
        order = order_fn(i)

        chain = ClassifierChain(
            estimator=XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'),
            order=order,
            random_state=RANDOM_SEED,
        )
        chain.fit(X_train, Y_train)

        # ClassifierChain.predict/predict_proba already return columns
        # back in the original Y column order (ANTIBIOTICS order here),
        # regardless of the internal chain `order` used for fitting.
        Y_proba_i = chain.predict_proba(X_test)
        Y_pred_i = chain.predict(X_test)

        chain_all_zero_frac = (Y_pred_i.sum(axis=1) == 0).mean()
        individual_all_zero_fracs.append(chain_all_zero_frac)

        proba_sum += Y_proba_i

    Y_proba = proba_sum / n_chains
    Y_pred = (Y_proba >= 0.5).astype(int)

    rows = []
    for i, antibiotic in enumerate(ANTIBIOTICS):
        auroc = roc_auc_score(Y_test[:, i], Y_proba[:, i])
        f1 = f1_score(Y_test[:, i], Y_pred[:, i])

        print(f"{antibiotic}: AUROC={auroc:.4f}, F1={f1:.4f}")

        rows.append({'antibiotic': antibiotic, 'auroc': auroc, 'f1': f1, 'split': split_name, 'order_name': 'ecc'})

    joint_metrics = multilabel_metrics(Y_test, Y_pred)
    all_zero_frac = (Y_pred.sum(axis=1) == 0).mean()

    print(f"ensemble joint metrics: {joint_metrics}")
    print(f"ensemble all_zero_frac: {all_zero_frac:.4f}")

    individual_arr = np.array(individual_all_zero_fracs)
    print(
        f"individual chain all_zero_frac: mean={individual_arr.mean():.4f}, "
        f"min={individual_arr.min():.4f}, max={individual_arr.max():.4f}"
    )

    results_df = pd.DataFrame(rows, columns=['antibiotic', 'auroc', 'f1', 'split', 'order_name'])
    results_df['hamming_loss'] = joint_metrics['hamming_loss']
    results_df['jaccard_score'] = joint_metrics['jaccard_score']
    results_df['all_zero_frac'] = all_zero_frac

    output_dir = PROJECT_ROOT / 'results' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f'{output_prefix}_{split_name}.csv', index=False)

    individual_fracs_df = pd.DataFrame({'chain_idx': range(n_chains), 'all_zero_frac': individual_all_zero_fracs})
    individual_fracs_df.to_csv(output_dir / f'{output_prefix}_{split_name}_individual_zero_fracs.csv', index=False)

    return {'results_df': results_df, 'individual_all_zero_fracs': individual_all_zero_fracs}


def train_ensemble(X, dataset, split, split_name, n_chains=10):
    """
    Train and evaluate an ensemble of Classifier Chains (ECC) with fully
    random antibiotic orderings.

    Example:
        result = train_ensemble(X, dataset, split, 'temporal', n_chains=10)
    """
    n_antibiotics = len(ANTIBIOTICS)

    def order_fn(i):
        return np.random.RandomState(RANDOM_SEED + i).permutation(n_antibiotics).tolist()

    return _train_chain_ensemble(X, dataset, split, split_name, order_fn, 'ensemble', n_chains)


def train_targeted_ensemble(X, dataset, split, split_name, seed_antibiotic, n_chains=10):
    """
    Train and evaluate an ensemble of Classifier Chains (ECC) where every
    chain places seed_antibiotic first and randomly shuffles the rest.

    Example:
        result = train_targeted_ensemble(X, dataset, split, 'temporal', 'Ampicillin-Amoxicillin')
    """
    n_antibiotics = len(ANTIBIOTICS)
    seed_idx = ANTIBIOTICS.index(seed_antibiotic)
    remaining_idx = [i for i in range(n_antibiotics) if i != seed_idx]

    def order_fn(i):
        shuffled_remaining = np.random.RandomState(RANDOM_SEED + i).permutation(remaining_idx).tolist()
        return [seed_idx] + shuffled_remaining

    return _train_chain_ensemble(X, dataset, split, split_name, order_fn, 'targeted_ensemble', n_chains)


if __name__ == '__main__':
    from src.data.features import to_feature_matrix
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.split import temporal_split

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    result = train_ensemble(X, dataset, split, 'temporal')
    print("\ntemporal split summary (random-order ensemble):")
    print(result['results_df'])

    targeted_result = train_targeted_ensemble(X, dataset, split, 'temporal', 'Ampicillin-Amoxicillin')
    print("\ntemporal split summary (targeted ensemble, seed=Ampicillin-Amoxicillin):")
    print(targeted_result['results_df'])

    # TODO(follow-up): random-split evaluation over 10 seeds, matching chains.py's approach.
