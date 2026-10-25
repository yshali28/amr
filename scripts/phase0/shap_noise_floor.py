"""
Phase 0: SHAP noise floor for E. coli -- how much do top-15 spectral feature
rankings move between two Binary Relevance baselines that differ only in the
model's random seed?

Why this matters. The write-up reports that top-15 overlap varies by antibiotic
when comparing baseline against chain, and treats that variation as a finding.
That only holds if feature rankings are stable under retraining in the first
place. If two independently-seeded baselines already disagree by the same
margin, the "finding" is training noise.

An earlier check reportedly returned 100% overlap across four antibiotics, but it
was never established whether that run varied the MODEL seed or only the
train/test split seed. This varies the model seed explicitly, holding the split
fixed, which is the comparison that matters.

Cost: two baseline refits (5 XGBoost models each) on the temporal split.

Saves results/metrics/shap_noise_floor.csv with columns:
[antibiotic, seed_a, seed_b, top15_overlap_count, top15_overlap_fraction].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.config import ANTIBIOTICS
from src.data.features import to_feature_matrix
from src.data.load import load_dataset, load_metadata_with_years
from src.data.split import temporal_split

TOP_K = 15
SEED_A = 42
SEED_B = 1337
# SHAP on 6000 features is the slow part, so subsample the test set. Attribution
# rankings stabilise well before the full test set is needed.
SHAP_SAMPLE = 300


def mean_abs_shap(X_train, y_train, X_shap, seed):
    """Fit one XGBoost model at the given seed, return mean |SHAP| per feature."""
    model = XGBClassifier(random_state=seed, eval_metric='logloss')
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    return np.abs(shap_values).mean(axis=0)


def top_k_indices(arr, k=TOP_K):
    return set(np.argsort(arr)[::-1][:k].tolist())


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    train_idx, test_idx = split['train_idx'], split['test_idx']
    X_train, X_test = X[train_idx], X[test_idx]

    rng = np.random.default_rng(SEED_A)
    sample_size = min(SHAP_SAMPLE, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_shap = X_test[sample_idx]
    print(f"SHAP sample: {sample_size} of {len(X_test)} test isolates")

    rows = []
    for antibiotic in ANTIBIOTICS:
        y = dataset.to_numpy(antibiotic)
        y_train = y[train_idx]

        print(f"[shap_noise_floor] {antibiotic}: fitting seed {SEED_A}...")
        shap_a = mean_abs_shap(X_train, y_train, X_shap, SEED_A)
        print(f"[shap_noise_floor] {antibiotic}: fitting seed {SEED_B}...")
        shap_b = mean_abs_shap(X_train, y_train, X_shap, SEED_B)

        overlap = top_k_indices(shap_a) & top_k_indices(shap_b)
        rows.append({
            'antibiotic': antibiotic,
            'seed_a': SEED_A,
            'seed_b': SEED_B,
            'top15_overlap_count': len(overlap),
            'top15_overlap_fraction': len(overlap) / TOP_K,
        })
        print(f"  overlap: {len(overlap)}/{TOP_K}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'shap_noise_floor.csv', index=False)
    print()
    print(result_df.to_string(index=False))

    floor = result_df['top15_overlap_fraction'].min()
    print(f"\nlowest overlap under seed change alone: {floor:.2%}")
    print("Any baseline-vs-chain overlap at or below this level is training noise,")
    print("not a chain effect.")


if __name__ == '__main__':
    main()
