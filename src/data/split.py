"""
Train/test splitting strategies for a DRIAMSDataset: temporal and random.

Example:
    from src.data.load import load_dataset
    from src.data.split import temporal_split, random_split

    dataset = load_dataset()

    split = temporal_split(dataset, metadata_df)
    X_train, y_train = dataset.to_numpy('Ciprofloxacin')
    X_train, y_train = X_train[split['train_idx']], y_train[split['train_idx']]

    split = random_split(dataset)
    X_test, y_test = dataset.to_numpy('Ciprofloxacin')
    X_test, y_test = X_test[split['test_idx']], y_test[split['test_idx']]
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TEST_YEARS, TRAIN_YEARS


def _attach_year(dataset, metadata_df):
    """
    Merge dataset.y with metadata_df on 'code' to recover year info.

    Example:
        merged = _attach_year(dataset, metadata_df)
        merged['year_folder']
    """
    y_with_pos = dataset.y.reset_index(drop=True).copy()
    y_with_pos['_pos'] = y_with_pos.index

    merged = y_with_pos.merge(metadata_df[['code', 'year_folder']], on='code', how='left')

    if merged['year_folder'].isna().any():
        n_missing = merged['year_folder'].isna().sum()
        raise ValueError(f"{n_missing} rows failed to match a year_folder after merge")

    return merged


def temporal_split(dataset, metadata_df):
    """
    Split dataset positions into train/test based on year_folder.

    Example:
        split = temporal_split(dataset, metadata_df)
        X, y = dataset.to_numpy('Ciprofloxacin')
        X_train, y_train = X[split['train_idx']], y[split['train_idx']]
        X_test, y_test = X[split['test_idx']], y[split['test_idx']]
    """
    merged = _attach_year(dataset, metadata_df)
    years = merged['year_folder'].astype(int)

    train_idx = merged.loc[years.isin(TRAIN_YEARS), '_pos'].to_numpy()
    test_idx = merged.loc[years.isin(TEST_YEARS), '_pos'].to_numpy()

    print(f"temporal_split: train={len(train_idx)}, test={len(test_idx)}")

    return {'train_idx': train_idx, 'test_idx': test_idx}


def random_split(dataset, test_size=0.25, seed=None):
    """
    Standard random split of dataset positions (no stratification, multi-label).

    Example:
        split = random_split(dataset)              # uses RANDOM_SEED from config
        split = random_split(dataset, seed=123)     # uses a custom seed
        X, y = dataset.to_numpy('Ciprofloxacin')
        X_train, y_train = X[split['train_idx']], y[split['train_idx']]
        X_test, y_test = X[split['test_idx']], y[split['test_idx']]
    """
    if seed is None:
        seed = RANDOM_SEED

    train_idx, test_idx = train_test_split(
        np.arange(dataset.n_samples), test_size=test_size, random_state=seed, stratify=None,
    )

    print(f"random_split: train={len(train_idx)}, test={len(test_idx)}")

    return {'train_idx': train_idx, 'test_idx': test_idx}
