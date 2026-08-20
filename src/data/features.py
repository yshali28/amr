"""
Convert a DRIAMSDataset's spectra into a dense feature matrix.

Example:
    from src.data.load import load_dataset
    from src.data.features import to_feature_matrix

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    print(X.shape)
"""

import numpy as np


def to_feature_matrix(dataset):
    return np.vstack([spectrum.intensities for spectrum in dataset.X])


if __name__ == '__main__':
    from src.data.load import load_dataset

    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    print(f"shape: {X.shape}")
