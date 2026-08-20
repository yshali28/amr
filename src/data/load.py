"""
Load the DRIAMS-A dataset via maldi_learn.

Example:
    from src.data.load import load_dataset

    dataset = load_dataset()
    print(dataset.n_samples)
"""

import glob
import os

import pandas as pd
from maldi_learn.driams import load_driams_dataset

from src.config import ANTIBIOTICS, DATA_ROOT, ID_FOLDER, SPECIES


def load_dataset():
    root = DATA_ROOT.parent

    dataset = load_driams_dataset(
        root=str(root),
        site='DRIAMS-A',
        years=['2015', '2016', '2017', '2018'],
        species=SPECIES,
        antibiotics=ANTIBIOTICS,
        handle_missing_resistance_measurements='remove_if_any_missing',
        spectra_type='binned_6000',
    )

    return dataset


def load_metadata_with_years():
    """
    Load 'code' -> 'year_folder' lookup info for temporal_split.

    Example:
        from src.data.load import load_metadata_with_years

        metadata_df = load_metadata_with_years()
        print(metadata_df.columns.tolist())
    """
    paths = glob.glob(os.path.join(ID_FOLDER, '*', '*_clean.csv'))

    frames = []
    for path in paths:
        year_folder = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path, low_memory=False)
        df['year_folder'] = year_folder
        frames.append(df)

    metadata_df = pd.concat(frames, ignore_index=True)

    return metadata_df[['code', 'year_folder']]


if __name__ == '__main__':
    dataset = load_dataset()
    print(f"n_samples: {dataset.n_samples}")
    print(f"antibiotics: {dataset.y.columns.tolist()}")
