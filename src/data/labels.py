"""
Build a binary label matrix for AMR resistance prediction.

Example:
    import pandas as pd
    from src.data.labels import build_label_matrix

    metadata_df = pd.read_csv("path/to/id/metadata.csv")
    isolates, labels = build_label_matrix(metadata_df)
"""

import pandas as pd

from src.config import ANTIBIOTICS, MISSING_TOKEN, SPECIES


def build_label_matrix(metadata_df):
    species_df = metadata_df[metadata_df['species'] == SPECIES]

    complete_mask = pd.Series(True, index=species_df.index)
    for antibiotic in ANTIBIOTICS:
        column = species_df[antibiotic]
        complete_mask &= column.notna() & (column != MISSING_TOKEN)

    filtered_df = species_df[complete_mask]

    label_matrix_df = pd.DataFrame(index=filtered_df.index)
    for antibiotic in ANTIBIOTICS:
        label_matrix_df[antibiotic] = filtered_df[antibiotic].apply(_to_binary_label)

    return filtered_df, label_matrix_df


def _to_binary_label(value):
    if 'R' in value:
        return 1
    if 'I' in value:
        return 1
    if 'S' in value:
        return 0
    raise ValueError(f"Unrecognized resistance value: {value!r}")


if __name__ == '__main__':
    import sys

    metadata_df = pd.read_csv(sys.argv[1])
    isolates, labels = build_label_matrix(metadata_df)
    print(isolates.shape, labels.shape)
    print(labels.sum())
