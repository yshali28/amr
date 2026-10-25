"""
One-off backfill: rerun the temporal-split baseline + Option A chain for
E. coli, S. aureus, and K. pneumoniae (baseline only) with the newly added
metric columns (jaccard_score_zd1, exact_match_ratio, restricted_jaccard,
chain_baseline_agreement), so R12 has a genuinely complete cross-species
table instead of N/A gaps.

Prints AUROC/F1 for direct comparison against the values already reported
in this conversation, as a reproducibility safeguard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def ecoli():
    from src.data.load import load_dataset, load_metadata_with_years
    from src.data.features import to_feature_matrix
    from src.data.split import temporal_split
    from src.models.baseline import train_baseline
    from src.models.chains import train_chain

    print("[r12_backfill] ecoli: loading dataset...")
    dataset = load_dataset()
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    print("[r12_backfill] ecoli: baseline...")
    train_baseline(X, dataset, split, 'temporal')

    OPTION_A = ['Ampicillin-Amoxicillin', 'Ceftriaxone', 'Amoxicillin-Clavulanic acid',
                'Ciprofloxacin', 'Cotrimoxazole']
    print("[r12_backfill] ecoli: chain option_a...")
    train_chain(X, dataset, split, 'temporal', OPTION_A, 'option_a')


def saureus():
    from maldi_learn.driams import load_driams_dataset
    from src.config_saureus import ANTIBIOTICS, DATA_ROOT, OPTION_A_SAUREUS, SPECIES
    from src.data.features import to_feature_matrix
    from src.data.load import load_metadata_with_years
    from src.data.split import temporal_split
    from src.models.baseline_saureus import train_baseline_saureus
    from src.models.chains_saureus import train_chain_saureus

    print("[r12_backfill] saureus: loading dataset...")
    dataset = load_driams_dataset(
        root=str(DATA_ROOT.parent), site='DRIAMS-A',
        years=['2015', '2016', '2017', '2018'], species=SPECIES, antibiotics=ANTIBIOTICS,
        handle_missing_resistance_measurements='remove_if_any_missing', spectra_type='binned_6000',
    )
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    print("[r12_backfill] saureus: baseline...")
    train_baseline_saureus(X, dataset, split, 'temporal')

    print("[r12_backfill] saureus: chain option_a...")
    train_chain_saureus(X, dataset, split, 'temporal', OPTION_A_SAUREUS, 'option_a')


def kpneumoniae():
    from maldi_learn.driams import load_driams_dataset
    from src.config_kpneumoniae import ANTIBIOTICS, DATA_ROOT, SPECIES
    from src.data.features import to_feature_matrix
    from src.data.load import load_metadata_with_years
    from src.data.split import temporal_split
    from src.models.baseline_kpneumoniae import train_baseline_kpneumoniae

    print("[r12_backfill] kpneumoniae: loading dataset...")
    dataset = load_driams_dataset(
        root=str(DATA_ROOT.parent), site='DRIAMS-A',
        years=['2015', '2016', '2017', '2018'], species=SPECIES, antibiotics=ANTIBIOTICS,
        handle_missing_resistance_measurements='remove_if_any_missing', spectra_type='binned_6000',
    )
    X = to_feature_matrix(dataset)
    metadata_df = load_metadata_with_years()
    split = temporal_split(dataset, metadata_df)

    print("[r12_backfill] kpneumoniae: baseline...")
    train_baseline_kpneumoniae(X, dataset, split, 'temporal')


if __name__ == '__main__':
    ecoli()
    saureus()
    kpneumoniae()
