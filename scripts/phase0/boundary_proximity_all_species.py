"""
Phase 0: boundary-proximity summary for all 4 species.

Why this exists. The scope condition for chain-baseline divergence is stated over all four
species, but `boundary_probas.py` only ever saved predicted-probability arrays for E. coli and
K. pneumoniae, so the claim is evidenced on two of the four. With two points the
correlation-vs-boundary contrast is an anecdote; with four it is a result.

This extends the same procedure to S. aureus and P. aeruginosa and additionally persists the
summary statistic itself, which was previously recomputed at figure time and never stored.

Boundary proximity is the fraction of baseline predicted probabilities falling inside
[0.3, 0.7] -- near enough to the 0.5 decision threshold that an inherited prior-label signal
could flip the decision. Where the baseline is confident, a chain's prior features are absorbed
without changing the output and the chain reduces to the baseline; where it is uncertain, they
decide.

Reuses `baseline_probas_{species}_temporal.csv` when already present rather than refitting, so
re-running is cheap and the two existing species keep the exact numbers already reported.

Cost: one Binary Relevance refit (5 XGBoost models) for each species whose probability file is
missing -- normally S. aureus and P. aeruginosa only.

Saves:
  results/metrics/baseline_probas_{species}_temporal.csv  (for any species missing it)
  results/metrics/boundary_proximity.csv with columns:
      [species, antibiotic, n_test, boundary_fraction]
    plus one row per species with antibiotic='__overall__'.
"""

import sys
from pathlib import Path

import pandas as pd
from maldi_learn.driams import load_driams_dataset
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR
from src.data.features import to_feature_matrix
from src.data.load import load_metadata_with_years
from src.data.split import temporal_split

SPECIES_CONFIGS = [
    ('ecoli', 'src.config'),
    ('saureus', 'src.config_saureus'),
    ('kpneumoniae', 'src.config_kpneumoniae'),
    ('paeruginosa', 'src.config_paeruginosa'),
]

# Window around the 0.5 decision threshold. Reported explicitly in the manuscript, since the
# absolute fractions depend on it -- the cross-species ordering does not.
BOUNDARY_LOW = 0.3
BOUNDARY_HIGH = 0.7


def fit_baseline_probas(label, module_name, metadata_df):
    """Fit Binary Relevance on the temporal train years, return test-set probabilities."""
    module = __import__(
        module_name, fromlist=['DATA_ROOT', 'SPECIES', 'ANTIBIOTICS', 'RANDOM_SEED']
    )
    DATA_ROOT, SPECIES, ANTIBIOTICS, RANDOM_SEED = (
        module.DATA_ROOT, module.SPECIES, module.ANTIBIOTICS, module.RANDOM_SEED
    )

    print(f"[boundary_proximity] {label}: loading dataset...")
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
    split = temporal_split(dataset, metadata_df)
    train_idx, test_idx = split['train_idx'], split['test_idx']
    X_train, X_test = X[train_idx], X[test_idx]

    proba_cols = {}
    for antibiotic in ANTIBIOTICS:
        print(f"[boundary_proximity] {label}/{antibiotic}: fitting...")
        y = dataset.to_numpy(antibiotic)
        model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
        model.fit(X_train, y[train_idx])
        proba_cols[antibiotic] = model.predict_proba(X_test)[:, 1]

    return pd.DataFrame(proba_cols)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_df = load_metadata_with_years()

    rows = []
    for label, module_name in SPECIES_CONFIGS:
        proba_path = METRICS_DIR / f'baseline_probas_{label}_temporal.csv'

        if proba_path.exists():
            print(f"[boundary_proximity] {label}: reusing existing {proba_path.name}")
            proba_df = pd.read_csv(proba_path)
        else:
            proba_df = fit_baseline_probas(label, module_name, metadata_df)
            proba_df.to_csv(proba_path, index=False)
            print(f"[boundary_proximity] {label}: saved {proba_df.shape} to {proba_path.name}")

        in_band = (proba_df >= BOUNDARY_LOW) & (proba_df <= BOUNDARY_HIGH)
        n_test = len(proba_df)

        for antibiotic in proba_df.columns:
            rows.append({
                'species': label,
                'antibiotic': antibiotic,
                'n_test': n_test,
                'boundary_fraction': in_band[antibiotic].mean(),
            })

        overall = in_band.to_numpy().mean()
        rows.append({
            'species': label,
            'antibiotic': '__overall__',
            'n_test': n_test,
            'boundary_fraction': overall,
        })
        print(f"[boundary_proximity] {label}: overall boundary fraction {overall:.4f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'boundary_proximity.csv', index=False)
    print()
    print(result_df.round(4).to_string(index=False))

    print()
    print("Overall by species, for the scope-condition contrast:")
    overall_df = (
        result_df[result_df.antibiotic == '__overall__']
        .sort_values('boundary_fraction', ascending=False)
    )
    print(overall_df[['species', 'boundary_fraction']].round(4).to_string(index=False))
    print()
    print("Read against chain-baseline agreement (from chains_*_temporal.csv):")
    print("  high boundary fraction -> low agreement (chain diverges from baseline)")
    print("  low  boundary fraction -> high agreement (chain reduces to baseline)")
    print("E. coli 0.1397 / agreement 0.6177 and K. pneumoniae 0.0202 / agreement 0.9354 are")
    print("the two points already established; S. aureus (0.9268) and P. aeruginosa (0.9971)")
    print("should fall between them if the scope condition holds.")


if __name__ == '__main__':
    main()
