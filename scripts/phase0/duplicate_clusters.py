"""
Phase 0: duplicate-cluster value-agreement and missingness-agreement checks,
for every antibiotic pair with adequate coverage, across all 4 species.

Systematic scan (not limited to a pre-selected candidate list): for each
species, every antibiotic column with >= MIN_NON_MISSING raw (non-'-',
non-NaN) values for that species is checked pairwise against every other
such column. This is a superset of any informally "checked" pairs from
earlier exploration -- it enumerates the full candidate pool systematically
so nothing pre-existing is silently excluded.

value_agreement: among rows where BOTH antibiotics have a non-missing
collapsed R/S(/I->R) label, fraction where the two labels match.
missingness_agreement: across the full species cohort, fraction of rows
where the two antibiotics' raw missingness (missing vs. present) matches.

Saves results/metrics/duplicate_clusters.csv with columns:
[species, antibiotic_a, antibiotic_b, n_jointly_labeled, value_agreement,
 missingness_agreement].
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase0._common import METRICS_DIR, collapse_label, load_raw_metadata
from src.config import ID_FOLDER, MISSING_TOKEN

MIN_NON_MISSING = 500

SPECIES_LABELS = {
    'Escherichia coli': 'ecoli',
    'Staphylococcus aureus': 'saureus',
    'Klebsiella pneumoniae': 'kpneumoniae',
    'Pseudomonas aeruginosa': 'paeruginosa',
}

NON_ANTIBIOTIC_COLS = {'code', 'species', 'laboratory_species', 'year_folder', 'Unnamed: 0.1', 'Unnamed: 0'}


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[duplicate_clusters] loading raw metadata...")
    metadata = load_raw_metadata(ID_FOLDER)
    antibiotic_cols = [c for c in metadata.columns if c not in NON_ANTIBIOTIC_COLS]

    rows = []
    for full_species, label in SPECIES_LABELS.items():
        species_df = metadata[metadata['species'] == full_species]
        is_missing = (species_df[antibiotic_cols] == MISSING_TOKEN) | species_df[antibiotic_cols].isna()

        non_missing_counts = (~is_missing).sum(axis=0)
        candidate_cols = non_missing_counts[non_missing_counts >= MIN_NON_MISSING].index.tolist()
        print(f"[duplicate_clusters] {label}: {len(candidate_cols)} candidate antibiotics with >= {MIN_NON_MISSING} non-missing")

        collapsed = species_df[candidate_cols].apply(lambda col: col.apply(lambda v: collapse_label(v, MISSING_TOKEN)))

        for a, b in combinations(candidate_cols, 2):
            jointly_labeled = collapsed[a].notna() & collapsed[b].notna()
            n_jointly_labeled = jointly_labeled.sum()
            if n_jointly_labeled == 0:
                continue

            value_agreement = (collapsed.loc[jointly_labeled, a] == collapsed.loc[jointly_labeled, b]).mean()
            missingness_agreement = (is_missing[a] == is_missing[b]).mean()

            rows.append({
                'species': label,
                'antibiotic_a': a,
                'antibiotic_b': b,
                'n_jointly_labeled': n_jointly_labeled,
                'value_agreement': value_agreement,
                'missingness_agreement': missingness_agreement,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'duplicate_clusters.csv', index=False)

    print(f"[duplicate_clusters] saved {len(result_df)} pairs total")
    high_agreement = result_df[(result_df['value_agreement'] >= 0.95) & (result_df['n_jointly_labeled'] >= MIN_NON_MISSING)]
    print(f"[duplicate_clusters] pairs with value_agreement >= 0.95 (n>={MIN_NON_MISSING}): {len(high_agreement)}")
    print(high_agreement.sort_values(['species', 'value_agreement'], ascending=[True, False]))


if __name__ == '__main__':
    main()
