"""
Phase 0: chance-corrected duplicate-cluster detection, all 4 species.

Why this exists. The criterion in scripts/phase0/duplicate_clusters.py is raw
value agreement > 95% together with missingness agreement > 95%. Raw agreement
is confounded by prevalence: for two independent labels with susceptibility
rates p and q, the agreement expected by chance alone is

    p*q + (1-p)*(1-q)

At p = q = 0.98 that is 0.9604, which already clears a 95% threshold. On a
low-prevalence panel the rule therefore fires on pairs that merely happen to be
rarely resistant. In S. aureus it flags Cotrimoxazole-Vancomycin (0.955) and
Daptomycin-Gentamicin (0.960), which share no resistance mechanism.

This script recomputes the same pairwise table and adds:
  - expected_agreement  : chance agreement given each label's own prevalence
  - excess_agreement    : observed - expected
  - cohen_kappa         : agreement corrected for chance

Cohen's kappa is the headline column. A genuine shared determination sits at
or near 1.0; a prevalence coincidence collapses toward 0 even when its raw
agreement is high.

Saves results/metrics/duplicate_clusters_kappa.csv with columns:
[species, antibiotic_a, antibiotic_b, n_jointly_labeled, prevalence_a,
 prevalence_b, value_agreement, expected_agreement, excess_agreement,
 cohen_kappa, missingness_agreement].
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

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

NON_ANTIBIOTIC_COLS = {'code', 'species', 'laboratory_species', 'year_folder',
                       'Unnamed: 0.1', 'Unnamed: 0'}

# Thresholds used for the summary print at the end.
KAPPA_DUPLICATE = 0.95
AGREEMENT_DUPLICATE = 0.95
MISSINGNESS_CORE = 0.95


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[duplicate_clusters_kappa] loading raw metadata...")
    metadata = load_raw_metadata(ID_FOLDER)

    rows = []
    for species_name, label in SPECIES_LABELS.items():
        species_df = metadata[metadata['species'] == species_name]
        if species_df.empty:
            print(f"  {label}: no rows, skipping")
            continue

        candidate_cols = []
        for col in species_df.columns:
            if col in NON_ANTIBIOTIC_COLS:
                continue
            collapsed = species_df[col].apply(lambda v: collapse_label(v, MISSING_TOKEN))
            if collapsed.notna().sum() >= MIN_NON_MISSING:
                candidate_cols.append(col)

        print(f"  {label}: {len(candidate_cols)} candidate antibiotic columns")

        collapsed_df = pd.DataFrame({
            col: species_df[col].apply(lambda v: collapse_label(v, MISSING_TOKEN))
            for col in candidate_cols
        })
        missing_df = collapsed_df.isna()

        for col_a, col_b in combinations(candidate_cols, 2):
            both_present = collapsed_df[col_a].notna() & collapsed_df[col_b].notna()
            n_joint = int(both_present.sum())
            if n_joint < MIN_NON_MISSING:
                continue

            labels_a = collapsed_df.loc[both_present, col_a].astype(int)
            labels_b = collapsed_df.loc[both_present, col_b].astype(int)

            observed = (labels_a == labels_b).mean()

            # Chance agreement from each label's own resistance prevalence.
            p_a = labels_a.mean()
            p_b = labels_b.mean()
            expected = p_a * p_b + (1 - p_a) * (1 - p_b)

            # cohen_kappa_score returns 0 when both labels are constant, which is
            # the behaviour we want here (no information to agree about).
            kappa = cohen_kappa_score(labels_a, labels_b)

            missingness = (missing_df[col_a] == missing_df[col_b]).mean()

            rows.append({
                'species': label,
                'antibiotic_a': col_a,
                'antibiotic_b': col_b,
                'n_jointly_labeled': n_joint,
                'prevalence_a': p_a,
                'prevalence_b': p_b,
                'value_agreement': observed,
                'expected_agreement': expected,
                'excess_agreement': observed - expected,
                'cohen_kappa': kappa,
                'missingness_agreement': missingness,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICS_DIR / 'duplicate_clusters_kappa.csv', index=False)
    print(f"\nwrote {len(result_df)} pairs to duplicate_clusters_kappa.csv")

    # How much the chance correction changes the picture.
    raw_hits = result_df[
        (result_df.value_agreement > AGREEMENT_DUPLICATE)
        & (result_df.missingness_agreement > MISSINGNESS_CORE)
    ]
    kappa_hits = raw_hits[raw_hits.cohen_kappa > KAPPA_DUPLICATE]

    print(f"\npairs passing the raw criterion   (agreement > {AGREEMENT_DUPLICATE}): {len(raw_hits)}")
    print(f"pairs surviving chance correction (kappa     > {KAPPA_DUPLICATE}): {len(kappa_hits)}")
    print(f"dropped as prevalence coincidence: {len(raw_hits) - len(kappa_hits)}")

    if not raw_hits.empty:
        print("\nby species, raw -> chance-corrected:")
        for species, group in raw_hits.groupby('species'):
            survived = (group.cohen_kappa > KAPPA_DUPLICATE).sum()
            print(f"  {species}: {len(group)} -> {survived}")

        print("\nlargest raw-agreement pairs that do NOT survive:")
        dropped = raw_hits[raw_hits.cohen_kappa <= KAPPA_DUPLICATE]
        cols = ['species', 'antibiotic_a', 'antibiotic_b', 'value_agreement',
                'expected_agreement', 'cohen_kappa']
        print(dropped.sort_values('value_agreement', ascending=False)[cols]
              .head(15).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
