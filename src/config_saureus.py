from src.config import DATA_ROOT, MISSING_TOKEN, RANDOM_SEED

SPECIES = 'Staphylococcus aureus'

# Panel excludes Cotrimoxazole -- dropped after F1=0.000 across all 11 baseline
# runs (degenerate predictor, contributes no chain-relevant signal).
# Metadata-only complete-case count is 3,626; the actual spectrum-validated
# sample count (via load_driams_dataset) is 3,434 -- use the latter in all
# reported results.
ANTIBIOTICS = ['Penicillin', 'Erythromycin', 'Clindamycin', 'Oxacillin', 'Ciprofloxacin']
TRAIN_YEARS = [2015, 2016, 2017]
TEST_YEARS = [2018]

# Option A: R-majority antibiotic seeded first, weakly correlated with the rest of the chain.
OPTION_A_SAUREUS = ['Penicillin', 'Erythromycin', 'Clindamycin', 'Oxacillin', 'Ciprofloxacin']

# Misseeded (Oxacillin): S-majority antibiotic seeded first, well-correlated with the rest
# of the chain -- the realistic failure case where an early error propagates. Penicillin
# (R-majority) is displaced to position 4, so this tests the compound condition "S-majority
# seed + R-majority label displaced late," not seed identity in isolation.
MISSEEDED_OXACILLIN = ['Oxacillin', 'Erythromycin', 'Clindamycin', 'Penicillin', 'Ciprofloxacin']

# Misseeded (Ciprofloxacin): S-majority antibiotic seeded first, weakly correlated with the
# rest of the chain -- a controlled contrast against misseeded_oxacillin. Penicillin
# (R-majority) is displaced to position 5, so this too tests the compound condition
# "S-majority seed + R-majority label displaced late," not seed identity in isolation.
MISSEEDED_CIPROFLOXACIN = ['Ciprofloxacin', 'Erythromycin', 'Clindamycin', 'Oxacillin', 'Penicillin']

# Misseeded (Ciprofloxacin), Penicillin second: cleaner variant that keeps Penicillin
# early (position 2) instead of displacing it late, to separate the two variables
# conflated in misseeded_oxacillin / misseeded_ciprofloxacin -- whether Penicillin
# specifically needs to be in position 1, or whether position 2 still calibrates.
MISSEEDED_CIPROFLOXACIN_PENICILLIN_SECOND = ['Ciprofloxacin', 'Penicillin', 'Erythromycin',
                                              'Clindamycin', 'Oxacillin']
