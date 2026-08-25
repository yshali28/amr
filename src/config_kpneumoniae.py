from src.config import DATA_ROOT, MISSING_TOKEN, RANDOM_SEED

SPECIES = 'Klebsiella pneumoniae'

# Meropenem removed from the modeling panel -- see MEROPENEM_CASE_STUDY below.
# Metadata-only complete-case count is 2,787; the actual spectrum-validated
# sample count (via load_driams_dataset) is 2,723 -- use the latter in all
# reported results (same caveat pattern as config_saureus.py).
#
# Cotrimoxazole's correlation with this panel (mean |corr| 0.541) is
# dramatically higher than its role in E. coli's panel (mean |corr|
# 0.091-0.147) -- same antibiotic, radically different correlation
# structure, illustrating K. pneumoniae's broad plasmid-mediated
# multi-drug-resistance coupling. No genuinely weakly-correlated internal
# control exists in this panel (unlike E. coli): every antibiotic in
# K. pneumoniae's candidate pool showed mean |corr| >= 0.24 with the core
# group.
ANTIBIOTICS = ['Ceftriaxone', 'Tobramycin', 'Piperacillin-Tazobactam',
               'Ciprofloxacin', 'Cotrimoxazole']
TRAIN_YEARS = [2015, 2016, 2017]
TEST_YEARS = [2018]

# Kept as a reference for the separate threshold case-study analysis
# (AUROC ~0.82, F1 near 0 at ~2% prevalence) -- not used in any chain/ensemble
# experiment, since it's excluded from the modeling panel above.
MEROPENEM_CASE_STUDY = 'Meropenem'

# Correlation-driven ordering (no R-majority label exists in this species --
# Ampicillin-Amoxicillin is 100% R, zero-variance, unusable as a target or seed.
# Every ordering here is necessarily S-majority-seeded, testing whether the
# anchoring/miscalibration effect appears even without a mitigating R-majority
# seed available, given this panel's much stronger correlation structure
# (mean |corr| 0.24-0.56) than E. coli (0.14-0.62) or S. aureus (0.15-0.33).
OPTION_A_KPNEUMONIAE = ['Ceftriaxone', 'Piperacillin-Tazobactam', 'Tobramycin',
                        'Ciprofloxacin', 'Cotrimoxazole']

# Reverse-correlation ordering: weakest-to-strongest, as a contrast
# to test whether ordering among the strongly-correlated core matters at all.
REVERSE_ORDER_KPNEUMONIAE = ['Cotrimoxazole', 'Ciprofloxacin', 'Tobramycin',
                              'Piperacillin-Tazobactam', 'Ceftriaxone']
