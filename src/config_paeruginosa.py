from src.config import DATA_ROOT, MISSING_TOKEN, RANDOM_SEED

SPECIES = 'Pseudomonas aeruginosa'

# No R-majority label exists here either. Aztreonam was the only candidate
# at 72.7% R, but costs 93.4% of the cohort (n=777 vs peers at 2,450-3,267)
# -- excluded. P. aeruginosa is a second no-valid-seed species alongside
# K. pneumoniae.
#
# Colistin removed from the modeling panel -- see COLISTIN_CASE_STUDY below.
# Tobramycin and Amikacin were both tried and rejected after producing
# near-zero test-set resistance rates once required in the joint
# complete-case population (a selection effect where isolates tested for
# less-common agents skew toward lower resistance). Cefepime was tested
# next and confirmed both independent (all value-match checks against the
# existing panel below 95%) and well-behaved (no near-zero R-rates).
#
# Ceftazidime/Piperacillin-Tazobactam remains a borderline-duplicate pair
# (92.1% value match, secondary tier) present in the final panel -- flagged
# as a limitation, consistent with how similar borderline pairs were
# handled in other species. Cefepime/Ceftazidime also shows a notable but
# sub-threshold value match (92.1%) worth the same note.
#
# n_samples=2,192 (spectrum-validated), temporal split train=1,491, test=701.
ANTIBIOTICS = ['Piperacillin-Tazobactam', 'Meropenem', 'Ciprofloxacin',
               'Ceftazidime', 'Cefepime']
TRAIN_YEARS = [2015, 2016, 2017]
TEST_YEARS = [2018]

# Kept as a reference for the separate threshold case-study analysis
# (F1=0.000 across all 11 baseline runs, AUROC ranging 0.50-0.87 depending
# on seed, at ~1.6% prevalence -- the third such case alongside Meropenem in
# K. pneumoniae and Cotrimoxazole in S. aureus/E. coli) -- not used in any
# chain/ensemble experiment, since it's excluded from the modeling panel above.
COLISTIN_CASE_STUDY = 'Colistin'

# Ordered by descending mean |correlation|: Piperacillin-Tazobactam 0.479,
# Cefepime 0.456, Ceftazidime 0.443, Meropenem 0.359, Ciprofloxacin 0.260.
OPTION_A_PAERUGINOSA = ['Piperacillin-Tazobactam', 'Cefepime', 'Ceftazidime',
                         'Meropenem', 'Ciprofloxacin']

# Exact reverse of OPTION_A_PAERUGINOSA (correlation-ascending order).
REVERSE_ORDER_PAERUGINOSA = ['Ciprofloxacin', 'Meropenem', 'Ceftazidime',
                              'Cefepime', 'Piperacillin-Tazobactam']
