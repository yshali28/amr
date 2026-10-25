"""
Numeric audit: every figure stated in the manuscript, recomputed from results/metrics/ and
compared against what the text says.

Each CLAIMS entry is (section, description, value as printed in the text, value recomputed from
the CSV). A mismatch beyond the stated tolerance is a number that must not go to print.

Run from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = Path('results/metrics')

failures = []
checked = 0


def get(filename, column, row_filter=None, row_value=None):
    df = pd.read_csv(METRICS / filename)
    if row_filter is not None:
        df = df[df[row_filter] == row_value]
    return float(df[column].iloc[0])


def check(section, description, printed, computed, tol=5e-4):
    global checked
    checked += 1
    if computed is None:
        failures.append((section, description, printed, 'NOT COMPUTABLE'))
        return
    if abs(printed - computed) > tol:
        failures.append((section, description, printed, round(computed, 6)))


# ---- Table A: cohorts and base rates
azr = pd.read_csv(METRICS / 'true_all_zero_rates.csv').set_index('species')
cov = pd.read_csv(METRICS / 'covariate_shift_test.csv').set_index('species')

for species, cohort, train, test, rate in [
    ('ecoli', 4644, 3331, 1313, 0.3641),
    ('saureus', 3454, 2388, 1066, 0.2195),
    ('kpneumoniae', 2727, 1984, 743, 0.7079),
    ('paeruginosa', 2192, 1491, 701, 0.7332),
]:
    check('III-D', f'{species} cohort', cohort, float(cov.loc[species, 'n_total']), tol=0)
    check('III-D', f'{species} train', train, float(cov.loc[species, 'n_train_years']), tol=0)
    check('III-D', f'{species} test', test, float(cov.loc[species, 'n_test_year']), tol=0)
    check('III-D', f'{species} all-zero rate', rate,
          float(azr.loc[species, 'true_all_zero_rate']))
    check('III-D', f'{species} test (all-zero file)', test, float(azr.loc[species, 'n_test']), tol=0)

# ---- Table IV: E. coli configurations
check('V-A', 'BR hamming', 0.2591, get('baseline_temporal.csv', 'hamming_loss'))
check('V-A', 'BR jaccard', 0.2323, get('baseline_temporal.csv', 'jaccard_score'))
check('V-A', 'BR exact match', 0.2544, get('baseline_temporal.csv', 'exact_match_ratio'))
check('V-A', 'BR restricted jaccard', 0.3652, get('baseline_temporal.csv', 'restricted_jaccard'))

for label, filename, hamming, jaccard, all_zero in [
    ('Option A', 'chains_option_a_temporal.csv', 0.2754, 0.2453, 0.3899),
    ('Option D', 'chains_option_d_temporal.csv', 0.2789, 0.1514, 0.6535),
    ('Option E', 'chains_option_e_temporal.csv', 0.2812, 0.1550, 0.6436),
    ('Option F', 'chains_option_f_temporal.csv', 0.2795, 0.1198, 0.7616),
    ('Random ECC', 'ensemble_temporal.csv', 0.2727, 0.0991, 0.7982),
    ('Seeded ECC', 'targeted_ensemble_temporal.csv', 0.2717, 0.2447, 0.3907),
]:
    check('V-A/B', f'{label} hamming', hamming, get(filename, 'hamming_loss'))
    check('V-A/B', f'{label} jaccard', jaccard, get(filename, 'jaccard_score'))
    check('V-A/B', f'{label} all-zero frac', all_zero, get(filename, 'all_zero_frac'))

check('V-A', 'Option A exact match', 0.2704, get('chains_option_a_temporal.csv', 'exact_match_ratio'))
check('V-A', 'Option A restricted jaccard', 0.3858,
      get('chains_option_a_temporal.csv', 'restricted_jaccard'))

# Claimed margins of Option A over baseline.
check('V-A', 'Option A jaccard margin', 0.0130,
      get('chains_option_a_temporal.csv', 'jaccard_score') - get('baseline_temporal.csv', 'jaccard_score'))
check('V-A', 'Option A exact-match margin', 0.0160,
      get('chains_option_a_temporal.csv', 'exact_match_ratio') - get('baseline_temporal.csv', 'exact_match_ratio'))
check('V-A', 'Option A restricted-jaccard margin', 0.0206,
      get('chains_option_a_temporal.csv', 'restricted_jaccard') - get('baseline_temporal.csv', 'restricted_jaccard'))

# Seed-position AUROC identity: chain position 1 must equal the baseline for that agent.
base = pd.read_csv(METRICS / 'baseline_temporal.csv').set_index('antibiotic')
chain_a = pd.read_csv(METRICS / 'chains_option_a_temporal.csv').set_index('antibiotic')
check('V-A', 'AMP AUROC baseline == chain pos 1', float(base.loc['Ampicillin-Amoxicillin', 'auroc']),
      float(chain_a.loc['Ampicillin-Amoxicillin', 'auroc']))
check('V-A', 'AMP AUROC printed', 0.6517, float(base.loc['Ampicillin-Amoxicillin', 'auroc']))

# Per-agent baseline AUROC range quoted in V-A.
check('V-A', 'min baseline AUROC', 0.604, base['auroc'].min(), tol=1e-3)
check('V-A', 'max baseline AUROC', 0.862, base['auroc'].max(), tol=1e-3)
check('V-A', 'min baseline F1', 0.103, base['f1'].min(), tol=1e-3)
check('V-A', 'max baseline F1', 0.676, base['f1'].max(), tol=1e-3)

# ---- Table V: probability divergence
div = pd.read_csv(METRICS / 'probability_divergence.csv')
for species, position, printed in [
    ('ecoli', 0, 0.0000), ('ecoli', 1, 0.0932), ('ecoli', 2, 0.1276),
    ('ecoli', 3, 0.1085), ('ecoli', 4, 0.2278),
    ('kpneumoniae', 0, 0.0000), ('kpneumoniae', 1, 0.0373), ('kpneumoniae', 2, 0.0268),
    ('kpneumoniae', 3, 0.0523), ('kpneumoniae', 4, 0.0682),
]:
    row = div[(div.species == species) & (div.chain_position == position)]
    check('V-C', f'divergence {species} pos {position + 1}', printed,
          float(row['mean_abs_proba_diff'].iloc[0]))

# ---- Label inversion
check('V-C', 'inverted-seed all-zero frac', 0.0046,
      get('label_inversion_test.csv', 'all_zero_frac'))
check('V-C', 'inverted-seed jaccard', 0.5525,
      get('label_inversion_test.csv', 'jaccard_score'))

# ---- Table VI: cross-species
for species, filename, chain_file, hamming, jaccard, zd1, exact, restricted, agreement in [
    ('ecoli', 'baseline_temporal.csv', 'chains_option_a_temporal.csv',
     0.2591, 0.2323, 0.4174, 0.2544, 0.3652, 0.6177),
    ('saureus', 'baseline_saureus_temporal.csv', 'chains_saureus_option_a_temporal.csv',
     0.1597, 0.5365, 0.5806, 0.4906, 0.6874, 0.9268),
    ('kpneumoniae', 'baseline_kpneumoniae_temporal.csv', 'chains_kpneumoniae_option_a_temporal.csv',
     0.1300, 0.0223, 0.7154, 0.6985, 0.0762, 0.9354),
    ('paeruginosa', 'baseline_paeruginosa_temporal.csv', 'chains_paeruginosa_option_a_temporal.csv',
     0.1070, 0.0000, 0.7318, 0.7318, 0.0000, 0.9971),
]:
    check('V-D', f'{species} hamming', hamming, get(filename, 'hamming_loss'))
    check('V-D', f'{species} jaccard', jaccard, get(filename, 'jaccard_score'))
    check('V-D', f'{species} jaccard zd1', zd1, get(filename, 'jaccard_score_zd1'))
    check('V-D', f'{species} exact match', exact, get(filename, 'exact_match_ratio'))
    check('V-D', f'{species} restricted jaccard', restricted, get(filename, 'restricted_jaccard'))
    check('V-D', f'{species} chain-baseline agreement', agreement,
          get(chain_file, 'chain_baseline_agreement'))

# ---- Correlations quoted in V-D
for species, low, high, mean_abs in [
    ('ecoli', 0.239, 0.542, 0.379),
    ('saureus', 0.046, 0.801, 0.263),
    ('kpneumoniae', 0.444, 0.653, 0.581),
    ('paeruginosa', 0.182, 0.667, 0.399),
]:
    matrix = pd.read_csv(METRICS / f'correlation_matrix_{species}.csv', index_col=0).to_numpy(float)
    pairwise = matrix[np.triu_indices_from(matrix, 1)]
    check('V-D', f'{species} corr min', low, pairwise.min(), tol=1e-3)
    check('V-D', f'{species} corr max', high, pairwise.max(), tol=1e-3)
    check('V-D', f'{species} corr mean |r|', mean_abs, np.abs(pairwise).mean(), tol=1e-3)

# ---- Table VIII: SHAP overlap and noise floor
overlap = pd.read_csv(METRICS / 'shap_baseline_chain_overlap.csv').sort_values('chain_position')
for position, printed in enumerate([15, 10, 7, 6, 2]):
    check('V-E', f'overlap pos {position + 1}', printed,
          float(overlap['top15_overlap_count'].iloc[position]), tol=0)

noise = pd.read_csv(METRICS / 'shap_noise_floor.csv')
check('V-E', 'noise floor min', 15, float(noise['top15_overlap_count'].min()), tol=0)
check('V-E', 'noise floor rows', 5, float(len(noise)), tol=0)

# ---- Prior attribution
prior = pd.read_csv(METRICS / 'shap_prior_attribution.csv')
ecoli_temporal = prior[(prior.species == 'ecoli') & (prior.split == 'temporal')
                       & (prior.chain_position > 0)]
for position, printed in zip([1, 2, 3, 4], [0.1179, 0.1356, 0.1220, 0.1161]):
    row = ecoli_temporal[ecoli_temporal.chain_position == position]
    check('V-E', f'prior attribution pos {position + 1}', printed,
          float(row['fraction_prior_attribution'].iloc[0]))

ecoli_random = prior[(prior.species == 'ecoli') & (prior.split == 'random')
                     & (prior.chain_position > 0)]
check('V-E', 'prior attribution random min', 0.1029,
      ecoli_random['fraction_prior_attribution'].min())
check('V-E', 'prior attribution random max', 0.1308,
      ecoli_random['fraction_prior_attribution'].max())
kp_prior = prior[(prior.species == 'kpneumoniae')]
check('V-E', 'K. pneumoniae prior attribution', 0.1582,
      float(kp_prior['fraction_prior_attribution'].iloc[0]))

# ---- V-F: matched random split
rand = pd.read_csv(METRICS / 'baseline_random_summary.csv')
check('V-F', 'random hamming mean', 0.2463, float(rand['hamming_loss_mean'].iloc[0]))
check('V-F', 'random hamming sd', 0.0051, float(rand['hamming_loss_std'].iloc[0]))
check('V-F', 'random jaccard mean', 0.2730, float(rand['jaccard_score_mean'].iloc[0]))
check('V-F', 'random jaccard sd', 0.0047, float(rand['jaccard_score_std'].iloc[0]))

rand_by_agent = rand.set_index('antibiotic')
check('V-F', 'CIP random AUROC mean', 0.822, float(rand_by_agent.loc['Ciprofloxacin', 'auroc_mean']), tol=1e-3)
check('V-F', 'CIP random AUROC sd', 0.012, float(rand_by_agent.loc['Ciprofloxacin', 'auroc_std']), tol=1e-3)
check('V-F', 'CRO random AUROC mean', 0.879, float(rand_by_agent.loc['Ceftriaxone', 'auroc_mean']), tol=1e-3)
check('V-F', 'CRO random AUROC sd', 0.011, float(rand_by_agent.loc['Ceftriaxone', 'auroc_std']), tol=1e-3)
check('V-F', 'CIP temporal AUROC', 0.795, float(base.loc['Ciprofloxacin', 'auroc']), tol=1e-3)
check('V-F', 'CRO temporal AUROC', 0.862, float(base.loc['Ceftriaxone', 'auroc']), tol=1e-3)

# Claimed "roughly nine standard deviations" on Jaccard.
sigma = (float(rand['jaccard_score_mean'].iloc[0]) - get('baseline_temporal.csv', 'jaccard_score')) \
    / float(rand['jaccard_score_std'].iloc[0])
check('V-F', 'jaccard gap in SDs (~9)', 9.0, sigma, tol=1.0)

# ---- V-F: covariate shift
for species, auroc in [('ecoli', 0.9603), ('saureus', 0.9716),
                       ('kpneumoniae', 0.9550), ('paeruginosa', 0.9462)]:
    check('V-F', f'{species} year-prediction AUROC', auroc, float(cov.loc[species, 'cv_auroc_mean']))
# Section V-F now states 0.006 for three species and 0.011 for K. pneumoniae.
for species, sd in [('ecoli', 0.006), ('saureus', 0.005),
                    ('kpneumoniae', 0.011), ('paeruginosa', 0.006)]:
    check('V-F', f'{species} covariate AUROC sd', sd,
          float(cov.loc[species, 'cv_auroc_std']), tol=1e-3)

# ---- Table IX: threshold tuning
thr = pd.read_csv(METRICS / 'threshold_tuning_ecoli.csv')


def thr_value(antibiotic, method, column):
    row = thr[(thr.antibiotic == antibiotic) & (thr.method == method)]
    return float(row[column].iloc[0])


for antibiotic, auroc, f1_naive, pred_naive, f1_cal, pred_cal, true_rate in [
    ('Ciprofloxacin', 0.724, 0.502, 0.438, 0.500, 0.291, 0.273),
    ('Cotrimoxazole', 0.617, 0.510, 0.679, 0.433, 0.381, 0.316),
    ('Ceftriaxone', 0.795, 0.526, 0.251, 0.536, 0.229, 0.183),
    ('Amoxicillin-Clavulanic acid', 0.564, 0.443, 0.928, 0.306, 0.203, 0.285),
    ('Ampicillin-Amoxicillin', 0.598, 0.742, 1.000, 0.688, 0.695, 0.590),
]:
    check('V-F', f'{antibiotic} AUROC', auroc, thr_value(antibiotic, 'naive_f1', 'auroc'), tol=1e-3)
    check('V-F', f'{antibiotic} naive F1', f1_naive, thr_value(antibiotic, 'naive_f1', 'f1_at_tuned'), tol=1e-3)
    check('V-F', f'{antibiotic} naive pred rate', pred_naive, thr_value(antibiotic, 'naive_f1', 'pred_rate'), tol=1e-3)
    check('V-F', f'{antibiotic} calib F1', f1_cal, thr_value(antibiotic, 'calibration', 'f1_at_tuned'), tol=1e-3)
    check('V-F', f'{antibiotic} calib pred rate', pred_cal, thr_value(antibiotic, 'calibration', 'pred_rate'), tol=1e-3)
    check('V-F', f'{antibiotic} true rate', true_rate, thr_value(antibiotic, 'naive_f1', 'true_rate'), tol=1e-3)

check('V-F', 'F1 at 0.5 min', 0.031, thr['f1_at_0.5'].min(), tol=1e-3)
check('V-F', 'F1 at 0.5 max', 0.713, thr['f1_at_0.5'].max(), tol=1e-3)
check('V-F', 'AMC naive threshold', 0.003, thr_value('Amoxicillin-Clavulanic acid', 'naive_f1', 'threshold'), tol=1e-3)

# ---- Table X: trivial predictor
triv = pd.read_csv(METRICS / 'trivial_predictor_baseline.csv').set_index('species')
baseline_files = {
    'ecoli': 'baseline_temporal.csv',
    'saureus': 'baseline_saureus_temporal.csv',
    'kpneumoniae': 'baseline_kpneumoniae_temporal.csv',
    'paeruginosa': 'baseline_paeruginosa_temporal.csv',
}
for species, triv_h, triv_e in [
    ('ecoli', 0.3292, 0.3641), ('saureus', 0.2589, 0.2195),
    ('kpneumoniae', 0.1392, 0.7079), ('paeruginosa', 0.1064, 0.7332),
]:
    check('V-G', f'{species} trivial hamming', triv_h, float(triv.loc[species, 'hamming_loss']))
    check('V-G', f'{species} trivial exact match', triv_e, float(triv.loc[species, 'exact_match_ratio']))

# The two directional claims the paper makes.
pa_wins = float(triv.loc['paeruginosa', 'hamming_loss']) < get(baseline_files['paeruginosa'], 'hamming_loss')
check('V-G', 'P. aeruginosa trivial beats BR on hamming', True, pa_wins)

exact_wins = sum(
    float(triv.loc[s, 'exact_match_ratio']) > get(baseline_files[s], 'exact_match_ratio')
    for s in ['ecoli', 'saureus', 'kpneumoniae', 'paeruginosa']
)
check('V-G', 'species where trivial beats BR on exact match (claimed 3)', 3, exact_wins, tol=0)

kp_margin = get(baseline_files['kpneumoniae'], 'hamming_loss') - float(triv.loc['kpneumoniae', 'hamming_loss'])
check('V-G', 'K. pneumoniae hamming margin (claimed 0.0092)', -0.0092, kp_margin)

# ---- Table F: duplicate clusters
dup = pd.read_csv(METRICS / 'duplicate_clusters_kappa.csv')
raw = dup[(dup.value_agreement > 0.95) & (dup.missingness_agreement > 0.95)]
survived = raw[raw.cohen_kappa > 0.95]
check('V-G', 'total candidate pairs', 675, float(len(dup)), tol=0)
check('V-G', 'raw criterion hits', 92, float(len(raw)), tol=0)
check('V-G', 'survive kappa', 39, float(len(survived)), tol=0)
check('V-G', 'dropped', 53, float(len(raw) - len(survived)), tol=0)
check('V-G', 'over-report factor (claimed 2.4)', 2.4, len(raw) / len(survived), tol=0.05)

for species, raw_n, surv_n in [('saureus', 71, 37), ('ecoli', 10, 1), ('kpneumoniae', 11, 1)]:
    group = raw[raw.species == species]
    check('V-G', f'{species} raw hits', raw_n, float(len(group)), tol=0)
    check('V-G', f'{species} survived', surv_n, float((group.cohen_kappa > 0.95).sum()), tol=0)

# Named pairs.
for a, b, species, agreement, kappa in [
    ('Vancomycin', 'Tigecycline', 'saureus', 0.9997, 0.0000),
    ('Linezolid', 'Tigecycline', 'saureus', 0.9997, 0.0000),
    ('Vancomycin', 'Teicoplanin', 'saureus', 0.9983, 0.0000),
    ('Teicoplanin', 'Tigecycline', 'saureus', 0.9981, -0.0005),
]:
    row = dup[(dup.species == species)
              & (((dup.antibiotic_a == a) & (dup.antibiotic_b == b))
                 | ((dup.antibiotic_a == b) & (dup.antibiotic_b == a)))]
    if row.empty:
        failures.append(('V-G', f'{a}/{b} pair not found', agreement, 'MISSING'))
        continue
    check('V-G', f'{a}/{b} agreement', agreement, float(row['value_agreement'].iloc[0]), tol=1e-3)
    check('V-G', f'{a}/{b} kappa', kappa, float(row['cohen_kappa'].iloc[0]), tol=1e-3)

# Carbapenem borderline pairs.
for a, b, species, kappa in [
    ('Meropenem', 'Imipenem', 'kpneumoniae', 0.922),
    ('Meropenem', 'Ertapenem', 'kpneumoniae', 0.835),
    ('Meropenem', 'Imipenem', 'ecoli', 0.639),
]:
    row = dup[(dup.species == species)
              & (((dup.antibiotic_a == a) & (dup.antibiotic_b == b))
                 | ((dup.antibiotic_a == b) & (dup.antibiotic_b == a)))]
    check('V-G', f'{species} {a}/{b} kappa', kappa, float(row['cohen_kappa'].iloc[0]), tol=1e-3)

# beta-lactam cluster: 9 members -> 36 pairs, all surviving.
BETA_LACTAM = ['Piperacillin-Tazobactam', 'Meropenem', 'Cefepime', 'Imipenem', 'Ceftriaxone',
               'Amoxicillin-Clavulanic acid', 'Cefuroxime', 'Oxacillin', 'Cefazolin']
cluster = survived[(survived.species == 'saureus')
                   & survived.antibiotic_a.isin(BETA_LACTAM)
                   & survived.antibiotic_b.isin(BETA_LACTAM)]
check('V-G', 'beta-lactam cluster pairs surviving (claimed 36)', 36, float(len(cluster)), tol=0)
oxacillin_pairs = cluster[(cluster.antibiotic_a == 'Oxacillin') | (cluster.antibiotic_b == 'Oxacillin')]
check('V-G', 'oxacillin pairs in cluster (claimed 8)', 8, float(len(oxacillin_pairs)), tol=0)
check('V-G', 'cluster min kappa (claimed 0.9922)', 0.9922, float(cluster['cohen_kappa'].min()), tol=1e-3)

# ---- Table II: P. aeruginosa selection effect
sel = pd.read_csv(METRICS / 'selection_effect_paeruginosa.csv')
print('\nselection_effect_paeruginosa.csv columns:', list(sel.columns))
print(sel.round(4).to_string(index=False))

# ---- report
print(f'\n{"=" * 70}')
print(f'checked {checked} values')
if failures:
    print(f'{len(failures)} MISMATCH(ES):\n')
    for section, description, printed, computed in failures:
        print(f'  [{section}] {description}: text says {printed}, CSV gives {computed}')
    sys.exit(1)
print('all values match their source CSVs')
