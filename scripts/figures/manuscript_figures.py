"""
Manuscript figures F2-F5 for the dependency-aware AMR paper.

These are the publication versions, kept separate from the exploratory r*.py figures. They use a
restrained serif style suited to a two-column IEEE template rather than the colour-coded
internal style, and they write to results/figures/manuscript/ so the existing figures are not
overwritten.

F2  all-susceptible prediction fraction by configuration, with the true rate as a reference line
F3  boundary proximity against chain-baseline divergence, four species
F4  top-15 SHAP feature overlap by chain position, against the retraining noise floor
F5  trivial all-susceptible predictor against the trained baseline, four species

F1 (the architecture diagram) is drawn separately and is not produced here.

Two reference lines are not optional. F2 without the true-rate line reads backwards, because a
lower all-susceptible fraction looks better when in fact the target is the true rate. F4 without
the noise floor makes the decline look like it could be retraining variance. Both are drawn.

Reads from results/metrics/: true_all_zero_rates.csv, chains_option_{a,d,e,f}_temporal.csv,
ensemble_temporal.csv, targeted_ensemble_temporal.csv, boundary_proximity.csv,
shap_baseline_chain_overlap.csv, shap_noise_floor.csv, trivial_predictor_baseline.csv,
baseline_temporal.csv, and the per-species baseline_*_temporal.csv files.

Saves results/figures/manuscript/f{2,3,4,5}_*.{png,pdf}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures' / 'manuscript'

# Two inks only: near-black for the data, mid grey for context and reference lines. Anything
# that needs a third distinction is separated by hatch or marker, not by hue, so the figures
# survive greyscale printing.
INK = '#1a1a1a'
GREY = '#8c8c8c'
LIGHT = '#d9d9d9'

SPECIES_LABELS = {
    'ecoli': 'E. coli',
    'saureus': 'S. aureus',
    'kpneumoniae': 'K. pneumoniae',
    'paeruginosa': 'P. aeruginosa',
}

SPECIES_ORDER = ['ecoli', 'saureus', 'kpneumoniae', 'paeruginosa']


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'savefig.dpi': 400,
    })


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=400, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')
    print(f'  saved {name}.png and {name}.pdf')


def read_metric(filename, column):
    """Read one joint-metric value from a results CSV (the column is constant down the file)."""
    df = pd.read_csv(METRICS_DIR / filename)
    return float(df[column].iloc[0])


# ---------------------------------------------------------------- F2

def figure_f2():
    """All-susceptible prediction fraction by configuration, against the true rate."""
    true_rate = pd.read_csv(METRICS_DIR / 'true_all_zero_rates.csv')
    true_rate = float(true_rate.loc[true_rate.species == 'ecoli', 'true_all_zero_rate'].iloc[0])

    # Drawn horizontally: the configuration names are too long for readable x tick labels, and
    # ordering by value makes the seeding effect legible at a glance.
    configs = [
        ('Random-order ECC', 'ensemble_temporal.csv'),
        ('Option F (R-majority last)', 'chains_option_f_temporal.csv'),
        ('Option D (Ceftriaxone seed)', 'chains_option_d_temporal.csv'),
        ('Option E (Ciprofloxacin seed)', 'chains_option_e_temporal.csv'),
        ('Seed-constrained ECC', 'targeted_ensemble_temporal.csv'),
        ('Option A (R-majority seed)', 'chains_option_a_temporal.csv'),
    ]

    labels = [label for label, _ in configs]
    values = [read_metric(filename, 'all_zero_frac') for _, filename in configs]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    positions = range(len(values))

    ax.barh(list(positions), values, color=LIGHT, edgecolor=INK, linewidth=0.7, height=0.66)

    ax.axvline(true_rate, color=INK, linestyle='--', linewidth=1.0)
    ax.text(true_rate - 0.012, len(values) - 0.35,
            f'true rate {true_rate:.3f}',
            ha='right', va='center', fontsize=7.5, color=INK)

    for y, value in zip(positions, values):
        ax.text(value + 0.008, y, f'{value:.3f}', ha='left', va='center', fontsize=7.5)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('fraction of test isolates predicted fully susceptible')
    ax.set_xlim(0, 0.88)

    fig.tight_layout()
    save_figure(fig, 'f2_all_susceptible_fraction')
    plt.close(fig)


# ---------------------------------------------------------------- F3

def figure_f3():
    """Boundary proximity against chain-baseline divergence, four species."""
    boundary = pd.read_csv(METRICS_DIR / 'boundary_proximity.csv')
    boundary = boundary[boundary.antibiotic == '__overall__'].set_index('species')

    # Chain-baseline agreement lives in each species' Option A chain results.
    agreement_files = {
        'ecoli': 'chains_option_a_temporal.csv',
        'saureus': 'chains_saureus_option_a_temporal.csv',
        'kpneumoniae': 'chains_kpneumoniae_option_a_temporal.csv',
        'paeruginosa': 'chains_paeruginosa_option_a_temporal.csv',
    }

    xs, ys, names = [], [], []
    for species in SPECIES_ORDER:
        path = METRICS_DIR / agreement_files[species]
        if not path.exists():
            print(f'  F3: missing {path.name}, skipping {species}')
            continue
        agreement = read_metric(agreement_files[species], 'chain_baseline_agreement')
        xs.append(float(boundary.loc[species, 'boundary_fraction']))
        ys.append(1.0 - agreement)
        names.append(SPECIES_LABELS[species])

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.plot(xs, ys, color=GREY, linewidth=0.8, zorder=1)
    ax.scatter(xs, ys, s=38, color=INK, zorder=2)

    # S. aureus and K. pneumoniae sit close together, so their labels are placed by hand rather
    # than at a uniform offset.
    label_offsets = {
        'E. coli': (-8, -11),
        'S. aureus': (7, 4),
        'K. pneumoniae': (2, -13),
        'P. aeruginosa': (7, 1),
    }
    alignments = {'E. coli': 'right'}

    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords='offset points',
                    xytext=label_offsets.get(name, (6, -2)),
                    ha=alignments.get(name, 'left'), fontsize=8, style='italic')

    ax.set_xlabel('boundary proximity\n(fraction of predictions in [0.3, 0.7])')
    ax.set_ylabel('chain-baseline divergence\n(1 - agreement)')
    ax.set_xlim(-0.012, 0.185)
    ax.set_ylim(-0.03, 0.46)

    fig.tight_layout()
    save_figure(fig, 'f3_boundary_proximity')
    plt.close(fig)


# ---------------------------------------------------------------- F4

def figure_f4():
    """Top-15 SHAP overlap by chain position, against the retraining noise floor."""
    overlap = pd.read_csv(METRICS_DIR / 'shap_baseline_chain_overlap.csv')
    overlap = overlap.sort_values('chain_position')

    noise = pd.read_csv(METRICS_DIR / 'shap_noise_floor.csv')
    floor = float(noise['top15_overlap_count'].min())

    positions = overlap['chain_position'].to_numpy() + 1
    counts = overlap['top15_overlap_count'].to_numpy()
    short_names = {
        'Ampicillin-Amoxicillin': 'AMP',
        'Ceftriaxone': 'CRO',
        'Amoxicillin-Clavulanic acid': 'AMC',
        'Ciprofloxacin': 'CIP',
        'Cotrimoxazole': 'SXT',
    }
    labels = [short_names.get(a, a) for a in overlap['antibiotic']]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.axhline(floor, color=GREY, linestyle='--', linewidth=1.0)
    ax.text(len(positions) + 0.35, floor - 0.4,
            f'noise floor: {floor:.0f}/15\n(model seed varied)',
            ha='right', va='top', fontsize=7.5, color=GREY)

    ax.plot(positions, counts, color=INK, linewidth=1.0, marker='o', markersize=4.5)

    # The curve descends to the right, so value labels go below-left of each marker to stay off
    # the line.
    for x, y in zip(positions, counts):
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points', xytext=(-7, -7),
                    ha='right', va='top', fontsize=7.5)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f'{p}\n{label}' for p, label in zip(positions, labels)], fontsize=7.5)
    ax.set_xlabel('chain position')
    ax.set_xlim(0.6, 5.5)
    ax.set_ylabel('top-15 spectral features shared\nwith independent baseline')
    ax.set_ylim(0, 16.5)

    fig.tight_layout()
    save_figure(fig, 'f4_shap_overlap_by_position')
    plt.close(fig)


# ---------------------------------------------------------------- F5

def figure_f5():
    """Trivial all-susceptible predictor against the trained baseline, four species."""
    trivial = pd.read_csv(METRICS_DIR / 'trivial_predictor_baseline.csv').set_index('species')

    baseline_files = {
        'ecoli': 'baseline_temporal.csv',
        'saureus': 'baseline_saureus_temporal.csv',
        'kpneumoniae': 'baseline_kpneumoniae_temporal.csv',
        'paeruginosa': 'baseline_paeruginosa_temporal.csv',
    }

    species, hamming_pairs, match_pairs = [], [], []
    for name in SPECIES_ORDER:
        path = METRICS_DIR / baseline_files[name]
        if not path.exists():
            print(f'  F5: missing {path.name}, skipping {name}')
            continue
        species.append(SPECIES_LABELS[name])
        hamming_pairs.append((float(trivial.loc[name, 'hamming_loss']),
                              read_metric(baseline_files[name], 'hamming_loss')))
        match_pairs.append((float(trivial.loc[name, 'exact_match_ratio']),
                            read_metric(baseline_files[name], 'exact_match_ratio')))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    width = 0.36

    # 'lower' / 'higher': which direction counts as the trivial predictor winning.
    panels = [
        (axes[0], hamming_pairs, 'Hamming loss (lower is better)', 'lower', 0.42),
        (axes[1], match_pairs, 'exact-match ratio (higher is better)', 'higher', 0.92),
    ]

    for ax, pairs, title, better, top in panels:
        positions = range(len(pairs))
        trivial_values = [p[0] for p in pairs]
        trained_values = [p[1] for p in pairs]

        ax.bar([x - width / 2 for x in positions], trivial_values, width=width,
               color='white', edgecolor=INK, linewidth=0.7, hatch='////',
               label='trivial all-susceptible')
        ax.bar([x + width / 2 for x in positions], trained_values, width=width,
               color=LIGHT, edgecolor=INK, linewidth=0.7,
               label='trained Binary Relevance')

        # The whole point of this figure is the species where the trivial predictor wins. The
        # margins there are a fraction of a percent, so they are invisible at bar scale and have
        # to be called out or the figure argues against itself.
        for x, (trivial_value, trained_value) in zip(positions, pairs):
            trivial_wins = (trivial_value < trained_value if better == 'lower'
                            else trivial_value > trained_value)
            if not trivial_wins:
                continue
            height = max(trivial_value, trained_value)
            ax.text(x, height + 0.025 * top, '*', ha='center', va='bottom',
                    fontsize=13, color=INK)

        ax.set_xticks(list(positions))
        ax.set_xticklabels(species, fontsize=7.5, style='italic', rotation=20, ha='right')
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, top)

    # One shared legend below the panels: three asterisks land in the right panel's upper area,
    # so an in-axes legend collides with them whichever corner it is put in.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.09))
    fig.text(0.5, -0.16, '* trivial predictor matches or beats the trained model',
             ha='center', fontsize=7.5)

    fig.tight_layout()
    save_figure(fig, 'f5_trivial_predictor')
    plt.close(fig)


def main():
    apply_style()
    for name, builder in [('F2', figure_f2), ('F3', figure_f3),
                          ('F4', figure_f4), ('F5', figure_f5)]:
        print(f'[manuscript_figures] {name}')
        builder()


if __name__ == '__main__':
    main()
