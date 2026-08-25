"""
R12: Cross-species comparison table -- exact_match_ratio, restricted_jaccard,
hamming_loss, jaccard_score from each species' baseline CSV (temporal split),
chain_baseline_agreement from the chains CSVs (only available for
K. pneumoniae and P. aeruginosa -- E. coli/S. aureus chains.py never computed
this column, shown as N/A), and |correlation| ranges from the Phase 0
correlation matrix CSVs.

Saves results/figures/r12_cross_species_comparison.png and a matching
results/figures/r12_cross_species_comparison.md.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import FIGURES_DIR, apply_style

METRICS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'metrics'

SPECIES = [
    ('ecoli', 'E. coli', 'baseline_temporal.csv', 'chains_option_a_temporal.csv'),
    ('saureus', 'S. aureus', 'baseline_saureus_temporal.csv', 'chains_saureus_option_a_temporal.csv'),
    ('kpneumoniae', 'K. pneumoniae', 'baseline_kpneumoniae_temporal.csv', 'chains_kpneumoniae_option_a_temporal.csv'),
    ('paeruginosa', 'P. aeruginosa', 'baseline_paeruginosa_temporal.csv', 'chains_paeruginosa_option_a_temporal.csv'),
]


def main():
    apply_style()

    rows = []
    for label, display_name, baseline_file, chains_file in SPECIES:
        baseline_df = pd.read_csv(METRICS_DIR / baseline_file)
        joint = baseline_df.iloc[0]

        corr = pd.read_csv(METRICS_DIR / f'correlation_matrix_{label}.csv', index_col=0)
        off_diag = corr.values[~np.eye(len(corr), dtype=bool)]
        corr_range = f"[{abs(off_diag).min():.2f}, {abs(off_diag).max():.2f}]"

        chains_path = METRICS_DIR / chains_file
        if chains_path.exists():
            chains_df = pd.read_csv(chains_path)
            agreement = chains_df['chain_baseline_agreement'].iloc[0] if 'chain_baseline_agreement' in chains_df.columns else np.nan
        else:
            agreement = np.nan

        rows.append({
            'species': display_name,
            'exact_match_ratio': joint.get('exact_match_ratio', np.nan),
            'restricted_jaccard': joint.get('restricted_jaccard', np.nan),
            'hamming_loss': joint['hamming_loss'],
            'jaccard_score': joint['jaccard_score'],
            'chain_baseline_agreement': agreement,
            '|corr| range': corr_range,
        })

    result_df = pd.DataFrame(rows)

    # markdown
    md_lines = ['| ' + ' | '.join(result_df.columns) + ' |',
                '| ' + ' | '.join(['---'] * len(result_df.columns)) + ' |']
    for _, row in result_df.iterrows():
        cells = []
        for col in result_df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append('N/A' if pd.isna(val) else f'{val:.3f}')
            else:
                cells.append(str(val))
        md_lines.append('| ' + ' | '.join(cells) + ' |')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / 'r12_cross_species_comparison.md').write_text('\n'.join(md_lines), encoding='utf-8')

    # image
    display_df = result_df.copy()
    for col in ['exact_match_ratio', 'restricted_jaccard', 'hamming_loss', 'jaccard_score', 'chain_baseline_agreement']:
        display_df[col] = display_df[col].apply(lambda v: 'N/A' if pd.isna(v) else f'{v:.3f}')

    fig, ax = plt.subplots(figsize=(12, 0.6 * (len(display_df) + 1) + 0.5))
    ax.axis('off')
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title('Cross-species comparison (temporal split, Binary Relevance baseline joint metrics)', pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'r12_cross_species_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'r12_cross_species_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

    print(result_df)
    print("saved r12_cross_species_comparison")


if __name__ == '__main__':
    main()
