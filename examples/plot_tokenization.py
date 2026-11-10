"""Regenerate the README figures from published aggregate measurements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.results / 'tokenization.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'reticular-diploid-v1'})
    navy, teal, gray = '#183B56', '#008C82', '#A5B3BF'
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    val = data['validation']
    values = [val['residual_only_mean'], val['residual_only_mean']/2, val['mean']]
    labels = ['Residual-only grammar', 'Nonreference allele entries', 'Frozen motif tokenizer']
    bars = ax.barh(labels[::-1], values[::-1], color=[teal, gray, navy], height=.55)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 160, bar.get_y()+bar.get_height()/2, f'{value:,.0f}', va='center', weight='bold')
    ax.set_xlim(0, 12600)
    ax.set_xlabel('Mean count per validation individual')
    ax.set_title('Genetic differences across 2.76 Mb', loc='left', weight='bold', pad=20)
    ax.text(0, 1.02, '55 validation individuals · 8,192 token IDs · exact reconstruction',
            transform=ax.transAxes, color='#536574', fontsize=10)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for suffix in ['svg', 'png']:
        fig.savefig(args.output / f'tokenization.{suffix}', dpi=180, metadata={'Date': None} if suffix=='svg' else None)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for split, color, people in [('training', navy, 400), ('validation', teal, 55)]:
        with (args.results / f'motif-frequency-{split}.csv').open() as handle:
            counts = np.array([int(row['occurrences']) for row in csv.DictReader(handle)])
        counts = np.sort(counts[counts>0])[::-1]
        ax.loglog(np.arange(1,len(counts)+1), counts/counts.sum(), color=color,
                  linewidth=2, label=f'{split.capitalize()} (n={people})')
    ax.set_title('Motif use has a common head and a long tail', loc='left', weight='bold', pad=20)
    ax.text(0, 1.02, 'Motif tokens only · D1 and D2 counted separately · independent ranks by split',
            transform=ax.transAxes, color='#536574', fontsize=10)
    ax.set_xlabel('Token rank within split (log scale)')
    ax.set_ylabel('Fraction of motif emissions (log scale)')
    ax.legend(frameon=False); ax.grid(alpha=.15)
    for suffix in ['svg', 'png']:
        fig.savefig(args.output / f'motif-frequency.{suffix}', dpi=180, metadata={'Date': None} if suffix=='svg' else None)
    plt.close(fig)


    bank = json.loads((args.results / 'biobank-ntp.json').read_text())
    steps = [row['step'] / 1000 for row in bank['curve']]
    losses = [row['validation_ce'] for row in bank['curve']]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(steps, losses, color=navy, marker='o', markersize=4, linewidth=2)
    best = bank['best_recorded_validation']
    ax.scatter([best['step']/1000], [best['ce']], color=teal, s=65, zorder=3)
    ax.annotate('Best recorded: 0.538 at 55K', (55, best['ce']), (37, .64),
                arrowprops={'arrowstyle': '-', 'color': teal}, color=teal, weight='bold')
    ax.set_title('Biobank validation improves, then deteriorates', loc='left', weight='bold', pad=20)
    ax.text(0, 1.02, '268,491 training / 1,000 validation people · one selected trajectory',
            transform=ax.transAxes, color='#536574', fontsize=10)
    ax.set_xlabel('Optimizer updates (thousands)')
    ax.set_ylabel('Reported validation CE (nats/token)')
    ax.grid(alpha=.15)
    for suffix in ['svg', 'png']:
        fig.savefig(args.output / f'biobank-ntp.{suffix}', dpi=180,
                    metadata={'Date': None} if suffix=='svg' else None)
    plt.close(fig)

    holdouts = json.loads((args.results / 'biobank-holdouts.json').read_text())
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for key, label, color in [('locked_test_reserve', 'Locked test (n=3,817)', navy),
                              ('within_family_holdout', 'Within-family holdout (n=1,503)', gray),
                              ('pooled', 'Pooled (n=5,320)', teal)]:
        ax.plot([r['optimizer_update']/1000 for r in holdouts['checkpoint_metrics']],
                [r['canonical_target_ce_nats_per_token'][key] for r in holdouts['checkpoint_metrics']],
                color=color, marker='o', linewidth=2, label=label)
    ax.set_title('The preselected checkpoint generalizes to 5,320 new people',
                 loc='left', weight='bold', fontsize=13, pad=20)
    ax.text(0, 1.02, '55K checkpoint fixed before evaluation · unchanged tokenizer · no refitting',
            transform=ax.transAxes, color='#536574', fontsize=10)
    ax.set_xlabel('Checkpoint optimizer updates (thousands)')
    ax.set_ylabel('Reported holdout CE (nats/token)')
    ax.set_xticks([55,77,99]); ax.legend(frameon=False); ax.grid(alpha=.15)
    for suffix in ['svg', 'png']:
        fig.savefig(args.output / f'biobank-holdouts.{suffix}', dpi=180,
                    metadata={'Date': None} if suffix=='svg' else None)
    plt.close(fig)


if __name__ == '__main__':
    main()
