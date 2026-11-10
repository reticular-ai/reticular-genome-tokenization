"""Regenerate editorial-style research figures from released aggregate measurements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np

PAPER, INK, PLUM, MUTED, TAUPE, GRID = (
    '#F7F4F0', '#24211F', '#4B2338', '#807971', '#B4A79C', '#E5DED7')


def rule(fig, y, left=.065, right=.95, color=GRID):
    fig.add_artist(Line2D([left, right], [y, y], transform=fig.transFigure,
                         color=color, linewidth=.8))


def page(title, subtitle, category, note):
    fig = plt.figure(figsize=(12, 7), facecolor=PAPER)
    fig.text(.065, .925, 'Reticular', family='DejaVu Serif', fontsize=22, color=INK)
    fig.text(.95, .935, category.upper(), ha='right', family='DejaVu Sans Mono',
             fontsize=9, color=MUTED)
    fig.text(.065, .815, title, family='DejaVu Serif', fontsize=25, color=INK)
    fig.text(.065, .758, subtitle, fontsize=11, color=MUTED)
    rule(fig, .71, color=INK)
    rule(fig, .10)
    fig.text(.065, .056, note, fontsize=8.5, color=MUTED, va='center')
    fig.text(.95, .045, 'reticular.com', family='DejaVu Sans Mono', fontsize=9,
             color=PLUM, ha='right')
    return fig


def chart(fig, left=.11):
    ax = fig.add_axes([left, .22, .95-left, .43], facecolor=PAPER)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(GRID)
    ax.tick_params(colors=MUTED, length=3, pad=7)
    ax.grid(axis='y', color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    return ax


def save(fig, output, name):
    for suffix in ['svg', 'png']:
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=180, metadata={'Date': None} if suffix == 'svg' else None)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': MUTED,
                         'axes.labelpad': 12, 'svg.hashsalt': 'reticular-editorial-v1'})
    data = json.loads((args.results / 'tokenization.json').read_text())
    val = data['validation']
    fig = page('Genetic differences, represented compactly',
               '2.76 Mb on chromosome 22 · 55 public validation individuals · exact reconstruction',
               '1000 Genomes / tokenization',
               'Counts, not compressed bytes. Allele entries are not serialized tokens.\nReference, catalog and dictionary are shared side information.')
    ax = chart(fig, left=.30)
    values = [val['residual_only_mean'], val['residual_only_mean']/2, val['mean']]
    labels = ['Residual-only grammar', 'Nonreference allele entries', 'Motif tokenizer']
    ax.barh([2, 1, 0], values, color=[TAUPE, '#D6CDC5', PLUM], height=.40)
    ax.set_yticks([2, 1, 0], labels, color=INK)
    ax.set_ylim(-.6, 2.6); ax.set_xlim(0, 12600)
    ax.grid(False); ax.grid(axis='x', color=GRID, linewidth=.7)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Mean count per validation individual')
    for y, value in zip([2, 1, 0], values):
        ax.text(value+180, y, f'{value:,.0f}', va='center', color=PLUM if y==0 else INK,
                fontsize=13, family='DejaVu Serif')
    save(fig, args.output, 'tokenization')

    fig = page('Common motifs, and a long tail',
               'Frequency of emitted motif tokens · independently ranked within each split',
               '1000 Genomes / motif use',
               'D1 and D2 counted separately; residuals and framing excluded.\nZero-frequency types omitted from log axes. This is not a fitted Zipf law.')
    ax = chart(fig)
    for split, color, people, style in [('training', PLUM, 400, '-'),
                                       ('validation', TAUPE, 55, '--')]:
        with (args.results / f'motif-frequency-{split}.csv').open() as handle:
            counts = np.array([int(row['occurrences']) for row in csv.DictReader(handle)])
        counts = np.sort(counts[counts > 0])[::-1]
        ax.loglog(np.arange(1,len(counts)+1), counts/counts.sum(), color=color,
                  linewidth=1.8, linestyle=style, label=f'{split.capitalize()} · n = {people:,}')
    ax.set_xlabel('Token rank within split'); ax.set_ylabel('Fraction of motif emissions')
    ax.legend(frameon=False, loc='lower left', fontsize=10)
    save(fig, args.output, 'motif-frequency')

    bank = json.loads((args.results / 'biobank-ntp.json').read_text())
    fig = page('Validation through training',
               '268,491 training individuals · 1,000 original validation individuals',
               'Biobank / historical trajectory',
               'One selected trajectory; validation was reused for model selection.\nReported canonical-target cross-entropy. Source: PR #359.')
    ax = chart(fig)
    ax.plot([r['step']/1000 for r in bank['curve']], [r['validation_ce'] for r in bank['curve']],
            color=PLUM, marker='o', markersize=3.5, linewidth=1.7)
    best = bank['best_recorded_validation']
    ax.annotate(f"Selected · {best['ce']:.3f} at {best['step']/1000:.0f}K",
                (best['step']/1000, best['ce']), (48, .655), color=PLUM, fontsize=11,
                arrowprops={'arrowstyle': '-', 'color': TAUPE, 'linewidth': .8})
    ax.set_xlabel('Optimizer updates (thousands)'); ax.set_ylabel('Validation CE · nats/token')
    save(fig, args.output, 'biobank-ntp')

    h = json.loads((args.results / 'biobank-holdouts.json').read_text())
    fig = page('Fixed checkpoints on new individuals',
               'The 55K reporting checkpoint was selected before this holdout evaluation',
               'Biobank / checkpoint comparison',
               'Unchanged model and tokenizer; no checkpoint reselection on these cohorts.\nSelected-person and original-training family exclusions apply. Source: PRs #372–373.')
    ax = chart(fig)
    for key, label, color, style, marker in [
        ('locked_test_reserve', 'Locked test · n = 3,817', TAUPE, '--', 's'),
        ('within_family_holdout', 'Within-family · n = 1,503', MUTED, ':', '^'),
        ('pooled', 'Pooled · n = 5,320', PLUM, '-', 'o')]:
        ax.plot([r['optimizer_update']/1000 for r in h['checkpoint_metrics']],
                [r['canonical_target_ce_nats_per_token'][key] for r in h['checkpoint_metrics']],
                color=color, linestyle=style, marker=marker, markersize=5, linewidth=1.6,
                label=label)
    ax.set_xticks([55, 77, 99]); ax.set_xlabel('Checkpoint updates (thousands)')
    ax.set_ylabel('Holdout CE · nats/token'); ax.legend(frameon=False, fontsize=9.5, loc='upper left')
    save(fig, args.output, 'biobank-holdouts')

    lengths = {r['cohort']: r for r in h['sequence_length_statistics']}
    chosen = next(r for r in h['checkpoint_metrics']
                  if r['optimizer_update'] == h['selection']['original_preselected_reporting_step'])
    scores = chosen['canonical_target_ce_nats_per_token']
    fig = page('Human variation, in hundreds of tokens',
               '2.76 Mb on chromosome 22 · 8,192-token vocabulary · preselected 55K checkpoint',
               'Biobank / held-out evaluation',
               'Full sequences include three framing tokens. CE is pooled by scored targets.\nSelected-person / family exclusions apply; see docs/biobank.md. Source: PRs #372–373.')
    for x, value, label, detail in [
        (.065, f"{lengths['pooled_selected_holdouts']['mean_sentence_tokens']:,.0f}",
         'tokens per individual', 'Mean full canonical sequence'),
        (.385, f"{scores['pooled']:.3f}", 'holdout cross-entropy', 'Nats per scored token'),
        (.705, f"{h['cohorts']['pooled']:,}", 'new individuals', 'Outside original train / validation')]:
        fig.text(x, .575, value, family='DejaVu Serif', fontsize=48, color=PLUM)
        fig.text(x, .515, label, fontsize=12, color=INK)
        fig.text(x, .472, detail, fontsize=9, color=MUTED)
    rule(fig, .415)
    fig.text(.065, .365, 'Reporting cohort', family='DejaVu Serif', fontsize=13)
    for x,label in [(.58,'Individuals'),(.77,'Mean tokens'),(.95,'CE')]:
        fig.text(x, .365, label, color=MUTED, fontsize=10, ha='right')
    for y,key,label in [(.285,'locked_test_reserve','Locked test'),
                        (.205,'within_family_holdout','Within-family holdout')]:
        fig.text(.065, y, label, family='DejaVu Serif', fontsize=14)
        fig.text(.58, y, f"{h['cohorts'][key]:,}", ha='right', fontsize=12)
        fig.text(.77, y, f"{lengths[key]['mean_sentence_tokens']:,.2f}", ha='right', fontsize=12)
        fig.text(.95, y, f"{scores[key]:.5f}", ha='right', fontsize=12, color=PLUM)
    save(fig, args.output, 'biobank-summary')


if __name__ == '__main__':
    main()
