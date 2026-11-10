"""Count motif emissions in canonical development streams, never overlapping windows."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reticular_diploid.data.public_diploid_corpus import load_corpus
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def count_motifs(canonical, motif_entries):
    ids = canonical['target_ids']
    ptr = canonical['target_indptr']
    split = canonical['participant_split']
    people = canonical['participant_indices']
    if (len(ptr) != len(split) + 1 or ptr[0] != 0 or ptr[-1] != len(ids)
            or np.any(np.diff(ptr) < 0) or np.any(ids < 0)
            or set(split.tolist()) != {0, 1} or len(set(people.tolist())) != len(split)):
        raise ValueError('Expected unique canonical training/validation streams only')
    counts = {name: np.zeros(motif_entries, dtype=np.int64)
              for name in ['training', 'validation']}
    for row, role in enumerate(split):
        stream = ids[ptr[row]:ptr[row + 1]]
        motif_ids = stream[stream < motif_entries]
        counts[['training', 'validation'][int(role)]] += np.bincount(
            motif_ids, minlength=motif_entries)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--tokenizer-config', type=Path, default=Path('tokenizer/config.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError('Use a new output directory')
    receipt, canonical, _ = load_corpus(args.corpus)
    config = json.loads(args.tokenizer_config.read_text())
    if receipt['tokenizer_sha256'] != config['files']['vocab.npz']['sha256']:
        raise ValueError('Corpus and dictionary differ')
    counts = count_motifs(canonical, config['motif_entries'])
    args.output.mkdir(parents=True)
    for split, values in counts.items():
        with (args.output / f'motif-frequency-{split}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(['token_id', 'occurrences'])
            writer.writerows(enumerate(values.tolist()))
    (args.output / 'motif-frequency.json').write_text(json.dumps({
        'code_commit': commit, 'tokenizer_sha256': receipt['tokenizer_sha256'],
        'canonical_sha256': receipt['canonical_sha256'], 'locked_test_used': False,
        'counting': 'Each canonical stream once; D1/D2 separate; no residuals or framing',
        'splits': {name: {'occurrences': int(values.sum()),
                         'observed_types': int(np.count_nonzero(values)),
                         'available_types': len(values)} for name, values in counts.items()},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
