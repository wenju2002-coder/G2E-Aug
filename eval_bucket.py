"""Frequency-stratified evaluation for the Graph2Edits-Aug archive.

This adapter uses the supplied Graph2Edits-Aug source and saved configuration.
Source hashes are recorded at inference time. Python 3.7+.

This script does not change the model, beam ranking, or training. It preserves
the upstream exact-match convention and keeps invalid candidates in their ranks.
Only load trusted checkpoints/joblib files: those formats can execute code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import numbers
import platform
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_URL = 'https://github.com/wenju2002-coder/G2E-Aug'
CHECKED_COMMIT = None
TOP_K = (1, 3, 5, 10)
BUCKETS = ('Many-shot', 'Medium-shot', 'Few-shot')
CLASS_NAMES = (
    'Heteroatom alkylation and arylation', 'Acylation and related processes',
    'C-C bond formation', 'Heterocycle formation', 'Protections',
    'Deprotections', 'Reductions', 'Oxidations',
    'Functional group interconversion', 'Functional group addition',
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('true', '1', 'yes'):
        return True
    if str(value).lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Use True or False, or omit the value for True.')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='uspto_50k', type=str.lower)
    parser.add_argument('--use_rxn_class', '--use-rxn-class', nargs='?', const=True,
                        default=False, type=parse_bool)
    parser.add_argument('--code_root', '--code-root', type=Path,
                        default=Path(__file__).resolve().parent,
                        help='Graph2Edits-Aug root containing models/ and utils/.')
    parser.add_argument('--checkpoint', type=Path,
                        help='Explicit checkpoint path, relative to code_root or absolute.')
    parser.add_argument('--experiments', help='Run subfolder under experiment/<dataset>/<setting>/.')
    parser.add_argument('--epoch', help='Explicit filename, e.g. epoch_86.pt or epoch_124.pt.')
    parser.add_argument('--test_file', '--test-file', type=Path,
                        help='Default: data/<dataset>/test/test.file.kekulized.')
    parser.add_argument('--train_file', '--train-file', type=Path,
                        help='Default: data/<dataset>/train/train.file.kekulized.')
    parser.add_argument('--frequency_map', '--frequency-map', type=Path,
                        help='Optional trusted edit->count joblib map, e.g. the Aug comparison map.')
    parser.add_argument('--output_dir', '--output-dir', type=Path,
                        help='NEW output directory; existing directories are never overwritten.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--beam_size', '--beam-size', type=int, default=10)
    parser.add_argument('--step_beam_size', '--step-beam-size', type=int, default=10)
    parser.add_argument('--max_steps', '--max-steps', type=int, default=9)
    parser.add_argument('--expected_test_reactions', '--expected-test-reactions', type=int,
                        help='Optional strict check; use 5004 for the manuscript protocol.')
    parser.add_argument('--max_reactions', '--max-reactions', type=int,
                        help='Smoke test only. A prefix run is explicitly labelled incomplete.')
    return parser


def resolve_input(root, value, default=None):
    path = Path(value if value is not None else default)
    return (path if path.is_absolute() else root / path).resolve()


def resolve_checkpoint(args, root):
    if args.checkpoint is not None:
        if args.experiments is not None or args.epoch is not None:
            raise ValueError('Use --checkpoint OR both --experiments and --epoch, not both forms.')
        return resolve_input(root, args.checkpoint)
    if not args.experiments or not args.epoch:
        raise ValueError('Specify --checkpoint, or BOTH --experiments and --epoch. No epoch is guessed.')
    if Path(args.epoch).name != args.epoch:
        raise ValueError('--epoch must be a filename. Use --checkpoint for an arbitrary path.')
    setting = 'with_rxn_class' if args.use_rxn_class else 'without_rxn_class'
    return (root / 'experiment' / args.dataset / setting / args.experiments / args.epoch).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def is_terminate(edit):
    return edit == 'Terminate' or (
        isinstance(edit, (tuple, list)) and len(edit) > 0 and edit[0] == 'Terminate')


def freeze_edit(edit):
    return tuple(freeze_edit(value) for value in edit) if isinstance(edit, (tuple, list)) else edit


def build_frequency_map(records):
    """Count edit occurrences, not reactions containing an edit; training only."""
    counter = Counter()
    for record in records:
        for edit in record.edits:
            if not is_terminate(edit):
                counter[freeze_edit(edit)] += 1
    if not counter:
        raise ValueError('No non-terminal training edits found; cannot define frequency buckets.')
    return counter


def validate_frequency_map(mapping):
    if not hasattr(mapping, 'items'):
        raise ValueError('Frequency map must be an edit->nonnegative integer count mapping.')
    counter = Counter()
    for edit, count in mapping.items():
        if is_terminate(edit):
            continue
        if isinstance(count, bool) or not isinstance(count, numbers.Real) or not math.isfinite(float(count)):
            raise ValueError('Invalid frequency for edit {!r}: {!r}'.format(edit, count))
        if count < 0 or int(count) != count:
            raise ValueError('Counts must be nonnegative integers, not probabilities or weights.')
        counter[freeze_edit(edit)] = int(count)
    if not counter or not any(counter.values()):
        raise ValueError('Frequency map is empty or contains only zero counts.')
    return counter


def frequency_hash(frequencies):
    """Serialization-independent signature for comparing baseline/Aug counters."""
    payload = sorted((repr(edit), int(count)) for edit, count in frequencies.items())
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest()


def bucket_for_edits(edits, frequencies):
    counts = [frequencies.get(freeze_edit(edit), 0) for edit in edits if not is_terminate(edit)]
    rarest = min(counts) if counts else None
    if rarest is None or rarest > 100:
        return 'Many-shot', rarest
    return ('Medium-shot' if rarest >= 20 else 'Few-shot'), rarest


def class_info(value, dataset):
    if value is None:
        return -1, '', 'Unspecified'
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError('Reaction class must be a preprocessed integer index, received {!r}.'.format(value))
    index = int(value)
    if dataset == 'uspto_50k' and not 0 <= index < 10:
        raise ValueError('USPTO-50K processed classes must be 0-9, not raw CSV classes 1-10.')
    return index, index + 1, CLASS_NAMES[index] if 0 <= index < 10 else 'Class {}'.format(index + 1)


def json_value(value):
    if hasattr(value, 'item'):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return str(value)


def canonical_reference(smiles, Chem):
    """Exactly the reference normalization used by upstream eval.py."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Cannot parse reference reactants.')
    for atom in molecule.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    molecule = Chem.MolFromSmiles(Chem.MolToSmiles(molecule))
    if molecule is None:
        raise ValueError('Cannot normalize reference reactants.')
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def score_paths(paths, reference, beam_size):
    """Do NOT filter/deduplicate/reorder beams or re-canonicalize predicted SMILES."""
    truth = set(reference.split('.'))
    predictions = []
    first_match = None
    for rank, path in enumerate(paths[:beam_size], 1):
        smiles = path.get('final_smi')
        valid = isinstance(smiles, str) and smiles not in ('', 'final_smi_unmapped')
        correct = valid and set(smiles.split('.')) == truth
        if correct and first_match is None:
            first_match = rank
        predictions.append({
            'rank': rank, 'final_smi': smiles if valid else None,
            'probability': json_value(path.get('prob')),
            'prediction_is_correct': bool(correct),
            'rxn_actions': [str(action) for action in path.get('rxn_actions', [])],
            'edits_prob': json_value(path.get('edits_prob', [])),
        })
    hits = {'top{}'.format(k): int(first_match is not None and first_match <= k) for k in TOP_K}
    return predictions, first_match, hits


def new_stats():
    return {'total': 0, **{'top{}'.format(k): 0 for k in TOP_K}}


def update_stats(stats, class_index, bucket, hits):
    for key in ((class_index, bucket), ('All', bucket), ('All', 'All')):
        stats[key]['total'] += 1
        for metric, hit in hits.items():
            stats[key][metric] += int(hit)


def summary_rows(stats, dataset):
    classes = sorted(key[0] for key in stats if isinstance(key[0], int))
    classes = sorted(set(classes))
    rows = []
    for section, indices, buckets in (
        ('by_reaction_class', classes, BUCKETS),
        ('overall', ['All'], BUCKETS), ('aggregate', ['All'], ['All']),
    ):
        for index in indices:
            for bucket in buckets:
                values = stats.get((index, bucket), new_stats())
                if index == 'All':
                    number, name = 'All', 'All reaction classes'
                else:
                    _, number, name = class_info(None if index == -1 else index, dataset)
                n = values['total']
                row = {'section': section, 'reaction_class': index,
                       'class_number_1_based': number, 'class_name': name,
                       'bucket': bucket, 'total_reactions': n}
                for k in TOP_K:
                    key = 'top{}'.format(k)
                    row['correct_' + key] = values[key]
                    row[key] = round(100.0 * values[key] / n, 6) if n else ''
                rows.append(row)
    return rows


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_reports(output, epoch_tag, stats, metadata):
    rows = summary_rows(stats, metadata['dataset'])
    with (output / ('bucket_eval_' + epoch_tag + '.csv')).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Graph2Edits frequency-stratified evaluation', '',
        '- Status: ' + metadata['status'],
        '- Full test set: ' + str(metadata['full_test_set']),
        '- Checkpoint: `' + metadata['checkpoint']['path'] + '`',
        '- Checkpoint SHA-256: `' + metadata['checkpoint']['sha256'] + '`',
        '- Use reaction class: ' + str(metadata['use_rxn_class']),
        '- Beam / step beam / max steps: {beam_size} / {step_beam_size} / {max_steps}'.format(**metadata),
        '- Evaluated / available test reactions: {evaluated_reactions} / {available_test_reactions}'.format(**metadata),
        '- Buckets: minimum NON-terminal training edit count; >100 / 20-100 / <20.',
        '- Class index 0-9 corresponds to standard class number 1-10.',
        '- Accuracy is in percent; CSV also contains exact integer success counts.', '',
        '| Section | Class number | Class name | Bucket | N | Top-1 | Top-3 | Top-5 | Top-10 |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        rates = ['{:.4f}%'.format(row['top{}'.format(k)]) if row['total_reactions'] else '-' for k in TOP_K]
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            row['section'], row['class_number_1_based'], row['class_name'], row['bucket'],
            row['total_reactions'], ' | '.join(rates)))
    text = '\n'.join(lines) + '\n'
    (output / ('bucket_eval_' + epoch_tag + '.md')).write_text(text, encoding='utf-8')
    print(text)


def load_runtime(root):
    if not (root / 'models' / 'graph2edits.py').is_file() or not (root / 'utils').is_dir():
        raise ValueError('Invalid --code_root. Copy this script into the Graph2Edits-Aug root, '
                         'or pass --code_root /path/to/Graph2Edits.')
    sys.path.insert(0, str(root))
    import joblib
    import torch
    from rdkit import Chem, RDLogger, rdBase
    from tqdm import tqdm
    from models import Graph2Edits, BeamSearch
    from utils.mol_features import ATOM_FDIM
    for model_class in (Graph2Edits, BeamSearch):
        source = Path(inspect.getfile(model_class)).resolve()
        if root not in source.parents:
            raise ValueError('Imported the wrong code release: ' + str(source))
    RDLogger.DisableLog('rdApp.*')
    return joblib, torch, Chem, rdBase, tqdm, Graph2Edits, BeamSearch, ATOM_FDIM


def run(args):
    root = args.code_root.resolve()
    if args.frequency_map is not None and args.train_file is not None:
        raise ValueError('Use --frequency_map OR --train_file, not both sources.')
    if args.beam_size < 10:
        raise ValueError('--beam_size must be >=10 to report Top-10.')
    if args.step_beam_size <= 0 or args.max_steps <= 0:
        raise ValueError('Step beam size and max steps must be positive.')
    if args.max_reactions is not None and args.max_reactions <= 0:
        raise ValueError('--max_reactions must be positive.')
    checkpoint_path = resolve_checkpoint(args, root)
    test_path = resolve_input(root, args.test_file, 'data/{}/test/test.file.kekulized'.format(args.dataset))
    frequency_source = resolve_input(root, args.frequency_map) if args.frequency_map else resolve_input(
        root, args.train_file, 'data/{}/train/train.file.kekulized'.format(args.dataset))
    for path in (checkpoint_path, test_path, frequency_source):
        if not path.is_file():
            raise FileNotFoundError('Missing input: {}. Run original preprocess.py for train/test first; '
                                    'or supply the matching file explicitly.'.format(path))
    joblib, torch, Chem, rdBase, tqdm, Graph2Edits, BeamSearch, atom_fdim = load_runtime(root)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu.')
    # Legacy torch lacks weights_only; modern torch defaults may reject trusted custom Vocab objects.
    load_kwargs = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False
    checkpoint = torch.load(str(checkpoint_path), **load_kwargs)
    if not isinstance(checkpoint, dict) or not all(key in checkpoint for key in ('state', 'saveables')):
        raise ValueError('Expected Graph2Edits-Aug checkpoint with state and saveables.')
    augmented = ('bond_break_linear', 'lg_gcn_', 'bond_logit_adjustment', 'atom_logit_adjustment')
    if not any(key.startswith('bond_break_linear.') for key in checkpoint['state']):
        raise ValueError('This is a baseline checkpoint. Use Graph2Edits-Aug weights.')
    config = checkpoint['saveables']['config']
    expected_dim = atom_fdim + (10 if args.use_rxn_class else 0)
    if config.get('n_atom_feat') != expected_dim:
        raise ValueError('Checkpoint n_atom_feat={} disagrees with --use_rxn_class={} '
                         '(expected {}). Check setting/weights/code.'.format(
                             config.get('n_atom_feat'), args.use_rxn_class, expected_dim))
    model = Graph2Edits(**checkpoint['saveables'], device=device)
    if not hasattr(model, 'bond_break_linear') or not hasattr(model, 'lg_gcn_in'):
        raise ValueError('--code_root must point to the supplied Graph2Edits-Aug implementation.')
    model.load_state_dict(checkpoint['state'], strict=True)
    model.to(device)
    model.eval()
    records = joblib.load(str(test_path))
    if not records:
        raise ValueError('Empty test data.')
    if args.expected_test_reactions is not None and len(records) != args.expected_test_reactions:
        raise ValueError('Expected {} test reactions, found {}. Do not silently change the denominator.'.format(
            args.expected_test_reactions, len(records)))
    if args.frequency_map:
        frequencies = validate_frequency_map(joblib.load(str(frequency_source)))
        training_count = None
    else:
        train_records = joblib.load(str(frequency_source))
        training_count = len(train_records)
        frequencies = build_frequency_map(train_records)
        del train_records
    limit = min(args.max_reactions, len(records)) if args.max_reactions else len(records)
    for record in records[:limit]:
        index, _, _ = class_info(getattr(record, 'rxn_class', None), args.dataset)
        if args.use_rxn_class and not 0 <= index < 10:
            raise ValueError('Class-conditioned evaluation requires a valid class label per reaction.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir.resolve() if args.output_dir else checkpoint_path.parent / (
        'bucket_eval_' + checkpoint_path.stem + '_' + stamp)
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'status': 'running', 'model': 'Graph2Edits-Aug', 'dataset': args.dataset,
        'started_utc': stamp, 'use_rxn_class': args.use_rxn_class,
        'beam_size': args.beam_size, 'step_beam_size': args.step_beam_size, 'max_steps': args.max_steps,
        'available_test_reactions': len(records), 'requested_reactions': limit,
        'expected_test_reactions': args.expected_test_reactions,
        'evaluated_reactions': 0, 'full_test_set': False,
        'checkpoint': {'path': str(checkpoint_path), 'sha256': sha256_file(checkpoint_path)},
        'test_file': {'path': str(test_path), 'sha256': sha256_file(test_path)},
        'frequency_source': {'path': str(frequency_source), 'sha256': sha256_file(frequency_source),
                             'kind': 'supplied_map' if args.frequency_map else 'training_edit_occurrences',
                             'training_reactions': training_count},
        'frequency_map_content_sha256': frequency_hash(frequencies),
        'frequency_actions': len(frequencies), 'no_edit_reactions': 0,
        'invalid_candidates': 0, 'reactions_with_invalid_candidates': 0,
        'reactions_without_candidates': 0,
        'bucket_protocol': {'many': '>100', 'medium': '20-100 inclusive', 'few': '<20',
                            'unseen_edit_count': 0, 'only_terminate_policy': 'Many-shot'},
        'match_policy': 'upstream exact-match: canonical unmapped reference fragment set vs raw beam fragment set',
        'code_root': str(root), 'checked_upstream_commit': CHECKED_COMMIT, 'upstream_url': UPSTREAM_URL,
        'model_config': config,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'rdkit': rdBase.rdkitVersion, 'joblib': joblib.__version__,
                     'numpy': __import__('numpy').__version__},
        'source_sha256': {str(path.relative_to(root)): sha256_file(path) for path in sorted(
            list((root / 'models').glob('*.py')) + list((root / 'utils').glob('*.py')) +
            [root / 'prepare_data.py']) if path.is_file()},
        'evaluator_sha256': sha256_file(Path(__file__)),
    }
    write_json(output / 'metadata.json', metadata)
    joblib.dump(frequencies, str(output / 'action_freq_map.pkl'), compress=3)
    search = BeamSearch(model=model, step_beam_size=args.step_beam_size,
                        beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    stats = defaultdict(new_stats)
    fields = ['reaction_index', 'reaction_id', 'class_index_0_based', 'class_number_1_based',
              'class_name', 'bucket', 'min_edit_frequency', 'rxn_smi', 'ground_truth_reactants',
              'predictions_top10_json', 'first_match_rank', 'top1', 'top3', 'top5', 'top10',
              'model', 'use_rxn_class', 'checkpoint_sha256']
    try:
        with (output / 'predictions_top10.csv').open('w', encoding='utf-8', newline='') as csv_fp, \
                (output / 'predictions.jsonl').open('w', encoding='utf-8') as json_fp, \
                (output / 'pred_results.txt').open('w', encoding='utf-8') as raw_fp:
            writer = csv.DictWriter(csv_fp, fieldnames=fields)
            writer.writeheader()
            progress = tqdm(range(limit), desc='Graph2Edits-Aug buckets')
            for index in progress:
                record = records[index]
                class_index, class_number, class_name = class_info(getattr(record, 'rxn_class', None), args.dataset)
                bucket, rarest = bucket_for_edits(record.edits, frequencies)
                reactants, product = record.rxn_smi.split('>>')
                reference = canonical_reference(reactants, Chem)
                with torch.no_grad():
                    paths = search.run_search(prod_smi=product, max_steps=args.max_steps,
                                              rxn_class=record.rxn_class if args.use_rxn_class else None)
                predictions, first_match, hits = score_paths(paths, reference, args.beam_size)
                update_stats(stats, class_index, bucket, hits)
                row = {
                    'reaction_index': index, 'reaction_id': json_value(getattr(record, 'rxn_id', '')),
                    'class_index_0_based': class_index, 'class_number_1_based': class_number,
                    'class_name': class_name, 'bucket': bucket, 'min_edit_frequency': rarest,
                    'rxn_smi': record.rxn_smi, 'ground_truth_reactants': reference,
                    'predictions_top10_json': json.dumps([p['final_smi'] for p in predictions[:10]], ensure_ascii=False),
                    'first_match_rank': first_match if first_match is not None else '', **hits,
                    'model': 'Graph2Edits-Aug', 'use_rxn_class': args.use_rxn_class,
                    'checkpoint_sha256': metadata['checkpoint']['sha256'],
                }
                writer.writerow(row)
                json_fp.write(json.dumps({**row, 'true_edits': json_value(record.edits),
                                          'predictions': predictions}, ensure_ascii=False, allow_nan=False) + '\n')
                raw_fp.write('({}) {}\n'.format(index, record.rxn_smi))
                for pred in predictions:
                    actions = '|'.join('({};{})'.format(action, prob) for action, prob in zip(
                        pred['rxn_actions'], pred['edits_prob']))
                    raw_fp.write('{} prediction_is_correct:{} probability:{} {} {}\n'.format(
                        pred['rank'] - 1, pred['prediction_is_correct'], pred['probability'],
                        pred['final_smi'] or 'final_smi_unmapped', actions))
                raw_fp.write('\n')
                metadata['evaluated_reactions'] = index + 1
                metadata['no_edit_reactions'] += int(rarest is None)
                invalid = sum(pred['final_smi'] is None for pred in predictions)
                metadata['invalid_candidates'] += invalid
                metadata['reactions_with_invalid_candidates'] += int(invalid > 0)
                metadata['reactions_without_candidates'] += int(not predictions)
                if (index + 1) % 50 == 0:
                    for handle in (csv_fp, json_fp, raw_fp):
                        handle.flush()
                    write_json(output / 'metadata.json', metadata)
                progress.set_postfix({'top1': '{:.4f}'.format(stats[('All', 'All')]['top1'] / (index + 1))})
        total = stats[('All', 'All')]['total']
        if total != limit or sum(stats.get(('All', b), new_stats())['total'] for b in BUCKETS) != total:
            raise AssertionError('Bucket counts do not reconcile with evaluated reaction count.')
        metadata.update(status='complete', evaluated_reactions=total,
                        full_test_set=total == len(records) and args.max_reactions is None,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        metadata['bucket_sizes'] = {b: stats.get(('All', b), new_stats())['total'] for b in BUCKETS}
        if args.max_reactions is not None:
            metadata['run_purpose'] = 'limited evaluation / smoke test; not a manuscript result'
        write_reports(output, checkpoint_path.stem, stats, metadata)
        write_json(output / 'metadata.json', metadata)
    except BaseException as exc:
        metadata.update(status='failed', full_test_set=False,
                        failure={'type': type(exc).__name__, 'message': str(exc)})
        write_json(output / 'metadata.json', metadata)
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    print('Saved new evaluation directory: ' + str(output))
    if metadata['invalid_candidates'] or metadata['reactions_without_candidates']:
        print('Diagnostic: {} invalid beam candidates and {} reactions without candidates. '
              'Their ranks/denominators were retained; inspect predictions.jsonl and software versions.'.format(
                  metadata['invalid_candidates'], metadata['reactions_without_candidates']))
    return output


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, 'Evaluation stopped: {}\n'.format(exc))


if __name__ == '__main__':
    main()
