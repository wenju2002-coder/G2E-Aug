"""Lightweight evaluator tests; no weights, data or scientific runtime needed."""
import unittest
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_bucket as evaluator


class EvaluatorTests(unittest.TestCase):
    def test_class_unknown_default(self):
        self.assertFalse(evaluator.make_parser().parse_args([]).use_rxn_class)

    def test_class_flag_and_values(self):
        for flags, expected in [(['--use_rxn_class'], True),
                                (['--use_rxn_class', 'True'], True),
                                (['--use_rxn_class', 'False'], False)]:
            self.assertEqual(evaluator.make_parser().parse_args(flags).use_rxn_class, expected)

    def test_bucket_boundaries(self):
        for count, expected in [(0, 'Few-shot'), (19, 'Few-shot'),
                                (20, 'Medium-shot'), (100, 'Medium-shot'),
                                (101, 'Many-shot')]:
            self.assertEqual(evaluator.bucket_for_edits(['edit', 'Terminate'], {'edit': count}),
                             (expected, count))

    def test_unseen_edit(self):
        self.assertEqual(evaluator.bucket_for_edits(['missing'], Counter()), ('Few-shot', 0))

    def test_terminal_tuple(self):
        self.assertTrue(evaluator.is_terminate(('Terminate',)))
        self.assertEqual(evaluator.bucket_for_edits([('Terminate',)], {}), ('Many-shot', None))

    def test_class_number_and_name(self):
        self.assertEqual(evaluator.class_info(2, 'uspto_50k'), (2, 3, 'C-C bond formation'))
        self.assertEqual(evaluator.class_info(6, 'uspto_50k'), (6, 7, 'Reductions'))

    def test_invalid_class(self):
        for value in [10, -1, True, '2']:
            with self.assertRaises(ValueError):
                evaluator.class_info(value, 'uspto_50k')

    def test_checkpoint_must_be_explicit(self):
        args = evaluator.make_parser().parse_args([])
        with self.assertRaises(ValueError):
            evaluator.resolve_checkpoint(args, Path.cwd())

    def test_archive_checkpoint_path(self):
        args = evaluator.make_parser().parse_args(['--use_rxn_class', 'True',
                    '--experiments', 'run', '--epoch', 'epoch_124.pt'])
        expected = Path.cwd() / 'experiment/uspto_50k/with_rxn_class/run/epoch_124.pt'
        self.assertEqual(evaluator.resolve_checkpoint(args, Path.cwd()), expected.resolve())

    def test_checkpoint_options_are_exclusive(self):
        args = evaluator.make_parser().parse_args(['--checkpoint', 'model.pt', '--epoch', 'other.pt'])
        with self.assertRaises(ValueError):
            evaluator.resolve_checkpoint(args, Path.cwd())

    def test_invalid_beams_keep_rank(self):
        paths = [{'final_smi': None}, {'final_smi': 'final_smi_unmapped'}, {'final_smi': 'CC.O'}]
        predictions, match, hits = evaluator.score_paths(paths, 'O.CC', 10)
        self.assertEqual(match, 3)
        self.assertEqual([p['rank'] for p in predictions], [1, 2, 3])
        self.assertEqual(hits, {'top1': 0, 'top3': 1, 'top5': 1, 'top10': 1})

    def test_fragment_multiplicity_is_ignored(self):
        _, rank, _ = evaluator.score_paths([{'final_smi': 'CC.O.O'}], 'CC.O', 10)
        self.assertEqual(rank, 1)

    def test_failed_predictions_remain_in_denominator(self):
        stats = defaultdict(evaluator.new_stats)
        _, _, hits = evaluator.score_paths([], 'CC', 10)
        evaluator.update_stats(stats, 2, 'Few-shot', hits)
        self.assertEqual(stats[('All', 'All')]['total'], 1)
        self.assertEqual(stats[('All', 'All')]['top1'], 0)


if __name__ == '__main__':
    unittest.main()
