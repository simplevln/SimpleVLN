import ast
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vln'))
from result_store import append_result, load_result_rows, result_directory_lock


def row(**kwargs):
    result = dict(scene_id='scene', episode_id=1, episode_instruction='向前走',
                  trial_id=0, trial_total=2, success=1, spl=.5, os=1, ne=1,
                  steps=3, path_length=4, ndtw=.8, sdtw=.8)
    result.update(kwargs)
    return result


def summary(rows, pass_k=2):
    tree = ast.parse((ROOT / 'vln/eval.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'summarize_results')
    ns = {'np': SimpleNamespace(isfinite=math.isfinite)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<summary>', 'exec'), ns)
    return ns['summarize_results'](rows, pass_k)


class ResultStoreTests(unittest.TestCase):
    def test_scope_deduplication_and_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as path:
            for value in (row(), row(success=0), row(trial_total=1), row(episode_id=2),
                          {'scene_id': 'scene', 'episode_id': 3, 'episode_instruction': 'go'},
                          row(trial_id=1, path_length=None), ['bad'], {'pass@2': 1}):
                append_result(path, value)
            rows = load_result_rows(path, pass_k=2,
                                    trajectories={('scene', '1', '向前走')},
                                    required_metrics=('path_length',))
            self.assertEqual(len(rows), 1)
            self.assertEqual(next(iter(rows.values()))['success'], 0)

    def test_append_after_truncated_utf8_or_missing_newline(self):
        with tempfile.TemporaryDirectory() as path:
            output = Path(path) / 'result.json'
            output.write_bytes(b'{"episode_instruction":"\xe5\x90')
            append_result(path, row())
            self.assertEqual(len(load_result_rows(path)), 1)
            output.write_text(json.dumps(row()))
            append_result(path, row(trial_id=1))
            self.assertEqual(len(load_result_rows(path)), 2)

    def test_exclusive_directory_lock(self):
        with tempfile.TemporaryDirectory() as path:
            with result_directory_lock(path):
                with self.assertRaises(RuntimeError):
                    with result_directory_lock(path):
                        pass
            with result_directory_lock(path):
                pass

    def test_summary_core_statistics(self):
        rows = [row(), row(trial_id=1, success=0, spl=0, os=0, ne=5,
                           path_length=6, ndtw=.4, sdtw=0)]
        result = summary(rows)
        for name, expected in {'sucs_all': .5, 'spls_all': .25, 'oss_all': .5,
                               'ones_all': 3, 'pass@2': 1, 'avg2': .5,
                               'pass_spl@2': .5, 'pass_os@2': 1, 'pass_ne@2': 1,
                               'length': 5, 'sdtw': .4}.items():
            self.assertEqual(result[name], expected, name)
        self.assertAlmostEqual(result['ndtw'], .6)

    def test_nonfinite_core_values_are_retained_but_optional_best_is_valid(self):
        result = summary([row(spl=float('nan'), ne=float('inf'), ndtw=float('nan')),
                          row(trial_id=1, success=0, ndtw=.4)])
        self.assertTrue(math.isnan(result['spls_all']))
        self.assertTrue(math.isinf(result['ones_all']))
        self.assertEqual(result['sucs_all'], .5)
        self.assertEqual(result['pass_ndtw@2'], .4)


if __name__ == '__main__':
    unittest.main()
