"""Regression tests for navigation sample conversion and supported CLI options."""

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tool'))
import convert_common as common
import convert_vln as vln
import convert_dagger as dagger


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, vln, 'MAX_HISTORY', 8)
        self.addCleanup(setattr, dagger, 'MAX_HISTORY', 8)

    def frames(self, video, count, rgb=False):
        folder = self.root / video / ('rgb' if rgb else '')
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f'{i:03d}.jpg').touch()
        return folder

    def episode(self, **kwargs):
        episode = dict(episode_id=1, video_id='scene_1', instruction=['go'],
                       actions=[-1, 1, 2, 3])
        episode.update(kwargs)
        return episode

    def test_nested_paths_and_both_image_roots(self):
        self.frames('images/nested/scene_1', 4, rgb=True)
        ep = self.episode(video='images/nested/scene_1', instructions=['go'])
        a, b = [], []
        random.seed(1); vln.convert_episode_vln(ep, self.root, a)
        random.seed(1); vln.convert_episode_vln(ep, self.root / 'images', b)
        self.assertEqual(a, b)
        self.assertTrue(a)
        c = []
        random.seed(1); dagger.convert_episode_atomic(ep, self.root, c)
        self.assertEqual(a, c)
        for sample in a:
            self.assertEqual(sample['messages'][0]['value'].count('<image>'), len(sample['images']))

    def test_json_whitespace_bom_jsonl_and_invalid_array(self):
        path = self.root / 'input.json'
        for prefix in ('', '\n  ', '\ufeff\n '):
            path.write_text(prefix + json.dumps([self.episode()], indent=2))
            self.assertEqual(common.load_episodes(path), [self.episode()])
        path.write_text('\ufeff\n' + json.dumps(self.episode()) + '\n' + json.dumps(self.episode()))
        self.assertEqual(len(common.load_episodes(path)), 2)
        path.write_text('[1,2]')
        with self.assertRaises(ValueError): common.load_episodes(path)

    def test_missing_frames_are_not_padded_for_movement(self):
        self.frames('scene_1', 1)
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(), self.root))
        output = []
        with self.assertWarns(UserWarning):
            vln.convert_episode_vln(self.episode(), self.root, output)
        self.assertEqual(output, [])

    def test_action_sequences_reject_strings_objects_and_scalars(self):
        self.frames('scene_1', 4)
        for field in ('actions', 'executed_actions'):
            for value in ('123', {'1': 1, '2': 2, '3': 3}, 123, True):
                with self.subTest(field=field, value=value), self.assertWarnsRegex(UserWarning, 'nonempty list'):
                    output = []
                    vln.convert_episode_vln(self.episode(**{field: value}), self.root, output)
                    self.assertEqual(output, [])

    def test_frame_gaps_and_duplicate_indices_are_rejected(self):
        folder = self.frames('scene_1', 4)
        (folder / '001.jpg').unlink()
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(), self.root))
        (folder / '001.jpg').touch(); (folder / '001.png').touch()
        with self.assertRaises(ValueError): common.list_episode_images(self.root, 'scene_1')

    def test_only_terminal_stop_can_duplicate_a_frame(self):
        self.frames('scene_1', 4)
        actions, images = common.prep_actions_and_images(self.episode(actions=[-1, 1, 2, 3, 0, 0]), self.root)
        self.assertEqual(actions[-1], 'stop')
        self.assertEqual(images[-2], images[-1])
        self.assertEqual(len(images), 5)
        self.assertIsNone(common.try_combine('stop', 'stop'))
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(actions=[-1, 1, 0, 2]), self.root))

    def test_numeric_string_actions_and_history_one(self):
        self.assertEqual(common.to_action_text('1'), 'forward 25 cm')
        self.assertEqual(common.to_action_text('0'), 'stop')
        self.assertEqual(common.uniform_sample_with_ends(list(range(4)), 1), [3])
        with self.assertRaises(ValueError): common.uniform_sample_with_ends([1], 0)

    def test_atomic_labels_use_execution_and_final_stop_frame(self):
        self.frames('scene_1', 4)
        ep = self.episode(actions=[-1, 1, 1, 1], executed_actions=[2, 1, 3, 0])
        output = []
        dagger.convert_episode_atomic(ep, self.root, output)
        self.assertTrue(output)
        self.assertIn('turn left 15 degree', output[0]['messages'][1]['value'])
        self.assertIn('stop', output[-1]['messages'][1]['value'])
        self.assertTrue(output[-1]['images'][-1].endswith('003.jpg'))

    def test_cli_creates_one_atomic_output_without_model_groups(self):
        self.frames('scene_1', 4)
        annotations = self.root / 'input.json'
        annotations.write_text('\ufeff\n ' + json.dumps([self.episode()]))
        output = self.root / 'new' / 'output.json'
        result = subprocess.run([sys.executable, str(ROOT / 'tool/convert_dagger.py'),
                                 '--annotation_path', str(annotations), '--image_root', str(self.root),
                                 '--output_path', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(output.read_text()))
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_standard_samples_match_original_converter(self):
        fixture = json.loads((ROOT / 'tests/fixtures/conversion_reference.json').read_text())
        episodes = fixture['episodes']
        for episode in episodes:
            self.frames(episode['video_id'], fixture['frame_count'])
        random.seed(fixture['seed']); result = []
        for ep in episodes:
            vln.convert_episode_vln(ep, self.root, result)
        common.relativize_image_paths(result, self.root)
        self.assertEqual(fixture['expected']['vln'], result)
        for sample in result:
            self.assertLessEqual(len(sample['images']), 9)
            self.assertGreaterEqual(len(sample['images']), 2)
            self.assertNotIn('image_roles', sample)
            self.assertNotIn('local observation', sample['messages'][0]['value'])
            self.assertEqual(sample['messages'][0]['value'].count('<image>'), len(sample['images']))

    def test_history_plus_current_indices(self):
        self.assertEqual(vln.sample_history_plus_current(list(range(12))),
                         [0, 1, 3, 4, 6, 7, 9, 10, 11])
        self.assertEqual(vln.sample_history_plus_current([0]), [0])

    def test_instruction_objects_are_read_as_text_and_bad_types_are_skipped(self):
        self.frames('scene_1', 4)
        output = []
        vln.convert_episode_vln(self.episode(instruction={'instruction_text': 'walk to the door'}), self.root, output)
        self.assertTrue(output)
        self.assertIn('walk to the door', output[0]['messages'][0]['value'])
        with self.assertWarns(UserWarning):
            self.assertEqual(common.get_instructions(self.episode(instruction=42)), [])

    def test_removed_modes_and_task_selection_are_rejected(self):
        output = self.root / 'output.json'
        output.write_text('previous output')
        cases = [('convert_dagger.py', '--mode', mode) for mode in ('merged', 'both')]
        cases += [('convert_vln.py', '--task_type', task) for task in ('idm', 'ffs')]
        for converter, flag, value in cases:
            with self.subTest(converter=converter, value=value):
                result = subprocess.run([sys.executable, str(ROOT / 'tool' / converter),
                    '--annotation_path', str(self.root / 'input.json'), '--image_root', str(self.root),
                    '--output_path', str(output), flag, value], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(value, result.stderr)
                self.assertEqual(output.read_text(), 'previous output')

    def test_cli_can_export_portable_image_paths(self):
        self.frames('scene_1', 4)
        annotations = self.root / 'input.json'
        annotations.write_text(json.dumps([self.episode(model_call_groups=[dict(call_step_id=0, actions=[1,2,3])])]))
        for converter in ('convert_vln.py', 'convert_dagger.py'):
            with self.subTest(converter=converter):
                output = self.root / converter / 'out.json'
                result = subprocess.run([sys.executable, str(ROOT / 'tool' / converter),
                    '--annotation_path', str(annotations), '--image_root', str(self.root),
                    '--output_path', str(output), '--relative_to', str(self.root)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for path in output.parent.glob('*.json'):
                    samples = json.loads(path.read_text())
                    self.assertTrue(samples)
                    for sample in samples:
                        for image in sample['images']:
                            self.assertFalse(Path(image).is_absolute())
                            self.assertTrue((self.root / image).is_file())

    def test_relative_export_rejects_images_outside_root(self):
        samples = [{'images': [str(self.root.parent / 'outside.jpg')]}]
        with self.assertRaisesRegex(ValueError, 'outside --relative_to'):
            common.relativize_image_paths(samples, self.root)

    def test_failed_serialization_keeps_existing_output(self):
        output = self.root / 'output.json'
        output.write_text('previous output')
        with self.assertRaises(TypeError):
            common.write_samples(output, [{'not_json': object()}])
        self.assertEqual(output.read_text(), 'previous output')
        self.assertEqual(list(self.root.iterdir()), [output])


if __name__ == '__main__': unittest.main()
