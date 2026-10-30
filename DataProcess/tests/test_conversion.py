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
            vln.convert_episode_idm(self.episode(), self.root, output)
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
        idm_actions, idm_images = common.prep_actions_and_images(self.episode(), self.root, True)
        self.assertEqual(len(idm_images), len(idm_actions) + 1)
        self.assertNotEqual(idm_images[-1], idm_images[-2])
        self.assertIsNone(common.try_combine('stop', 'stop'))
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(actions=[-1, 1, 0, 2]), self.root))

    def test_numeric_string_actions_and_history_one(self):
        self.assertEqual(common.to_action_text('1'), 'forward 25 cm')
        self.assertEqual(common.to_action_text('0'), 'stop')
        self.assertEqual(common.uniform_sample_with_ends(list(range(4)), 1), [3])
        with self.assertRaises(ValueError): common.uniform_sample_with_ends([1], 0)

    def test_merged_labels_use_execution_and_stop_uses_final_frame(self):
        self.frames('images/scene_1', 7, rgb=True)
        ep = self.episode(video='images/scene_1', actions=[-1, 1, 1, 1, 2, 2, 3],
                          model_call_groups=[dict(call_step_id=0, actions=[1, 1, 1], model_output='forward 300 cm')])
        output = []
        dagger.convert_episode_merged(ep, self.root, output)
        self.assertEqual(output[0]['messages'][1]['value'], 'forward 75 cm')
        self.assertEqual(output[-1]['messages'][1]['value'], 'stop')
        self.assertTrue(output[-1]['images'][-1].endswith('006.jpg'))
        # An expert changed the first action; the new explicit group reflects it.
        ep.update(executed_actions=[2, 0], actions=[-1, 2], terminal_action=0,
                  model_call_groups=[dict(call_step_id=0, planned_actions=[1,1,1],
                                         actions=[2], executed_actions=[2], model_output='forward 300 cm')])
        output = []; dagger.convert_episode_merged(ep, self.root, output)
        self.assertEqual(output[0]['messages'][1]['value'], 'turn left 15 degree')
        self.assertTrue(output[-1]['images'][-1].endswith('001.jpg'))
        ep['model_call_groups'][0]['executed_actions'] = [1]
        with self.assertWarns(UserWarning):
            output = []; dagger.convert_episode_merged(ep, self.root, output)
        self.assertEqual(output, [])

    def test_merged_conversion_rejects_noninteger_call_indices(self):
        self.frames('scene_1', 4)
        for index in (0.5, True, float('inf'), float('nan')):
            with self.subTest(index=index), self.assertWarns(UserWarning):
                output = []
                dagger.convert_episode_merged(self.episode(model_call_groups=[
                    dict(call_step_id=index, actions=[1, 2, 3])]), self.root, output)
                self.assertEqual(output, [])

    def test_merged_conversion_rejects_malformed_group_actions_without_partial_output(self):
        self.frames('scene_1', 4)
        for field in ('actions', 'executed_actions'):
            for value in ('23', {'2': 2, '3': 3}, 23, True):
                with self.subTest(field=field, value=value), self.assertWarnsRegex(UserWarning, 'must be a list'):
                    output = []
                    dagger.convert_episode_merged(self.episode(executed_actions=[1, 2, 3, 0],
                        model_call_groups=[dict(call_step_id=0, executed_actions=[1]),
                                           dict(call_step_id=1, **{field: value})]), self.root, output)
                    self.assertEqual(output, [])

    def test_cli_creates_parent_directories_and_both_distinct_outputs(self):
        self.frames('scene_1', 4)
        annotations = self.root / 'input.json'
        ep = self.episode(model_call_groups=[dict(call_step_id=0, actions=[1,2,3])])
        annotations.write_text('\ufeff\n ' + json.dumps([ep]))
        output = self.root / 'new' / 'output'
        result = subprocess.run([sys.executable, str(ROOT / 'tool/convert_dagger.py'),
                                 '--annotation_path', str(annotations), '--image_root', str(self.root),
                                 '--output_path', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.with_name('output_atomic.json').exists())
        self.assertTrue(output.with_name('output_merged.json').exists())

    def test_standard_samples_match_original_converter(self):
        fixture = json.loads((ROOT / 'tests/fixtures/conversion_reference.json').read_text())
        episodes = fixture['episodes']
        for episode in episodes:
            self.frames(episode['video_id'], fixture['frame_count'])
        for task in ('vln', 'idm', 'ffs'):
            random.seed(fixture['seed']); result = []
            pool = vln.build_scene_frame_pool(episodes, self.root) if task == 'ffs' else None
            for ep in episodes:
                if task == 'ffs': vln.convert_episode_ffs(ep, self.root, pool, result)
                else: getattr(vln, 'convert_episode_' + task)(ep, self.root, result)
            common.relativize_image_paths(result, self.root)
            self.assertEqual(fixture['expected'][task], result, task)

    def test_instruction_objects_are_read_as_text_and_bad_types_are_skipped(self):
        self.frames('scene_1', 4)
        output = []
        vln.convert_episode_vln(self.episode(instruction={'instruction_text': 'walk to the door'}), self.root, output)
        self.assertTrue(output)
        self.assertIn('walk to the door', output[0]['messages'][0]['value'])
        with self.assertWarns(UserWarning):
            self.assertEqual(common.get_instructions(self.episode(instruction=42)), [])

    def test_both_mode_does_not_write_a_partial_conversion(self):
        self.frames('scene_1', 4)
        annotations = self.root / 'input.json'
        annotations.write_text(json.dumps([self.episode()]))  # No model call groups.
        atomic = self.root / 'output_atomic.json'
        atomic.write_text('previous output')
        result = subprocess.run([sys.executable, str(ROOT / 'tool/convert_dagger.py'),
                                 '--annotation_path', str(annotations), '--image_root', str(self.root),
                                 '--output_path', str(self.root / 'output.json'), '--mode', 'both'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('model_call_groups', result.stderr)
        self.assertEqual(atomic.read_text(), 'previous output')
        self.assertFalse((self.root / 'output_merged.json').exists())

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
