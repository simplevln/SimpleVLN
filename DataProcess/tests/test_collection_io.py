from contextlib import redirect_stdout
import io
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vln'))
from collection_io import append_records, completed_keys, load_records, merge_annotations, write_snapshot


def writer(path, worker):
    for step in range(10):
        append_records(path, [dict(worker=worker, step=step, text='向前走' * 2000)])


class CollectionIOTests(unittest.TestCase):
    def test_completion_requires_saved_annotations_and_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            append_records(root / 'result.json', [dict(scene='scene', episode_id=1, trajectory_id=2, save=1)])
            self.assertEqual(completed_keys(root, 'annotations_actions'), set())
            annotation = dict(id=1, scene_id='scene', episode_id=1, trajectory_id=2,
                              video='images/scene_1', actions=[-1,1])
            append_records(root / 'annotations_actions_0.jsonl', [annotation])
            self.assertEqual(completed_keys(root, 'annotations_actions'), set())
            images = root / annotation['video'] / 'rgb'; images.mkdir(parents=True)
            for index in range(2): (images / f'{index:03d}.jpg').touch()
            self.assertEqual(completed_keys(root, 'annotations_actions'), {('scene','1','2')})
            append_records(root / 'result.json', [dict(scene='scene', episode_id=3, trajectory_id=2, save=0)])
            self.assertEqual(completed_keys(root, 'annotations_actions'), {('scene','1','2'), ('scene','3','2')})

    def test_concurrent_append_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'rows.jsonl'; output.write_bytes(b'{"partial":')
            processes = [mp.get_context('spawn').Process(target=writer, args=(output,i)) for i in range(3)]
            for process in processes: process.start()
            for process in processes:
                process.join(10); self.assertEqual(process.exitcode, 0)
            rows = load_records(output)
            self.assertEqual(len(rows), 30)
            self.assertEqual(len({(r['worker'],r['step']) for r in rows}), 30)

    def test_resume_validates_latest_annotation_even_when_older_frames_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = dict(id=1, scene_id='scene', episode_id=1, trajectory_id=2,
                            video='images/scene_1', actions=[-1, 1])
            latest = dict(previous, actions=[-1, 1, 1, 1, 1])
            write_snapshot(root / 'annotations_actions.json', [previous])
            write_snapshot(root / 'annotations_actions_0.json', [previous])
            append_records(root / 'annotations_actions_0.jsonl', [previous, latest])
            append_records(root / 'result.json', [dict(scene='scene', episode_id=1, trajectory_id=2, save=1)])
            images = root / previous['video'] / 'rgb'; images.mkdir(parents=True)
            for index in range(4): (images / f'{index:03d}.jpg').touch()
            self.assertEqual(completed_keys(root, 'annotations_actions'), set())
            with redirect_stdout(io.StringIO()): merge_annotations(root)
            self.assertEqual(load_records(root / 'annotations_actions.json'), [latest])
            self.assertEqual(completed_keys(root, 'annotations_actions'), set())
            (images / '004.jpg').touch()
            self.assertEqual(completed_keys(root, 'annotations_actions'), {('scene', '1', '2')})

    def test_latest_result_controls_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = dict(scene='scene', episode_id=1, trajectory_id=2)
            append_records(root / 'result.json', [dict(result, save=0), dict(result, save=1)])
            # The latest run claims a save but has no matching annotation or images.
            self.assertEqual(completed_keys(root, 'annotations_actions'), set())
            append_records(root / 'result.json', [dict(result, save=0)])
            self.assertEqual(completed_keys(root, 'annotations_actions'), {('scene', '1', '2')})

    def test_merge_retains_legacy_shards_and_latest_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'annotations_actions_1.json').write_text(json.dumps([dict(id=2,video='b')]))
            append_records(root/'annotations_actions_0.jsonl', [dict(id=1,video='a',value=1),dict(id=1,video='a',value=2)])
            with redirect_stdout(io.StringIO()): merge_annotations(folder)
            records=json.loads((root/'annotations_actions.json').read_text())
            self.assertEqual({r['id'] for r in records},{1,2})
            self.assertEqual(next(r for r in records if r['id']==1)['value'],2)

    def test_merge_preserves_snapshot_when_old_shards_are_absent(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            target = root / 'annotations_actions.json'
            write_snapshot(target, [dict(id=1, video='old')])
            append_records(root / 'annotations_actions_0.jsonl', [dict(id='2', video='new')])
            merge_annotations(root)
            expected = [dict(id=1, video='old'), dict(id='2', video='new')]
            self.assertEqual(load_records(target), expected)
            merge_annotations(root)
            self.assertEqual(load_records(target), expected)

    def test_partial_journal_preserves_records_in_its_legacy_shard(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            legacy = [dict(id=1, video='a'), dict(id=2, video='b', value=1)]
            write_snapshot(root / 'annotations_actions_0.json', legacy)
            append_records(root / 'annotations_actions_0.jsonl', [dict(id=2, video='b', value=2)])
            merge_annotations(root)
            self.assertEqual(load_records(root / 'annotations_actions.json'),
                             [dict(id=1, video='a'), dict(id=2, video='b', value=2)])
            self.assertEqual(load_records(root / 'annotations_actions_0.json'), legacy)

    def test_corrupt_snapshot_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'annotations_actions.json'
            target.write_text('{')
            append_records(root / 'annotations_actions_0.jsonl', [dict(id=2, video='new')])
            with self.assertRaisesRegex(ValueError, 'Invalid annotation JSON'):
                merge_annotations(root)
            self.assertEqual(target.read_text(), '{')

    def test_failed_snapshot_serialization_keeps_previous_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'annotations_actions.json'
            target.write_text('previous snapshot')
            with self.assertRaises(TypeError):
                write_snapshot(target, [object()])
            self.assertEqual(target.read_text(), 'previous snapshot')
            self.assertEqual(list(root.iterdir()), [target])


if __name__ == '__main__': unittest.main()
