import ast
import base64
from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tool'))
sys.path.insert(0, str(ROOT / 'vln'))
import convert_dagger
from collection_io import (
    append_records, completed_keys, deduplicate_records, load_records, write_snapshot,
)


class Follower:
    def __init__(self, sim, **kwargs): self.sim = sim
    def get_next_action(self, target):
        return self.sim.direction if self.sim.steps < self.sim.goal_step else 0


class Environment:
    def __init__(self, goal_step, direction):
        self.goal_step, self.direction = goal_step, direction
        self.current_episode = SimpleNamespace(episode_id='1', trajectory_id=1,
            scene_id='/scenes/scene/scene.glb', reference_path=[[0,0,0], [1,0,0]],
            instruction=SimpleNamespace(instruction_text='go'))
        self.sim = self
    def reset(self):
        self.steps = 0; self.episode_over = False
        return self.observation()
    def observation(self): return {'rgb': np.full((2,2,3), self.steps % 255, dtype=np.uint8)}
    def step(self, action):
        self.steps += 1
        self.episode_over = action == 0
        return self.observation()
    def get_metrics(self):
        return dict(distance_to_goal=0 if self.steps >= self.goal_step else 1,
                    pl=np.float64(.5), success=float(self.episode_over))


def collector_class():
    tree = ast.parse((ROOT / 'vln/dagger.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DAggerCollector')
    ns = dict(np=np, Image=Image, os=os, io=io, base64=base64, re=re, random=random,
              habitat=SimpleNamespace(Env=Environment), Dict=dict, json=json,
              closing=closing, time=time, tqdm=tqdm, completed_keys=completed_keys,
              append_records=append_records, deduplicate_records=deduplicate_records,
              load_records=load_records, write_snapshot=write_snapshot,
              ShortestPathFollower=Follower, MIDGOAL_RADIUS=.5, GOAL_RADIUS=.25,
              RELATIVE_PATH_LENGTH_THRESHOLD=.93, SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD=.85)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<collector>', 'exec'), ns)
    return ns['DAggerCollector']


class CollectionTests(unittest.TestCase):
    def test_successful_collection_preserves_shards_and_resumes_without_environment(self):
        for journal_exists in (False, True):
            with self.subTest(journal_exists=journal_exists), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                previous = dict(id=0, video='previous')
                shard = root / 'annotations_actions_0.json'
                write_snapshot(shard, [previous])
                if journal_exists:
                    append_records(root / 'annotations_actions_0.jsonl', [dict(id=2, video='journal')])
                cls = collector_class(); obj = cls.__new__(cls)
                env = Environment(goal_step=2, direction=1)
                env.current_episode.goals = [SimpleNamespace(radius=0)]
                obj.update_dataset.__globals__['habitat'].make_dataset = lambda **kwargs: SimpleNamespace(
                    episodes=[env.current_episode])
                closed = []
                env.close = lambda: closed.append(True)
                obj.rank, obj.world_size, obj.data_path = 0, 1, None
                obj.config = SimpleNamespace(habitat=SimpleNamespace(dataset=SimpleNamespace(type='R2RVLN-v1')))
                obj.args = SimpleNamespace(annotations_actions='annotations_actions', collect_images=True,
                    dagger_bo_n=1, dagger_save_video=False, force_expert=False, dagger_data_it=0,
                    dagger_save_model_trace=False, max_action_history=200)
                obj.dagger_config = SimpleNamespace(update_size=1, p=0)
                obj.output_path, obj.dataset = folder, 'r2r'
                obj.gt_annotations = {'1': {'actions': [1, 1, 1]}}
                obj._call_vllm_eval_style = lambda *args, **kwargs: 'forward 50 cm'
                obj.config_env = lambda: env
                with redirect_stdout(io.StringIO()):
                    obj.update_dataset()
                records = load_records(shard)
                self.assertIn(previous, records)
                self.assertEqual({row['id'] for row in records}, {0, 1, 2} if journal_exists else {0, 1})
                self.assertEqual(completed_keys(root, 'annotations_actions'), {('scene', '1', '1')})
                self.assertEqual(closed, [True])
                def unexpected_environment():
                    self.fail('A completed collection must not open another environment')
                obj.config_env = unexpected_environment
                with redirect_stdout(io.StringIO()):
                    obj.update_dataset()
                self.assertEqual(load_records(shard), records)

    def test_failed_collection_closes_environment_without_marking_completion(self):
        cls = collector_class(); obj = cls.__new__(cls)
        episode = SimpleNamespace(scene_id='scene/scene.glb', episode_id='1', trajectory_id=1,
                                  goals=[SimpleNamespace(radius=0)])
        obj.update_dataset.__globals__['habitat'].make_dataset = lambda **kwargs: SimpleNamespace(episodes=[episode])
        obj.rank, obj.world_size, obj.data_path = 0, 1, None
        obj.config = SimpleNamespace(habitat=SimpleNamespace(dataset=SimpleNamespace(type='R2RVLN-v1')))
        obj.args = SimpleNamespace(annotations_actions='annotations_actions', collect_images=True,
                                   dagger_bo_n=1, dagger_save_video=False, force_expert=False)
        obj.dagger_config = SimpleNamespace(update_size=1)
        closed = []
        obj.config_env = lambda: SimpleNamespace(close=lambda: closed.append(True))
        def fail(**kwargs):
            raise RuntimeError('model request failed')
        obj.generate = fail
        with tempfile.TemporaryDirectory() as folder:
            obj.output_path = folder
            with self.assertRaisesRegex(RuntimeError, 'model request failed'):
                obj.update_dataset()
            self.assertEqual(closed, [True])
            self.assertFalse((Path(folder) / 'result.json').exists())

    def test_model_requests_match_eval_settings(self):
        cls = collector_class(); obj = cls.__new__(cls)
        recorded = []
        def create(**kwargs):
            recorded.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='forward 25 cm'))])
        obj.vllm_clients = [SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))]
        obj.vllm_model_name = 'mock'
        obj._call_vllm_eval_style([Image.new('RGB', (2,2)) for _ in range(30)], 'go')
        request = recorded[0]
        self.assertEqual(request['max_completion_tokens'], 512)
        self.assertEqual(request['temperature'], .3)
        self.assertEqual(request['top_p'], .95)
        self.assertNotIn('extra_body', request)
        self.assertEqual(request['messages'][0]['content'][0]['text'], 'You are a visual language navigation model.')
        content = request['messages'][1]['content']
        self.assertEqual(sum(item['type'] == 'image_url' for item in content), 9)
        random.seed(7)
        expanded = cls._expand_actions([[1,300],[2,90],[3,15]])
        self.assertEqual(expanded, [1,1,1,2,2,2])
        self.assertTrue(cls._expand_actions([[None,None]])[0] in (1,2,3))

    def collect(self, path, goal_step, direction):
        cls = collector_class(); obj = cls.__new__(cls)
        obj.args = SimpleNamespace(dagger_data_it=0, collect_images=True,
                                   dagger_save_model_trace=False, max_action_history=200)
        obj.dagger_config = SimpleNamespace(p=0)
        obj.output_path = path; obj.dataset = 'r2r'; obj.rank = 0
        obj.gt_annotations = {'1': {'actions': [1] * max(goal_step,3)}}
        histories = []
        def infer(rgb_list, *args, **kwargs):
            histories.append(len(rgb_list)); return 'forward 300 cm'
        obj._call_vllm_eval_style = infer
        env = Environment(goal_step, direction)
        with redirect_stdout(io.StringIO()):
            result = obj.generate(env, save_video=False)
        return result, histories

    def test_collector_records_actual_terminal_action_and_converts(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.collect(folder, goal_step=2, direction=2)
            annotation = result['anno_actions'][0]
            self.assertTrue(result['metrics']['save'])
            self.assertIsInstance(result['metrics']['save'], int)
            json.dumps(result['metrics'])
            self.assertEqual(annotation['executed_actions'], [1,1,0])
            self.assertEqual(annotation['terminal_action'], 0)
            self.assertEqual(annotation['model_call_groups'][0]['planned_actions'], [1,1,1])
            self.assertEqual(annotation['model_call_groups'][0]['executed_actions'], [1,1,0])
            output = []; convert_dagger.convert_episode_merged(annotation, folder, output)
            self.assertEqual(output[0]['messages'][1]['value'], 'forward 50 cm, stop')
            self.assertTrue(all(Path(image).exists() for sample in output for image in sample['images']))

    def test_collection_history_window_is_200(self):
        with tempfile.TemporaryDirectory() as folder:
            result, histories = self.collect(folder, goal_step=205, direction=1)
            self.assertEqual(max(histories), 200)
            self.assertTrue(result['metrics']['save'])
            self.assertIsInstance(result['metrics']['save'], int)
            json.dumps(result['metrics'])
            annotation = result['anno_actions'][0]
            self.assertEqual(len(annotation['executed_actions']), 206)
            self.assertEqual(len(list((Path(folder) / annotation['video'] / 'rgb').glob('*.jpg'))), 206)


if __name__ == '__main__': unittest.main()
