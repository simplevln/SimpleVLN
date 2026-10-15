"""Check navigation requests and action execution without Habitat or model weights."""

import argparse
import ast
import base64
from contextlib import nullcontext, redirect_stderr
from copy import deepcopy
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'vln/eval.py').read_text())


class Frame:
    def __init__(self, label):
        self.label = str(label)

    def astype(self, _):
        return self

    def convert(self, _):
        return self

    def save(self, stream, format):
        stream.write(self.label.encode())


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.requests = []
        self.output = 'forward 25 cm'

        def client(**settings):
            def create(**request):
                self.requests.append((settings['base_url'], deepcopy(request)))
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=self.output))])
            return SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        nodes = [n for n in TREE.body if isinstance(n, (ast.Assign, ast.ClassDef))
                 or isinstance(n, ast.FunctionDef) and n.name == 'encode_image_base64']
        nodes = [n for n in nodes if not isinstance(n, ast.Assign)
                 or any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets)]
        namespace = dict(Agent=object, OpenAI=client, os=os, re=re, BytesIO=BytesIO,
                         base64=base64, Image=SimpleNamespace(fromarray=lambda x: x),
                         random=SimpleNamespace(randint=lambda a, b: 3))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<navigation>', 'exec'), namespace)
        self.agent = namespace['SimpleVLN_Agent'](
            'http://server-a/v1,http://server-b/v1', 'model', self.folder.name,
            25, 15, 10, 1.0)

    def act(self, frame):
        return self.agent.act({'rgb': Frame(frame), 'instruction': {'text': 'Go to the door.'}},
                              {}, 'episode')['action']

    def image_labels(self, request):
        content = request['messages'][1]['content']
        return [base64.b64decode(item['image_url']['url'].split(',', 1)[1]).decode()
                for item in content if item['type'] == 'image_url']

    def test_prompt_images_sampling_and_request_parameters(self):
        for i in range(12):
            self.assertEqual(self.act(i), 1)
        first = self.requests[0][1]
        self.assertEqual(first['messages'][0], {'role': 'system', 'content': [
            {'type': 'text', 'text': 'You are a visual language navigation model.'}]})
        self.assertEqual(self.image_labels(first), ['0', '0'])
        texts = [item['text'] for item in first['messages'][1]['content']
                 if item['type'] == 'text']
        self.assertEqual(texts, [
            'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations',
            'and an image of the current observation',
            " Your assigned task is: 'Go to the door.'. Analyze this series of images to decide your next move, "
            'which could involve turning left or right by a specific degree or moving forward a certain distance.',
        ])
        self.assertEqual((first['model'], first['temperature'], first['top_p'],
                          first['max_completion_tokens']), ('model', .3, .95, 512))
        self.assertEqual([endpoint for endpoint, _ in self.requests[:4]],
                         ['http://server-a/v1', 'http://server-b/v1'] * 2)
        self.assertEqual(self.image_labels(self.requests[-1][1]),
                         ['2', '3', '4', '5', '7', '8', '9', '10', '11'])

    def test_actions_are_limited_and_pending_actions_avoid_extra_requests(self):
        self.output = '<think>plan\nnext move</think>forward 150 cm, left 30 degrees, right 90 degrees'
        self.assertEqual([self.act(i) for i in range(5)], [1, 1, 1, 2, 2])
        self.assertEqual(len(self.requests), 1)
        self.agent.reset()
        self.output = 'stop'
        self.assertEqual(self.act(5), 0)
        self.assertEqual(self.image_labels(self.requests[-1][1]), ['5', '5'])

    def test_action_parsing_and_invalid_output_fallback(self):
        for text, expected in [('forward', (1, 25)), ('turn left', (2, 15)),
                               ('turn right', (3, 15)), ('<answer>STOP</answer>', (0, None)),
                               ('unrecognized', (None, None))]:
            with self.subTest(text=text):
                self.assertEqual(self.agent.extract_result(text), expected)
        self.output = 'unrecognized'
        self.assertEqual(self.act(0), 3)


class CommandLineTests(unittest.TestCase):
    def test_video_alias_and_legacy_generation_flag(self):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = []
        namespace = dict(argparse=argparse, seed_all=lambda: None,
                         resolve_vllm_urls=lambda _: 'http://server/v1',
                         result_directory_lock=lambda _: nullcontext(),
                         run_evaluation=lambda args, _: calls.append(args), print=lambda *a: None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<cli>', 'exec'), namespace)
        argv = ['eval.py', '--exp-config', 'config.yaml', '--split-num', '1', '--result-path', 'results']
        for video_flag in ('--save-video', '--save_vedio'):
            with patch('sys.argv', argv + [video_flag, '--num-generations', '1']):
                namespace['main']()
            self.assertTrue(calls[-1].save_video)
        with patch('sys.argv', argv + ['--num-generations', '2']), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                namespace['main']()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
