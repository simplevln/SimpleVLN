import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recipe.vln_simplevln.batch_utils import object_vector, padding_multiple
from recipe.vln_simplevln.env_server.worker import materialize_dataset_json
from recipe.vln_simplevln.make_manifest import episode_rows


def test_padding_divides_actor_and_logprob_micro_batches():
    config = NS(trainer=NS(nnodes=1, n_gpus_per_node=4), actor_rollout_ref=NS(
        actor=NS(ppo_micro_batch_size_per_gpu=1, ppo_mini_batch_size=4),
        rollout=NS(log_prob_micro_batch_size_per_gpu=3, n=2)))
    multiple = padding_multiple(config)
    assert multiple == 24
    assert multiple % 8 == 0  # Whole actor mini-batches.
    assert (multiple // 4) % 3 == 0  # Complete log-prob micro-batches on each GPU.


def test_image_lists_from_different_generation_batches_concatenate():
    first = object_vector([[0, 0], [0, 0]])
    later = object_vector([[0, 1, 2]])
    joined = np.concatenate([first, later])
    assert joined.shape == (3,)
    assert joined.tolist() == [[0, 0], [0, 0], [0, 1, 2]]


def test_parquet_round_trip_restores_one_dataset(tmp_path):
    episode = {'episode_id': 7, 'scene_id': 'mp3d/scene/scene.glb',
               'instruction': {'instruction_text': 'go'},
               'start_position': [0, 0, 0], 'reference_path': [[0, 0, 0], [1, 0, 0]]}
    rows = episode_rows([
        episode,
        {'episode_id': 8, 'scene_id': 'hm3d/train/a/a.basis.glb', 'instruction': 'other'},
    ])
    out = tmp_path / 'vln_train.parquet'
    pq.write_table(pa.Table.from_pylist(rows), out)
    path = materialize_dataset_json(str(out), str(tmp_path))
    with gzip.open(path, 'rt') as stream:
        payload = json.load(stream)
    assert path.endswith('vln_train.json.gz')
    assert len(payload['episodes']) == 2
    assert payload['episodes'][0]['start_position'] == [0, 0, 0]
    assert payload['episodes'][0]['reference_path'] == [[0, 0, 0], [1, 0, 0]]
    assert 'dataset_name' not in rows[0]['extra_info']


def test_duplicate_episode_key_is_rejected():
    episode = {'episode_id': 1, 'scene_id': 'mp3d/a/a.glb'}
    with pytest.raises(ValueError, match='Duplicate'):
        episode_rows([episode, episode])


def test_manifest_cli_combines_sources_without_losing_routing(tmp_path):
    for name in ['r2r', 'rxr']:
        with gzip.open(tmp_path / f'{name}.json.gz', 'wt') as f:
            json.dump({'episodes': [{'episode_id': 1, 'scene_id': 'mp3d/scene/scene.glb',
                                     'instruction': {'instruction_text': name}}]}, f)
    config = tmp_path / 'datasets.yaml'
    config.write_text('datasets:\n' + ''.join(
        f'  {name}:\n    data_path: ${{oc.env:TEST_ANNOTATIONS}}/{name}.json.gz\n    split: train\n'
        for name in ['r2r', 'rxr']))
    output = tmp_path / 'output' / 'combined.parquet'
    result = subprocess.run([sys.executable, '-B', '-m', 'recipe.vln_simplevln.make_manifest',
                             '--config', str(config), '--output', str(output)],
                            env={**os.environ, 'TEST_ANNOTATIONS': str(tmp_path)},
                            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = pq.read_table(output).to_pylist()
    assert 'dataset_name' not in rows[0]['extra_info']
    assert [r['extra_info']['instruction'] for r in rows] == ['r2r', 'rxr']
    assert {r['extra_info']['episode_id'] for r in rows} == {'1'}
