"""CPU tensor regressions; no model, Ray cluster, or Habitat process is started.

Load the production class bodies without the distributed training imports. Torch,
TensorDict and DataProto are real; only agent workers and the Ray object store are
simulated. These tests require the training environment's tensor dependencies.
"""
import ast
import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('tensordict')
pytest.importorskip('ray')
pytest.importorskip('transformers')

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from recipe.vln_simplevln.batch_utils import object_vector, padding_multiple

RECIPE = Path(__file__).resolve().parents[1]


def load_definition(filename, name, namespace):
    tree = ast.parse((RECIPE / filename).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RECIPE / filename), 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def manager():
    import os
    import time
    namespace = dict(asyncio=asyncio, os=os, time=time, np=np, torch=torch,
                     DataProto=DataProto, AgentLoopManager=object,
                     auto_await=lambda f: f, object_vector=object_vector,
                     padding_multiple=padding_multiple, ray=NS(put=lambda x: x))
    load_definition('online_rollout_manager.py', '_fmt_dur', namespace)
    cls = load_definition('online_rollout_manager.py', 'VLNOnlineRolloutManager', namespace)
    instance = cls()
    instance.rollout_config = NS(prompt_length=6, response_length=3)
    instance.config = NS(trainer=NS(nnodes=1, n_gpus_per_node=2), actor_rollout_ref=NS(
        actor=NS(ppo_micro_batch_size_per_gpu=1, ppo_mini_batch_size=2),
        rollout=NS(log_prob_micro_batch_size_per_gpu=1, n=2)))
    return instance


@pytest.fixture
def trainer():
    cls = load_definition('vln_dapo_trainer.py', 'VLNDAPORayTrainer', dict(
        RayPPOTrainer=object, DataProto=DataProto, torch=torch, np=np,
        padding_multiple=padding_multiple))
    instance = cls()
    instance.tokenizer = NS(pad_token_id=99)
    return instance


def decision(uid='episode', trajectory='a', reward=1., success=1., indices=None):
    return dict(prompt_ids=[11, 12], response_ids=[21, 22], response_mask=[1, 1],
                position_ids=[[axis * 10 + i for i in range(4)] for axis in range(4)],
                uid=uid, trajectory_uid=trajectory, trajectory_reward=reward,
                trajectory_success=success, decision_loss_weight=1., turn_id=0,
                action_text='stop', image_indices=indices or [0, 0], raw_prompt='go')


def test_flatten_and_repad_preserve_multimodal_alignment(manager, trainer):
    rows = [decision(), decision(trajectory='b', indices=[0, 1, 2]), decision(trajectory='c')]
    meta = {'pad_token_id': 99, 'temperature': 1.0}
    output = manager._build_output(rows, [1, 1, 1], [1, 1, 1], meta)
    assert len(output) == 4
    assert not meta.get('vln_flattened')  # Output metadata must not mutate the input.
    assert output.non_tensor_batch['image_indices'].shape == (4,)
    assert output.batch['attention_mask'][0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert output.batch['rm_scores'][0].tolist() == [0, 1, 0]
    assert output.batch['position_ids'][0, :, 4:8].tolist() == decision()['position_ids']
    assert output.batch['response_mask'][-1].sum() == 0
    real = output.select_idxs(~output.non_tensor_batch['is_padding'])
    repadded = trainer._vln_repad(real, 4)
    assert len(repadded) == 4
    for key in output.batch.keys():
        assert torch.equal(repadded.batch[key], output.batch[key]), key
    assert repadded.non_tensor_batch['image_indices'][-1] == []
    assert repadded.non_tensor_batch['image_buffer_ref'][-1] is None


def test_group_filter_and_success_metric_use_episode_outcomes(manager, trainer):
    rows = [decision(trajectory='a', reward=1., success=1.),
            decision(trajectory='b', reward=.2, success=0.),
            decision(uid='zero', trajectory='c', reward=0., success=0.)]
    output = manager._build_output(rows, [], [], {})
    real = output.select_idxs(~output.non_tensor_batch['is_padding'])
    mask, groups, filtered = trainer._valid_uids(real, 1e-6)
    assert mask.tolist() == [True, True, False]
    assert groups == ['episode'] and filtered == 1
    metrics = trainer._vln_metrics_from_batch(output)
    assert metrics['vln/traj_sr'] == pytest.approx(1 / 3)
    assert metrics['vln/traj_count'] == 3


def test_trajectory_advantage_is_not_weighted_by_number_of_rows():
    fn = load_definition('patches.py', '_traj_grpo_advantage', dict(torch=torch, defaultdict=defaultdict))
    rewards = torch.tensor([[0., 1.], [0., 0.], [0., 0.], [0., 0.]])
    mask = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [0., 0.]])
    args = dict(token_level_rewards=rewards, response_mask=mask,
                index=['g', 'g', 'g', 'pad'], traj_index=['a', 'b', 'b', 'pad'])
    adv, _ = fn(**args)
    assert adv[:, 0].tolist() == pytest.approx([2 ** -.5, -2 ** -.5, -2 ** -.5, 0], abs=2e-6)
    weighted, _ = fn(**args, decision_weight=[1., .5, .5, 0.])
    assert weighted[0].sum() == pytest.approx(-weighted[1:3].sum().item())
    assert torch.isfinite(weighted).all()


def episode_inputs(count, validate=True):
    return DataProto.from_dict(
        tensors={'episode_index': torch.arange(count).unsqueeze(-1)},
        non_tensors={'uid': np.array([str(i) for i in range(count)], dtype=object)},
        meta_info={'validate': validate, 'temperature': 1.0})


class EpisodeWorker:
    def __init__(self, seen, fail_at=None):
        self.generate_sequences = NS(remote=self.generate)
        self.seen = seen
        self.fail_at = fail_at

    async def generate(self, batch):
        indices = batch.batch['episode_index'].flatten().tolist()
        assert indices  # Never dispatch an empty chunk to an agent worker.
        self.seen.append(indices)
        await asyncio.sleep(0)
        if self.fail_at in indices:
            raise RuntimeError('episode failed')
        n = len(indices)
        trajectories = []
        for i in indices:
            d = decision(uid=str(i))
            trajectories.append(dict(group_uid=str(i), trajectory_uid=f'{i}-trial',
                                     reward=float(i % 2), num_decisions=i + 1,
                                     metrics={'success': i % 2}, decisions=[d] * (i + 1)))
        return DataProto.from_dict(
            tensors={'responses': torch.tensor(indices).unsqueeze(-1),
                     'rm_scores': torch.tensor([i % 2 for i in indices]).unsqueeze(-1)},
            non_tensors={**batch.non_tensor_batch, 'trajectory': object_vector(trajectories)},
            meta_info={'reward_extra_keys': [], 'metrics': [dict(
                generate_sequences=1., tool_calls=2., compute_score=0., num_preempted=0)] * n})


@pytest.mark.parametrize('scheduler', ['window', 'sliding'])
def test_validation_preserves_episode_order_counts_and_rewards(manager, monkeypatch, scheduler):
    monkeypatch.setenv('VLN_ROLLOUT_WINDOW', '2')
    monkeypatch.setenv('VLN_ROLLOUT_SCHEDULER', scheduler)
    seen = []
    manager.agent_loop_workers = [EpisodeWorker(seen) for _ in range(4)]
    original = episode_inputs(5)
    padded, pad = pad_dataproto_to_divisor(original, 4)
    output = asyncio.run(manager.generate_sequences(padded))
    assert len(output) == len(padded)
    result = original.union(unpad_dataproto(output, pad))
    assert result.batch['responses'].flatten().tolist() == list(range(5))
    assert result.batch['rm_scores'].flatten().tolist() == [0, 1, 0, 1, 0]
    assert result.non_tensor_batch['uid'].tolist() == [str(i) for i in range(5)]
    assert not result.meta_info.get('vln_flattened')
    assert [i for chunk in seen for i in chunk] == [0, 1, 2, 3, 4, 0, 1, 2]


def test_validation_does_not_drop_failed_episodes(manager, monkeypatch):
    monkeypatch.setenv('VLN_ROLLOUT_WINDOW', '2')
    manager.agent_loop_workers = [EpisodeWorker([], fail_at=2) for _ in range(4)]
    with pytest.raises(RuntimeError, match='episode failed'):
        asyncio.run(manager.generate_sequences(episode_inputs(3)))


@pytest.mark.parametrize('scheduler', ['window', 'sliding'])
@pytest.mark.parametrize('window_size', [2, 6])
def test_partial_training_windows(manager, monkeypatch, scheduler, window_size):
    monkeypatch.setenv('VLN_ROLLOUT_WINDOW', str(window_size))
    monkeypatch.setenv('VLN_ROLLOUT_SCHEDULER', scheduler)
    seen = []
    manager.agent_loop_workers = [EpisodeWorker(seen) for _ in range(4)]
    output = asyncio.run(manager.generate_sequences(episode_inputs(7, validate=False)))
    real = ~output.non_tensor_batch['is_padding']
    assert real.sum() == 28  # Sum of 1 through 7 decisions.
    assert output.non_tensor_batch['uid'][real].tolist() == [str(i) for i in range(7) for _ in range(i + 1)]
    assert output.meta_info['temperature'] == 1.0
    assert sorted(i for chunk in seen for i in chunk) == list(range(7))
