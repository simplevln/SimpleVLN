"""Exercise the production DAPO loop with CPU tensors and a stateful dataloader."""
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import uuid

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchdata')
pytest.importorskip('tensordict')
pytest.importorskip('ray')
pytest.importorskip('transformers')
pytest.importorskip('datasets')
pytest.importorskip('cachetools')

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as StockTrainer
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from recipe.vln_simplevln.batch_utils import padding_multiple
from recipe.vln_simplevln.tests.test_rollout_batches import decision, load_definition, manager

# Reuse the tensor-manager fixture; training services below are simulated.
_ = manager


@contextmanager
def timer(name, times, **kwargs):
    yield
    times[name] = times.get(name, 0) + 1.


@pytest.fixture
def make_trainer(tmp_path, manager):
    def build(outcomes, *, epochs=1, target=2, max_steps=99, save_freq=5, resume=None):
        retention = BaseCheckpointManager.__new__(BaseCheckpointManager)
        retention.previous_saved_paths = []

        def base_save(self):
            self.saved.append(self.global_steps)
            StockTrainer._save_checkpoint(self)

        def base_load(self):
            self.loads.append(str(self.config.trainer.resume_from_path))
            StockTrainer._load_checkpoint(self)

        runtime = NS(
            AdvantageEstimator=NS(GRPO='grpo', REMAX='remax'), tqdm=Mock(return_value=Mock()),
            marked_timer=timer, extract_reward=lambda b: (b.batch['rm_scores'], {}),
            agg_loss=lambda **kw: kw['loss_mat'].sum(), reduce_metrics=lambda x: {},
            compute_advantage=lambda b, **kw: b, should_save_ckpt_esi=lambda **kw: False,
            compute_data_metrics=lambda **kw: {}, compute_timing_metrics=lambda **kw: {},
            compute_throughout_metrics=lambda **kw: {}, pprint=lambda *args: None,
            find_latest_ckpt_path=find_latest_ckpt_path)
        cls = load_definition('vln_dapo_trainer.py', 'VLNDAPORayTrainer', dict(
            RayPPOTrainer=object, DataProto=DataProto, torch=torch, np=np, uuid=uuid,
            json=json, hashlib=hashlib, io=io, Path=Path, _rt=runtime, padding_multiple=padding_multiple,
            _base_save_checkpoint=base_save, _base_load_checkpoint=base_load))
        trainer = cls()
        trainer.config = OmegaConf.create(dict(
            data=dict(train_batch_size=target, gen_batch_size=1),
            algorithm=dict(adv_estimator='grpo', use_kl_in_reward=False, gamma=1., lam=1.,
                           filter_groups=dict(enable=True, max_num_gen_batches=0)),
            trainer=dict(project_name='test', experiment_name='cpu', logger=[], nnodes=1,
                         n_gpus_per_node=2, total_epochs=epochs, val_before_train=False,
                         default_local_dir=str(tmp_path), resume_mode='resume_path' if resume else 'disable',
                         default_hdfs_dir=None, del_local_ckpt_after_load=False, max_actor_ckpt_to_keep=1,
                         resume_from_path=str(resume) if resume else None,
                         test_freq=-1, save_freq=save_freq, balance_batch=False,
                         critic_warmup=0, esi_redundant_time=0),
            actor_rollout_ref=dict(rollout=dict(n=2, temperature=1., log_prob_micro_batch_size_per_gpu=1),
                                   actor=dict(loss_agg_mode='token-mean', loss_scale_factor=None,
                                              checkpoint=dict(async_save=False),
                                              ppo_micro_batch_size_per_gpu=1, ppo_mini_batch_size=2))))
        trainer.total_training_steps = max_steps
        trainer.use_critic = trainer.use_rm = trainer.use_reference_policy = False
        trainer._dump_executor = NS(_shutdown=False)
        trainer._shutdown_dump_executor = Mock()
        trainer.checkpoint_manager = Mock()
        trainer.actor_rollout_wg = Mock()

        def save_actor(local_path, remote_path, step, max_ckpt_to_keep):
            actor = Path(local_path)
            actor.mkdir(parents=True, exist_ok=True)
            (actor / 'weights').write_text('saved model')
            retention.register_checkpoint(local_path, max_ckpt_to_keep)

        def load_actor(local_path, **kwargs):
            assert (Path(local_path) / 'weights').is_file()

        trainer.actor_rollout_wg.save_checkpoint.side_effect = save_actor
        trainer.actor_rollout_wg.load_checkpoint.side_effect = load_actor
        trainer.resource_pool_manager = NS(get_n_gpus=lambda: 2)
        trainer.train_dataset = [{'sample_id': torch.tensor(i)} for i in range(len(outcomes))]
        trainer.train_dataloader = StatefulDataLoader(trainer.train_dataset, batch_size=1, num_workers=0)
        trainer._get_gen_batch = lambda b: b
        trainer.tokenizer = NS(pad_token_id=0)
        trainer.saved, trainer.updates, trainer.seen, trainer.timings, trainer.loads = [], [], [], [], []
        runtime.compute_timing_metrics = lambda **kw: trainer.timings.append(dict(kw['timing_raw'])) or {}

        def generate(batch):
            sample = int(batch.batch['sample_id'][0])
            trainer.seen.append(sample)
            uid = batch.non_tensor_batch['uid'][0]
            rewards = [0., float(outcomes[sample])]
            rows = [decision(uid=uid, trajectory=f'{uid}-{i}', reward=r, success=r)
                    for i, r in enumerate(rewards)]
            return manager._build_output(rows, [], [], batch.meta_info)

        trainer.async_rollout_manager = NS(generate_sequences=generate)

        def logprob(batch):
            zeros = torch.zeros_like(batch.batch['rm_scores'])
            return DataProto.from_dict(tensors={'entropys': zeros, 'old_log_probs': zeros}), 0.

        trainer._compute_old_log_prob = logprob

        def update(batch):
            uids = batch.non_tensor_batch['uid'][~batch.non_tensor_batch['is_padding']]
            trainer.updates.append((trainer.global_steps, len(set(uids))))
            return DataProto(meta_info={'metrics': []})

        trainer._update_actor = update
        return trainer
    return build


def test_epoch_limit_trains_partial_final_batch_and_saves(make_trainer):
    trainer = make_trainer([True, True, True], target=2, save_freq=5)
    trainer.fit()
    assert trainer.updates == [(1, 2), (2, 1)]
    assert trainer.saved == [2]
    assert trainer.seen == [0, 1, 2]
    trainer._shutdown_dump_executor.assert_called_once()


def test_filtered_tail_still_saves_last_update_and_finished_epoch(make_trainer):
    trainer = make_trainer([True, True, False], target=2)
    trainer.fit()
    assert trainer.updates == [(1, 2)]
    assert trainer.saved == [1]
    progress = json.loads((Path(trainer.config.trainer.default_local_dir) /
                           'global_step_1/vln_progress.json').read_text())
    assert progress['epoch'] == 1 and progress['batches_in_epoch'] == 0


def test_all_filtered_run_reports_that_no_update_was_possible(make_trainer):
    trainer = make_trainer([False, False])
    with pytest.raises(RuntimeError, match='No training update'):
        trainer.fit()
    assert trainer.updates == trainer.saved == []
    trainer._shutdown_dump_executor.assert_called_once()


@pytest.mark.parametrize('at_epoch_end', [False, True])
def test_resume_tracks_data_epoch_separately_from_optimizer_steps(make_trainer, at_epoch_end):
    trainer = make_trainer([True] * 3, epochs=5)
    iterator = iter(trainer.train_dataloader)
    consumed = 3 if at_epoch_end else 1
    for _ in range(consumed):
        next(iterator)
    trainer.global_steps = 3  # Divides loader length, but is unrelated to the data cursor.
    trainer._vln_epoch, trainer._vln_batches_in_epoch = 2, consumed
    trainer._save_checkpoint()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_3'
    resumed = make_trainer([True] * 3, epochs=4, target=1, resume=folder)
    resumed.fit()
    assert resumed.seen == ([] if at_epoch_end else [1, 2]) + [0, 1, 2]
    assert [step for step, _ in resumed.updates] == list(range(4, 4 + len(resumed.seen)))


def test_completed_step_budget_does_not_run_one_more_update(make_trainer):
    trainer = make_trainer([True] * 3)
    next(iter(trainer.train_dataloader))
    trainer.global_steps = 3
    trainer._vln_epoch, trainer._vln_batches_in_epoch = 0, 1
    trainer._save_checkpoint()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_3'
    resumed = make_trainer([True] * 3, resume=folder, max_steps=3)
    resumed.fit()
    assert resumed.seen == resumed.updates == []
    resumed.checkpoint_manager.update_weights.assert_not_called()


def test_resume_after_filtered_tail_starts_the_next_data_epoch(make_trainer):
    trainer = make_trainer([True, True, False])
    trainer.fit()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_1'
    resumed = make_trainer([True, True, False], epochs=2, resume=folder)
    resumed.fit()
    assert resumed.seen == [0, 1, 2]
    assert resumed.updates == [(2, 2)]


def test_legacy_checkpoint_does_not_silently_guess_data_epoch(make_trainer):
    trainer = make_trainer([True])
    trainer.fit()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_1'
    (folder / 'vln_progress.json').unlink()
    resumed = make_trainer([True], epochs=2, resume=folder)
    with pytest.raises(ValueError, match='lacks vln_progress.json'):
        resumed.fit()
    assert resumed.seen == []
    assert resumed.loads == []
    resumed._shutdown_dump_executor.assert_called_once()


def test_filtered_tail_does_not_delete_checkpoint_under_retention(make_trainer):
    trainer = make_trainer([True, True, False], save_freq=1)
    trainer.fit()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_1'
    assert trainer.saved == [1]  # Saving the actor twice would delete it with retention=1.
    assert (folder / 'actor/weights').read_text() == 'saved model'
    progress = json.loads((folder / 'vln_progress.json').read_text())
    assert progress['epoch'] == 1 and progress['batches_in_epoch'] == 0
    assert progress['data_sha256'] == hashlib.sha256((folder / 'data.pt').read_bytes()).hexdigest()


def test_sampling_time_includes_batches_without_an_update(make_trainer):
    trainer = make_trainer([True, True, False, True])
    trainer.fit()
    assert [m['gen'] for m in trainer.timings] == [2, 2]
    assert [m['step'] for m in trainer.timings] == [2, 2]


def test_mismatched_data_cursor_is_rejected(make_trainer):
    trainer = make_trainer([True] * 3)
    next(iter(trainer.train_dataloader))
    trainer.global_steps = 1
    trainer._vln_epoch, trainer._vln_batches_in_epoch = 0, 1
    trainer._save_checkpoint()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_1'
    different = make_trainer([True] * 3)
    torch.save(different.train_dataloader.state_dict(), folder / 'data.pt')
    resumed = make_trainer([True] * 3, resume=folder)
    with pytest.raises(ValueError, match='do not match'):
        resumed.fit()
    assert resumed.seen == []
    assert resumed.loads == []


def test_validation_only_does_not_train_with_default_val_before_train_false(make_trainer):
    trainer = make_trainer([True])
    trainer.fit()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_1'
    (folder / 'vln_progress.json').unlink()  # Legacy weights are still usable for validation.
    resumed = make_trainer([True], resume=folder, max_steps=1)
    resumed.config.trainer.val_only = True
    resumed._validate = Mock(return_value={'validation/reward': .5})
    resumed.fit()
    resumed._validate.assert_called_once()
    assert resumed.seen == resumed.updates == []


@pytest.mark.parametrize('mode', ['auto', 'resume_path'])
def test_resume_uses_real_checkpoint_tracker_and_normalizes_paths(make_trainer, mode):
    trainer = make_trainer([True])
    trainer.fit()
    folder = Path(trainer.config.trainer.default_local_dir) / 'global_step_1'
    resumed = make_trainer([True], epochs=2)
    resumed.config.trainer.resume_mode = mode
    resumed.config.trainer.resume_from_path = str(folder) + '/' if mode == 'resume_path' else None
    original_path = resumed.config.trainer.resume_from_path
    resumed.fit()
    assert resumed.loads == [str(folder)]
    assert resumed.config.trainer.resume_mode == mode
    assert resumed.config.trainer.resume_from_path == original_path
    assert resumed.updates == [(2, 1)]
