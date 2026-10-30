import os
from pathlib import Path
import shlex
import subprocess
import sys

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra import compose, initialize_config_dir
import pytest

RECIPE = Path(__file__).resolve().parents[1]


def launch(preset='qwen3vl_4b', **overrides):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('VLN_', 'MODEL_', 'TRAIN_', 'PPO_', 'ROLLOUT_'))}
    env.update(MODEL_PATH='/path/to/model', DRY_RUN='1')
    env.update({key: str(value) for key, value in overrides.items()})
    return subprocess.run(['bash', str(RECIPE / f'run_dapo_4node_{preset}.sh')],
                          cwd='/tmp', env=env, capture_output=True, text=True)


@pytest.mark.parametrize('preset,micro,logprob,padding', [
    ('qwen3vl_4b', '16', '32', 'True'), ('qwen3vl_8b', '16', '32', 'True'),
    ('qwen35_4b', '4', '8', 'False'), ('qwen35_9b', '2', '4', 'False'),
])
def test_presets_produce_valid_hydra_arguments(preset, micro, logprob, padding):
    result = launch(preset)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    OverridesParser.create().parse_overrides(args[3:])
    with initialize_config_dir(config_dir=str(RECIPE.parents[1] / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=args[3:])
    assert config.algorithm.filter_groups.enable
    assert config.actor_rollout_ref.rollout.agent.default_agent_loop == 'vln_full_episode_agent'
    values = dict(arg.split('=', 1) for arg in args[3:])
    assert values['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'] == micro
    assert values['actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu'] == logprob
    assert values['actor_rollout_ref.model.use_remove_padding'] == padding
    assert Path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path).is_file()
    assert values['trainer.logger'] == '[console]'
    assert values['actor_rollout_ref.rollout.max_model_len'] == '4352'


@pytest.mark.parametrize('override', [
    {'NGPUS': '0'}, {'NUM_WORKERS': '0'}, {'ROLLOUT_TP': '3'},
    {'ROLLOUT_SCHEDULER': 'bad'}, {'MAX_MODEL_LEN': '4096'},
])
def test_invalid_settings_fail_before_startup(override):
    assert launch(**override).returncode != 0


@pytest.mark.parametrize('window', [2, 9])
def test_window_size_does_not_have_to_divide_worker_count(window):
    result = launch(ROLLOUT_SCHEDULER='window', NUM_WORKERS='8', ROLLOUT_WINDOW=window)
    assert result.returncode == 0, result.stderr


def test_remote_url_and_filter_flag_reach_ray_workers():
    result = launch(VLN_ENV_SERVER_URL='http://env.example:8002', FILTER_GROUPS_ENABLE='false')
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    parsed = OverridesParser.create().parse_overrides(args[3:])
    values = {item.key_or_group: item.value() for item in parsed}
    assert values['ray_kwargs.ray_init.runtime_env.env_vars.VLN_ENV_SERVER_URL'] == 'http://env.example:8002'
    assert values['algorithm.filter_groups']['enable'] is False
    assert 'data.gen_batch_size' not in values


@pytest.mark.parametrize('name', [
    'model with spaces', 'model,checkpoint', "owner's-model", 'checkpoint=final',
    'model[final]{v1}', 'model"quoted"', 'model\\with\\slashes',
    'model\\"quoted"\\', '123',
])
def test_paths_and_names_remain_literal_hydra_strings(name):
    root = f'/tmp/{name}'
    result = launch(MODEL_PATH=root, TRAIN_FILE=f'{root}/train.parquet',
                    VAL_FILE=f'{root}/val.parquet', AGENT_CFG=f'{root}/agent.yaml',
                    CKPT_DIR=f'{root}/output', PROJECT=name, EXPERIMENT=name)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)[3:]
    with initialize_config_dir(config_dir=str(RECIPE.parents[1] / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=args)
    assert config.actor_rollout_ref.model.path == root
    assert list(config.data.train_files) == [f'{root}/train.parquet']
    assert config.data.val_files == f'{root}/val.parquet'
    assert config.actor_rollout_ref.rollout.agent.agent_loop_config_path == f'{root}/agent.yaml'
    assert config.trainer.default_local_dir == f'{root}/output'
    assert config.trainer.project_name == name
    assert config.trainer.experiment_name == name


def test_launch_creates_log_and_preserves_training_failure(tmp_path):
    model = tmp_path / 'model with spaces'
    model.mkdir()
    data = tmp_path / 'episodes.parquet'
    data.touch()
    fake_python = tmp_path / 'python'
    fake_python.write_text(f'#!{sys.executable}\nprint("trainer started")\nraise SystemExit(17)\n')
    fake_python.chmod(0o755)
    log = tmp_path / 'new' / 'train.log'
    result = launch(DRY_RUN='0', PYTHON=fake_python, MODEL_PATH=model,
                    TRAIN_FILE=data, VAL_FILE=data, LOG_FILE=log, RAY_TMPDIR=tmp_path / 'ray')
    assert result.returncode == 17
    assert 'trainer started' in log.read_text()
