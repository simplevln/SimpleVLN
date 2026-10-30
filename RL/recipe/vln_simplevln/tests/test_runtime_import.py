"""Check real entrypoint imports and runtime patch installation in a fresh process."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_training_runtime_imports_and_patch_installation():
    dependencies = ('torch', 'torchdata', 'tensordict', 'transformers', 'datasets',
                    'ray', 'cachetools', 'codetiming', 'peft', 'accelerate', 'openai')
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f'Requires training dependencies: {", ".join(missing)}')
    script = '''
import verl
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from recipe.vln_simplevln import vln_dapo_trainer as recipe
from recipe.vln_simplevln.online_rollout_manager import VLNOnlineRolloutManager
from recipe.vln_simplevln.verl_agent_loop import VLNFullEpisodeAgentLoop
assert RayPPOTrainer.fit is recipe.VLNDAPORayTrainer.fit
assert RayPPOTrainer._save_checkpoint is recipe.VLNDAPORayTrainer._save_checkpoint
assert RayPPOTrainer._save_checkpoint is not recipe._base_save_checkpoint
assert RayPPOTrainer._load_checkpoint is not recipe._base_load_checkpoint
assert verl.DataProto.union.__module__ == "verl.protocol"
print("runtime-import-ok")
'''
    result = subprocess.run(
        [sys.executable, '-B', '-c', script],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=60,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
             'VERL_USE_EXTERNAL_MODULES': 'recipe.vln_simplevln.patches,recipe.vln_simplevln.dapo_patch'})
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'runtime-import-ok' in result.stdout
