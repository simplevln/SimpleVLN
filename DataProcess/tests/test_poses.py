from contextlib import nullcontext
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tool'))
import trajectory_making as poses


class PoseTests(unittest.TestCase):
    def test_numpy_quaternion_matches_list_and_quaternion_object(self):
        value = 2 ** -.5
        expected = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        for quaternion in ([0., 0., value, value], np.array([0., 0., value, value]),
                           NS(w=value, x=0., y=0., z=value), NS(real=value, imag=[0., 0., value])):
            np.testing.assert_allclose(poses._quat_to_rotmat(quaternion), expected, atol=1e-12)

    def test_missing_episode_fails_and_closes_environment(self):
        closed = []
        environment = NS(episodes=[], close=lambda: closed.append(True))
        config = NS(habitat=NS(dataset=NS(data_path=''), environment=NS(max_episode_steps=400),
            simulator=NS(agents=NS(main_agent=NS(sim_sensors=NS(rgb_sensor=NS(width=640, height=480, hfov=90)))))))
        habitat_config = NS(read_write=lambda cfg: nullcontext())
        modules = {
            'habitat': NS(Env=lambda **kwargs: environment, config=habitat_config),
            'habitat.config': habitat_config,
            'habitat.config.default': NS(get_config=lambda path: config),
            'habitat.datasets': NS(make_dataset=None),
            'vln.habitat_extensions': NS(measures=None),
        }
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules), patch.dict(os.environ):
            dataset_path = str(Path(folder) / "dataset's final,version=1.json.gz")
            with self.assertRaisesRegex(ValueError, 'Episode 7 is missing'):
                poses._worker_process_episodes((0, 'envdrop', folder, dataset_path, [7], {}))
            self.assertEqual(config.habitat.dataset.data_path, dataset_path)
            self.assertEqual(config.habitat.environment.max_episode_steps, 0)
            self.assertEqual(closed, [True])


if __name__ == '__main__':
    unittest.main()
