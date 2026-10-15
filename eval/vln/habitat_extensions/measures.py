import gzip
import json
from typing import Any, List, Union
from dataclasses import dataclass

import numpy as np
from fastdtw import fastdtw
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success
from hydra.core.config_store import ConfigStore
from numpy import ndarray
from omegaconf import DictConfig

from habitat_extensions.task import RxRVLNCEDatasetV1


def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        self._config = config
        self._success_distance = getattr(self._config, 'success_distance', 3.0)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < self._success_distance)


@registry.register_measure
class NDTW(Measure):
    """NDTW (Normalized Dynamic Time Warping)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "ndtw"

    def __init__(
        self, *args: Any, sim: Simulator, config: DictConfig, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.dtw_func = fastdtw
        self._success_distance = getattr(self._config, 'success_distance', 3.0)
        dataset = kwargs["dataset"]
        if hasattr(dataset, "config") and dataset.config is not None:
            split = dataset.config["split"]
        else:
            split = getattr(config, "split", "val_unseen")
        if "{role}" in config.gt_path:
            self.gt_json = {}
            for role in RxRVLNCEDatasetV1.annotation_roles:
                with gzip.open(
                    config.gt_path.format(split=split, role=role), "rt"
                ) as f:
                    self.gt_json.update(json.load(f))
        else:
            with gzip.open(
                config.gt_path.format(split=split), "rt"
            ) as f:
                self.gt_json = json.load(f)

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[episode.episode_id]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance
            / (len(self.gt_locations) * self._success_distance)
        )
        self._metric = nDTW


@registry.register_measure
class PathLength(Measure):
    """Path Length (PL)
    PL = sum(euclidean_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

@registry.register_measure
class SDTW(Measure):
    """SDTW (Success Weighted be nDTW)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "sdtw"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [NDTW.cls_uuid, Success.cls_uuid]
        )
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        nDTW = task.measurements.measures[NDTW.cls_uuid].get_metric()
        self._metric = ep_success * nDTW

cs = ConfigStore.instance()


@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 3.0


@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"
    success_distance: float = 3.0
    gt_path: str = ""


cs.store(
    group="habitat/task/measurements",
    name="oracle_success",
    node=OracleSuccessMeasurementConfig,
)
cs.store(
    group="habitat/task/measurements",
    name="ndtw",
    node=NDTWMeasurementConfig,
)
