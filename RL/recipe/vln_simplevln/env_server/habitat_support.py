"""Oracle navigation measurements used by the SimpleVLN training environments.

Adapted from the VLN-CE evaluation measurements. Standard Habitat R2RVLN-v1
loads the common episode schema used by all five released training sources.
"""

from dataclasses import dataclass

from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal
from hydra.core.config_store import ConfigStore


@registry.register_measure
class OracleNavigationError(Measure):
    cls_uuid = "oracle_navigation_error"

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args, task, **kwargs):
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = min(self._metric, distance)


@registry.register_measure
class OracleSuccess(Measure):
    cls_uuid = "oracle_success"

    def __init__(self, *args, config, **kwargs):
        self._success_distance = config.success_distance
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args, task, **kwargs):
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or distance < self._success_distance)


@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 3.0


ConfigStore.instance().store(
    group="habitat/task/measurements",
    name="oracle_success",
    node=OracleSuccessMeasurementConfig,
)
