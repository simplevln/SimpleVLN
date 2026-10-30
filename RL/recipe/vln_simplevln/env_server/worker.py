"""Habitat env worker — one process, one Habitat Env, one active session at a time.

IPC: parent sends dict commands via cmd_queue, worker sends dict responses via resp_queue.
Commands:
  {"op": "reset",    "episode_id": str, "scene_id": str, "session_id": str}
  {"op": "step",     "actions": [int, ...]}
  {"op": "metrics"}
  {"op": "close"}
Worker sends {"ok": bool, ...payload...} or {"ok": False, "error": str}.
"""
from __future__ import annotations

import base64
import io
import multiprocessing as mp
import gzip
import json
import os
import tempfile
import traceback

import numpy as np
from PIL import Image


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype("uint8")).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_obs(rgb: np.ndarray, instruction: str, step: int) -> dict:
    h, w = rgb.shape[:2]
    return {"rgb_jpeg_base64": _encode_jpeg_b64(rgb), "width": w, "height": h, "step": step,
            "instruction": instruction}


def _make_metrics(info: dict, collisions: int) -> dict:
    ne = float(info.get("distance_to_goal", 0.0))
    return {
        "distance_to_goal": ne,
        "success": float(info.get("success", 0.0)),
        "spl": float(info.get("spl", 0.0)),
        "oracle_success": float(info.get("oracle_success", 0.0)),
        "oracle_navigation_error": float(info.get("oracle_navigation_error", ne)),
        "collisions": float(collisions),
    }


def _scene_key(scene_id: str) -> str:
    """mp3d/5LpN.../5LpN....glb -> 5LpN... (matches manifest scene_id basename)."""
    return os.path.basename(str(scene_id)).split(".")[0]


def materialize_dataset_json(data_path: str, dataset_name: str, dest_dir: str) -> str:
    """Turn one dataset's rows from the shared parquet into a Habitat json.gz.

    The released RL package is a single parquet. Habitat still loads
    ``{"episodes": [...]}``. Non-parquet paths are returned unchanged.
    """
    path = str(data_path)
    if not path.endswith(".parquet"):
        return path
    import pyarrow.parquet as pq

    episodes = []
    for row in pq.read_table(path).to_pylist():
        extra = row.get("extra_info") or {}
        if str(extra.get("dataset_name")) != dataset_name:
            continue
        episode = row.get("episode")
        if isinstance(episode, str):
            episode = json.loads(episode)
        if not isinstance(episode, dict):
            raise ValueError(
                f"{path} has no Habitat episode for {dataset_name}. "
                "Rebuild the parquet with recipe.vln_simplevln.make_manifest."
            )
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"No episodes for {dataset_name} in {path}")
    out = os.path.join(dest_dir, f"{dataset_name}.json.gz")
    with gzip.open(out, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)
    return out


def _load_env(config_path: str):
    """Load the unified habitat config: one shared Env + every dataset listed under
    ``datasets:``. Episodes are indexed under a per-``dataset_name`` composite key so
    reset() can route requests by dataset_name even when scene::episode_id collides.
    """
    import habitat
    from habitat import Env
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig, FogOfWarConfig, TopDownMapMeasurementConfig,
    )
    from recipe.vln_simplevln.env_server import habitat_support
    _ = habitat_support  # Registers the two oracle measurements used for training.

    cfg = get_config(config_path)
    ds_specs = cfg.get("datasets")
    if not ds_specs or len(ds_specs) == 0:
        raise ValueError(
            f"{config_path} has no top-level `datasets:` map; env_server requires at least one entry."
        )

    with habitat.config.read_write(cfg):
        # collisions is read by step (); top_down_map is visualization-only and is
        # NOT used by training. Its fog_of_war path crashes under some numba/numpy
        # versions ("can't unbox array from PyObject"), so it is OFF by default.
        # Set VLN_ENABLE_TOP_DOWN_MAP=1 to re-enable (e.g. for rendering videos).
        measures_to_add = {"collisions": CollisionsMeasurementConfig()}
        if os.environ.get("VLN_ENABLE_TOP_DOWN_MAP", "0") == "1":
            measures_to_add["top_down_map"] = TopDownMapMeasurementConfig(
                map_padding=3, map_resolution=1024,
                draw_source=True, draw_border=True, draw_shortest_path=True,
                draw_view_points=True, draw_goal_positions=True, draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
            )
        cfg.habitat.task.measurements.update(measures_to_add)

    episode_cache = tempfile.mkdtemp(prefix="vln_rl_episodes_")

    def _override_dataset_section(ds_cfg, spec, name):
        """Copy per-dataset {type,split,scenes_dir,data_path} overrides into a dataset cfg."""
        with habitat.config.read_write(ds_cfg):
            for k in ("type", "split", "scenes_dir", "data_path"):
                if k not in spec:
                    continue
                value = spec[k]
                if k == "data_path":
                    value = materialize_dataset_json(str(value), name, episode_cache)
                setattr(ds_cfg, k, value)
        return ds_cfg

    ds_names = list(ds_specs.keys())
    primary_name = ds_names[0]

    # Build the primary dataset (Env is constructed from it).
    _override_dataset_section(cfg.habitat.dataset, ds_specs[primary_name], primary_name)
    dataset = habitat.datasets.make_dataset(
        id_dataset=cfg.habitat.dataset.type, config=cfg.habitat.dataset)

    # Episode identity includes the dataset; scene/id pairs overlap across sources.
    episodes_by_full: dict = {}
    episodes_by_bare: dict = {}

    def _index(ds, name: str):
        for ep in ds.episodes:
            sk = _scene_key(ep.scene_id)
            full_key = f"{name}::{sk}::{ep.episode_id}"
            if full_key in episodes_by_full:
                raise ValueError(f"Duplicate episode key: {full_key}")
            episodes_by_full[full_key] = ep
            bare_key = f"{name}::{ep.episode_id}"
            # Ambiguous IDs require an explicit scene rather than an arbitrary match.
            episodes_by_bare[bare_key] = None if bare_key in episodes_by_bare else ep

    _index(dataset, primary_name)

    # Merge every additional dataset. Build a fresh dataset-section copy for each, then
    # ``make_dataset`` produces its episode list. Episodes are indexed under the entry name.
    for name in ds_names[1:]:
        sub_cfg = get_config(config_path)
        _override_dataset_section(sub_cfg.habitat.dataset, ds_specs[name], name)
        ds2 = habitat.datasets.make_dataset(
            id_dataset=sub_cfg.habitat.dataset.type, config=sub_cfg.habitat.dataset)
        dataset.episodes.extend(ds2.episodes)
        _index(ds2, name)

    env = Env(config=cfg, dataset=dataset)
    return env, episodes_by_full, episodes_by_bare, primary_name


def worker_loop(worker_id: int, config_path: str,
                cmd_queue: mp.Queue, resp_queue: mp.Queue,
                gpu_id: int = -1):
    try:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env, episodes_by_full, episodes_by_bare, primary_name = _load_env(config_path)
        resp_queue.put({"ok": True, "msg": f"worker {worker_id} ready ({len(episodes_by_full)} episodes)"})
    except Exception:
        resp_queue.put({"ok": False, "error": traceback.format_exc()})
        return

    step_count = 0
    collisions = 0
    current_episode = None

    while True:
        cmd = cmd_queue.get()
        op = cmd.get("op")
        try:
            if op == "close":
                env.close()
                resp_queue.put({"ok": True})
                return

            if op == "reset":
                ep_id = str(cmd["episode_id"])
                scene = cmd.get("scene_id")
                name = cmd.get("dataset_name") or primary_name
                if scene:
                    ep = episodes_by_full.get(f"{name}::{_scene_key(scene)}::{ep_id}")
                else:
                    ep = episodes_by_bare.get(f"{name}::{ep_id}")
                if ep is None:
                    resp_queue.put({"ok": False, "error": f"episode_id {ep_id} (scene={scene}, dataset_name={name}) not found"})
                    continue
                env.current_episode = ep
                obs = env.reset()
                current_episode = ep
                step_count = 0
                collisions = 0
                scene_id = os.path.basename(ep.scene_id).split(".")[0]
                info = env.get_metrics()
                resp_queue.put({
                    "ok": True,
                    "obs": _make_obs(obs["rgb"], obs["instruction"]["text"], step_count),
                    "metrics": _make_metrics(info, collisions),
                    "scene_id": scene_id,
                    "episode_id": ep_id,
                    "instruction": obs["instruction"]["text"],
                    "done": env.episode_over,
                })
                continue

            if op == "step":
                actions = cmd["actions"]
                last_obs = None
                done = False
                done_reason = None
                executed_actions = []
                for action in actions:
                    obs = env.step({"action": int(action)})
                    last_obs = obs
                    executed_actions.append(int(action))
                    step_count += 1
                    done = env.episode_over
                    if action == 0:
                        done_reason = "stop_action"
                    if done:
                        done_reason = done_reason or "episode_over"
                        break
                info = env.get_metrics()
                raw_col = info.get("collisions", {})
                collisions = int(raw_col.get("count", 0)) if isinstance(raw_col, dict) else 0
                resp_queue.put({
                    "ok": True,
                    "obs": _make_obs(last_obs["rgb"], last_obs["instruction"]["text"], step_count),
                    "metrics": _make_metrics(info, collisions),
                    "done": done,
                    "done_reason": done_reason,
                    "executed_actions": executed_actions,
                })
                continue

            if op == "metrics":
                info = env.get_metrics() if current_episode is not None else {}
                resp_queue.put({"ok": True, "metrics": _make_metrics(info, collisions),
                                "done": env.episode_over if current_episode else False})
                continue

            if op == "episodes":
                limit = int(cmd.get("limit", 200))
                name = cmd.get("dataset_name")
                keys = [key for key in episodes_by_full if name is None or key.startswith(f"{name}::")]
                resp_queue.put({"ok": True, "episode_ids": keys[:limit]})
                continue

            resp_queue.put({"ok": False, "error": f"unknown op: {op}"})

        except Exception:
            resp_queue.put({"ok": False, "error": traceback.format_exc()})
