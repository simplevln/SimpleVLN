"""Habitat env worker — one process, one Habitat Env, one active session at a time.

IPC: parent sends dict commands via cmd_queue, worker sends dict responses via resp_queue.
Commands:
  {"op": "reset",    "episode_id": str, "scene_id": str, "instruction": str, "session_id": str}
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


def materialize_dataset_json(data_path: str, dest_dir: str) -> str:
    """Turn the training parquet into the Habitat json.gz the simulator loads.

    Non-parquet paths are returned unchanged.
    """
    path = str(data_path)
    if not path.endswith(".parquet"):
        return path
    import pyarrow.parquet as pq

    episodes = []
    for row in pq.read_table(path).to_pylist():
        episode = row.get("episode")
        if isinstance(episode, str):
            episode = json.loads(episode)
        if not isinstance(episode, dict):
            raise ValueError(
                f"{path} has no Habitat episode record. "
                "Rebuild the parquet with recipe.vln_simplevln.make_manifest."
            )
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"No episodes in {path}")
    out = os.path.join(dest_dir, "vln_train.json.gz")
    with gzip.open(out, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)
    return out


def _episode_instruction(episode) -> str:
    instruction = getattr(episode, "instruction", None)
    if instruction is None and isinstance(episode, dict):
        instruction = episode.get("instruction")
    text = getattr(instruction, "instruction_text", None)
    if text is None and isinstance(instruction, dict):
        text = instruction.get("instruction_text", "")
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return str(text or "")


def _select_episode(matches, instruction):
    if not matches:
        return None
    if instruction:
        exact = [episode for episode in matches if _episode_instruction(episode) == instruction]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return None
    return matches[0] if len(matches) == 1 else None


def _load_env(config_path: str):
    """Load the training parquet into one Habitat Env.

    Episodes are indexed by scene and episode id. When that pair is shared,
    reset() uses the instruction text.
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
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.data_path = materialize_dataset_json(
            str(cfg.habitat.dataset.data_path), episode_cache)
    dataset = habitat.datasets.make_dataset(
        id_dataset=cfg.habitat.dataset.type, config=cfg.habitat.dataset)

    # A few scene and episode-id pairs occur twice. The instruction picks one.
    episodes_by_full: dict = {}
    episodes_by_bare: dict = {}
    for episode in dataset.episodes:
        full_key = f"{_scene_key(episode.scene_id)}::{episode.episode_id}"
        episodes_by_full.setdefault(full_key, []).append(episode)
        episodes_by_bare.setdefault(str(episode.episode_id), []).append(episode)

    env = Env(config=cfg, dataset=dataset)
    return env, episodes_by_full, episodes_by_bare


def worker_loop(worker_id: int, config_path: str,
                cmd_queue: mp.Queue, resp_queue: mp.Queue,
                gpu_id: int = -1):
    try:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env, episodes_by_full, episodes_by_bare = _load_env(config_path)
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
                instruction = cmd.get("instruction")
                matches = (episodes_by_full.get(f"{_scene_key(scene)}::{ep_id}")
                           if scene else episodes_by_bare.get(ep_id))
                ep = _select_episode(matches or [], instruction)
                if ep is None:
                    resp_queue.put({"ok": False, "error": f"episode_id {ep_id} (scene={scene}) not found"})
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
                resp_queue.put({"ok": True, "episode_ids": list(episodes_by_full)[:limit]})
                continue

            resp_queue.put({"ok": False, "error": f"unknown op: {op}"})

        except Exception:
            resp_queue.put({"ok": False, "error": traceback.format_exc()})
