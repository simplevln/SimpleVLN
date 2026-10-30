import os
import copy
from contextlib import closing
from collection_io import (
    append_records, completed_keys, deduplicate_records, load_records,
    merge_annotations, write_snapshot,
)
from urllib.parse import urlsplit

os.environ.setdefault("HABITAT_SIM_MULTI_THREAD", "1")

import json
import argparse
import multiprocessing as mp
import logging

import re
import random
import numpy as np
import tqdm
import habitat
import time
import gzip
from PIL import Image
from omegaconf import OmegaConf
from typing import Dict
import io

from habitat.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image, append_text_underneath_image
from habitat_extensions import measures
_ = measures  # Registers the collection measurements with Habitat.

import base64
from openai import OpenAI
DATASET = "rxr"

MIDGOAL_RADIUS = 0.5
GOAL_RADIUS = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.85

class DAggerCollector:
    def __init__(self, args, rank, world_size, vllm_clients=None, vllm_model_name="qwen3vl"):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.vllm_clients = vllm_clients or []
        self.vllm_model_name = vllm_model_name
        logging.getLogger("habitat").setLevel(logging.ERROR)
        logging.getLogger("habitat_sim").setLevel(logging.ERROR)

        self.dataset = self.args.dagger_dataset.lower()
        self.output_path = self.args.dagger_output_path
        self.data_path = self.args.dagger_data_path
        self.config = get_habitat_config(args.habitat_config_path)
        if self.config.habitat.simulator.forward_step_size != 0.25 or self.config.habitat.simulator.turn_angle != 15:
            raise ValueError("Converters require 0.25 m forward steps and 15 degree turns")

        gt_path = self.args.dagger_gt_annotations_path
        gt_open = gzip.open if str(gt_path).endswith('.gz') else open
        with gt_open(gt_path, 'rt', encoding='utf-8') as f:
            gt_data = json.load(f)
        gt_map = {}
        if isinstance(gt_data, dict):
            # Support both {"episode_id": annotation, ...} and wrapped list formats.
            candidate_list = None
            for key in ("annotations", "episodes", "data"):
                if isinstance(gt_data.get(key), list):
                    candidate_list = gt_data[key]
                    break
            if candidate_list is None:
                for epid, item in gt_data.items():
                    if isinstance(item, dict):
                        gt_map[str(epid)] = item
            else:
                for item in candidate_list:
                    if not isinstance(item, dict):
                        continue
                    epid = item.get("episode_id", item.get("id"))
                    if epid is None:
                        continue
                    gt_map[str(epid)] = item
        else:
            for item in gt_data:
                if not isinstance(item, dict):
                    continue
                epid = item.get("episode_id", item.get("id"))
                if epid is None:
                    continue
                gt_map[str(epid)] = item
        self.gt_annotations = gt_map
        
        with read_write(self.config):
            measurements = {}
            if self.args.dagger_save_video:
                measurements.update(
                    {
                        "top_down_map": TopDownMapMeasurementConfig(
                            map_padding=3,
                            map_resolution=1024,
                            draw_source=True,
                            draw_border=True,
                            draw_shortest_path=True,
                            draw_view_points=True,
                            draw_goal_positions=True,
                            draw_goal_aabbs=True,
                            fog_of_war=FogOfWarConfig(
                                draw=True,
                                visibility_dist=5.0,
                                fov=90,
                            ),
                        ),
                        "collisions": CollisionsMeasurementConfig(),
                    }
                )
            if measurements:
                self.config.habitat.task.measurements.update(measurements)

        self.dagger_config = OmegaConf.create({
            "p": self.args.dagger_p,
            "update_size": self.args.dagger_update_size,
        })
        print(self.dagger_config)

    def _rgb_to_data_url(self, rgb) -> str:
        # Match Eval's JPEG encoding defaults.
        img = Image.fromarray(rgb).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def uniform_sample_with_ends(data, n):
        if len(data) <= n:
            return data
        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]

    def _call_vllm_eval_style(self, rgb_list, instruction, episode_id=0):
        """Call vLLM with eval-style single-turn batched-history messages."""
        eval_messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a visual language navigation model."}]}
        ]
        content = []
        content.append({"type": "text", "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations"})

        if len(rgb_list) > 1:
            for img in self.uniform_sample_with_ends(rgb_list[:-1], 8):
                data_url = self._rgb_to_data_url(np.array(img))
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            data_url = self._rgb_to_data_url(np.array(rgb_list[-1]))
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        content.append({"type": "text", "text": "and an image of the current observation"})
        data_url = self._rgb_to_data_url(np.array(rgb_list[-1]))
        content.append({"type": "image_url", "image_url": {"url": data_url}})

        instr = instruction if isinstance(instruction, str) else instruction[0]
        content.append({"type": "text", "text": f" Your assigned task is: '{instr}'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."})

        eval_messages.append({"role": "user", "content": content})

        client = self.vllm_clients[hash(episode_id) % len(self.vllm_clients)]
        model_name = self.vllm_model_name
        outputs = client.chat.completions.create(
            messages=eval_messages,
            model=model_name,
            max_completion_tokens=512,
            temperature=0.3,
            top_p=0.95,
        )

        output_text = outputs.choices[0].message.content
        output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
        return output_text.strip()

    @staticmethod
    def _extract_result(output, forward_distance=25, turn_angle=15):
        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()
        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, forward_distance
            return 1, float(match.group())
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, turn_angle
            return 2, float(match.group())
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, turn_angle
            return 3, float(match.group())
        return None, None

    @staticmethod
    def _extract_multi_result(output, forward_distance=25, turn_angle=15):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = DAggerCollector._extract_result(sub_action, forward_distance, turn_angle)
            result.append([action_index, numeric])
        return result

    @staticmethod
    def _expand_actions(parsed_result, forward_distance=25, turn_angle=15):
        action_list = []
        for action_index, numeric in parsed_result[:2]:
            if action_index == 0:
                action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric / forward_distance))):
                    action_list.append(1)
            elif action_index == 2:
                for _ in range(min(3, round(numeric / turn_angle))):
                    action_list.append(2)
            elif action_index == 3:
                for _ in range(min(3, round(numeric / turn_angle))):
                    action_list.append(3)
            if action_index is None or not action_list:
                action_list.append(random.randint(1, 3))
        return action_list

    def config_env(self) -> habitat.Env:
        config = copy.deepcopy(self.config)
        if self.data_path is not None:
            with read_write(config):
                config.habitat.dataset.data_path = self.data_path
        return habitat.Env(config=config)

    def generate(self, env: habitat.Env, save_video: bool = True, force_expert: bool = False) -> Dict:
        """
        Generate a trajectory for the given episode.
        """
        beta = 0 if self.dagger_config.p == 0 else self.dagger_config.p ** self.args.dagger_data_it
        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        episode = env.current_episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = episode.trajectory_id
        instructions = episode.instruction.instruction_text
        ref_path = episode.reference_path

        observation = env.reset()
        annotation_actions = []
        rgb_data_list, rgb_list = [], []
        step_id = 0
        actions = [-1]
        executed_actions = []
        active_model_group = None
        next_waypoint_id = 1

        if save_video:
            os.makedirs(os.path.join(self.output_path, 'videos'), exist_ok=True)

        vis_frames = []
        from_expert = bool(force_expert)
        force_episode_end = False
        model_success = True
        action_seq = []
        metrics = None
        accumulated_error = 0 

        ref_actions_len = len(self.gt_annotations[str(episode_id)]["actions"])

        trace_steps = []
        model_call_groups = []
        instr_str = instructions if isinstance(instructions, str) else (instructions[0] if instructions else "")
        while not env.episode_over:
            if not action_seq:
                active_model_group = None
            step_trace_llm = None
            generated_this_round = False
            generated_branch = None
            generated_actions = None
            rgb = observation["rgb"]

            if self.args.collect_images:
                rgb_path = os.path.join(
                    self.output_path,
                    scene_id,
                    f"{episode_id:06d}",
                    "rgb",
                    f"{step_id:03d}.jpg",
                )
                rgb_data_list.append((rgb, rgb_path))

            image = Image.fromarray(rgb).convert("RGB")
            rgb_list.append(image)
            if len(rgb_list) > self.args.max_action_history:
                rgb_list = rgb_list[1:]

            if not action_seq:
                from_expert = True if force_expert else random.random() < beta
                if from_expert:
                    action = agent.get_next_action(ref_path[next_waypoint_id])
                    action_seq = [action]
                    generated_this_round = True
                    generated_branch = "expert"
                    generated_actions = list(action_seq)
                else:
                    llm_text = self._call_vllm_eval_style(rgb_list, instructions, episode_id=episode_id)
                    action_seq = self._expand_actions(self._extract_multi_result(llm_text))
                    if self.args.dagger_save_model_trace:
                        step_trace_llm = llm_text
                    chat_action_seq = []
                    for a in action_seq:
                        if a == 0:
                            fallback_action = agent.get_next_action(ref_path[next_waypoint_id])
                            chat_action_seq.append(fallback_action if fallback_action != 0 else 1)
                        else:
                            chat_action_seq.append(a)
                    action_seq = chat_action_seq

                    active_model_group = {
                        "call_step_id": step_id,
                        "model_output": llm_text,
                        "planned_actions": list(chat_action_seq),
                        "actions": [],
                        "executed_actions": [],
                    }
                    model_call_groups.append(active_model_group)

                    generated_this_round = True
                    generated_branch = "model"
                    generated_actions = list(chat_action_seq)

            action_source = "expert" if from_expert else "model"

            if len(action_seq) == 0:
                action_seq = [0]

            action = action_seq.pop(0)
            expert_action = agent.get_next_action(ref_path[next_waypoint_id])
            if action != expert_action:
                accumulated_error += 1

            while agent.get_next_action(ref_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                force_expert = False
                if next_waypoint_id == len(ref_path) - 1:
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_waypoint_id >= len(ref_path):
                    force_episode_end = True
                    action = 0
                    action_source = "expert"
                    break

            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(ref_path)

            accumulated_error_ratio_denom = max(1, int(ref_actions_len / (len(ref_path) - 1)))
            accumulated_error_ratio = accumulated_error / accumulated_error_ratio_denom
            error_not_toleranted = (
                (from_expert == False and action == 0 and metrics["distance_to_goal"] >= 3.0)
                or (accumulated_error_ratio > 0.8)
                or accumulated_error > 12
            )
            acc_err_before_reset = None

            if wp_id_available and error_not_toleranted:
                model_success = False
                acc_err_before_reset = int(accumulated_error)
                accumulated_error = 0
                force_expert = True
                action_source = "expert"

                action = agent.get_next_action(ref_path[next_waypoint_id])
                if action == 0 and not force_episode_end:
                    action = 1
                action_seq = []

                generated_this_round = True
                generated_branch = "expert"
                generated_actions = [int(action)]
            if action == 0 and not force_episode_end:
                action = agent.get_next_action(ref_path[next_waypoint_id])
                if action == 0 and not force_episode_end:
                    action = 1

            if self.args.dagger_save_model_trace and generated_this_round:
                pre_m = env.get_metrics()
                trace_entry = {
                    "step_id": step_id,
                    "decision_branch": generated_branch,
                    "distance_to_goal": float(pre_m["distance_to_goal"]),
                    "instruction": instr_str,
                    "actions": [int(x) for x in generated_actions] if generated_actions is not None else None,
                    "model_raw": step_trace_llm if generated_branch == "model" else None,
                    "expert_action": int(expert_action),
                    "next_waypoint_id": int(next_waypoint_id),
                    "ref_path_len": int(len(ref_path)),
                    "from_expert": bool(from_expert),
                    "force_episode_end": bool(force_episode_end),
                    "accumulated_error": int(accumulated_error),
                    "accumulated_error_ratio_denom": int(accumulated_error_ratio_denom),
                    "accumulated_error_ratio": float(accumulated_error_ratio),
                }
                if acc_err_before_reset is not None:
                    trace_entry["accumulated_error_before_reset"] = acc_err_before_reset
                trace_steps.append(trace_entry)

            observation = env.step(action)
            executed_actions.append(int(action))
            if active_model_group is not None:
                active_model_group["actions"].append(int(action))
                active_model_group["executed_actions"].append(int(action))
            metrics = env.get_metrics()

            if save_video:
                metrics = env.get_metrics()
                if metrics['top_down_map'] is not None:
                    _pil = Image.fromarray(observation['rgb'])
                    resized_rgb = np.array(_pil.resize((int(_pil.width * 1.6), int(_pil.height * 1.6)), Image.LANCZOS))
                    frame = observations_to_image({'rgb': resized_rgb}, metrics)
                    frame = append_text_underneath_image(frame, episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                    frame = append_text_underneath_image(frame, action_source)
                    frame = append_text_underneath_image(frame, f"force_expert is {force_expert}")
                    frame = append_text_underneath_image(frame, f"step: {step_id}")
                    frame = append_text_underneath_image(frame, f"next wp id: {next_waypoint_id} / {len(ref_path) - 1}")
                    vis_frames.append(frame)

            if env.episode_over or force_episode_end:
                break
            actions.append(action)
            step_id += 1

        if self.args.collect_images:
            assert len(rgb_data_list) == len(actions), f"Length of rgbs and actions mismatch, rgb_data_list: {len(rgb_data_list)}, actions: {(actions)}"

        annotation_actions.append({
            "id": episode_id,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            "video": f"{scene_id}/{episode_id:06d}",
            "instructions": [instructions] if isinstance(instructions, str) else instructions,
            "actions": actions,
            "executed_actions": executed_actions,
            "terminal_action": executed_actions[-1],
            "model_call_groups": model_call_groups,
        })

        episode_save = bool(executed_actions[-1] == 0 and metrics["distance_to_goal"] < MIDGOAL_RADIUS and (((not model_success) and (metrics["pl"] < RELATIVE_PATH_LENGTH_THRESHOLD)) or (metrics["pl"] < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD)))
        if episode_save and self.args.collect_images:
            os.makedirs(os.path.join(self.output_path, scene_id, f"{episode_id:06d}", "rgb"), exist_ok=True)

            for rgb, rgb_path in rgb_data_list:
                Image.fromarray(rgb).convert("RGB").save(rgb_path)

        if save_video:
            if episode_save:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'save_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()
            else:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'notsave_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()

        metrics.update({
            "step_id": step_id,
            "ref_actions_len": ref_actions_len,
            "accumulated_error": accumulated_error,
            "save": int(episode_save),
            "model_success": model_success,
            "force_episode_end": force_episode_end,
            }
        )

        if self.args.dagger_save_model_trace and trace_steps:
            trace_dir = os.path.join(self.output_path, "model_traces")
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(
                trace_dir,
                f"{scene_id}_{episode_id:06d}_{trajectory_id}_r{self.rank}.json",
            )
            with open(trace_file, "w", encoding="utf-8") as tf:
                json.dump(
                    {
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "instruction": instr_str,
                        "steps": trace_steps,
                    },
                    tf,
                    ensure_ascii=False,
                    indent=2,
                )

        episode_dict = dict(
            anno_actions=annotation_actions,
            metrics=metrics,
        )

        return episode_dict

    def update_dataset(self):
        """Update dataset with the collected data."""

        seed = self.rank
        random.seed(seed)
        np.random.seed(seed)

        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        dataset = habitat.make_dataset(
            id_dataset=self.config.habitat.dataset.type, config=self.config.habitat.dataset
        )
        scene_episode_dict = {}
        start = time.time()
        for episode in dataset.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)
        assigned_episodes = []
        for scene_id in sorted(scene_episode_dict):
            assigned_episodes.extend(scene_episode_dict[scene_id][self.rank::self.world_size])

        # A saved episode is complete only when annotations and frames also exist.
        processed_keys = completed_keys(self.output_path, self.args.annotations_actions,
                                        require_images=self.args.collect_images)

        if processed_keys:
            assigned_episodes = [
                ep
                for ep in assigned_episodes
                if (
                    ep.scene_id.split('/')[-2],
                    str(ep.episode_id),
                    str(ep.trajectory_id),
                ) not in processed_keys
            ]
        num_collect_episodes = 0
        assigned_total = len(assigned_episodes)
        if not assigned_total:
            print(f"rank{self.rank}: no pending episodes")
            return
        bo_n = max(1, int(getattr(self.args, 'dagger_bo_n', 1)))


        ann_prefix = self.args.annotations_actions
        rank_jsonl = os.path.join(self.output_path, f"{ann_prefix}_{self.rank}.jsonl")
        rank_json = os.path.join(self.output_path, f"{ann_prefix}_{self.rank}.json")
        # One-time: legacy .json shard -> .jsonl (append path for this run; avoids full re-read on each commit)
        if os.path.exists(rank_json) and (not os.path.exists(rank_jsonl)):
            append_records(rank_jsonl, load_records(rank_json, strict_json=True))

        with closing(self.config_env()) as env, tqdm.tqdm(
            total=assigned_total,
            dynamic_ncols=True,
            desc=f"rank{self.rank}",
            disable=(self.rank != 0),
        ) as pbar:
            processed_episodes = 0
            stop_collect = False

            for ep in assigned_episodes:
                if stop_collect:
                    break

                last_episode_dagger = None
                skip = False
                for attempt in range(bo_n):
                    try:
                        env.current_episode = ep
                        env.current_episode.goals[0].radius = MIDGOAL_RADIUS
                        episode_dagger = self.generate(
                            env=env,
                            save_video=self.args.dagger_save_video,
                            force_expert=self.args.force_expert,
                        )
                    except ZeroDivisionError:
                        skip = True
                        break
                    last_episode_dagger = episode_dagger
                    if episode_dagger['metrics']['save']:
                        break

                processed_episodes += 1

                if skip or last_episode_dagger is None:
                    pbar.update()
                    pbar.set_postfix({
                        "save": f"{num_collect_episodes}/{self.dagger_config.update_size}",
                        "processed": f"{processed_episodes}/{assigned_total}",
                    })
                    if processed_episodes >= self.dagger_config.update_size:
                        stop_collect = True
                    continue

                scan = ep.scene_id.split('/')[-2]

                if last_episode_dagger['metrics']['save']:
                    append_records(rank_jsonl, last_episode_dagger['anno_actions'])

                rel_pl = last_episode_dagger['metrics'].get('pl', None)
                result = {
                    'scene': scan,
                    'episode_id': ep.episode_id,
                    'trajectory_id': ep.trajectory_id,
                    'save': last_episode_dagger['metrics']['save'],
                    'model_success': last_episode_dagger['metrics']['model_success'],
                    'success': last_episode_dagger['metrics']['success'],
                    'distance_to_goal': last_episode_dagger['metrics'].get('distance_to_goal', None),
                    'relative_pl': rel_pl,
                    'step_id': last_episode_dagger['metrics']['step_id'],
                    'ref_actions': last_episode_dagger['metrics']['ref_actions_len'],
                    'accumulated_error': last_episode_dagger['metrics']['accumulated_error'],
                    'force_episode_end': last_episode_dagger['metrics']['force_episode_end'],
                }
                append_records(os.path.join(self.output_path, "result.json"), [result])

                if not last_episode_dagger['metrics']['save']:
                    pbar.update()
                    pbar.set_postfix({
                        "save": f"{num_collect_episodes}/{self.dagger_config.update_size}",
                        "processed": f"{processed_episodes}/{assigned_total}",
                    })
                    if processed_episodes >= self.dagger_config.update_size:
                        stop_collect = True
                    continue

                print(
                    f"model_success = {last_episode_dagger['metrics']['model_success']}, "
                    f"scene {scan} id {ep.episode_id} trajectory {ep.trajectory_id}"
                )

                num_collect_episodes += 1
                pbar.update()
                pbar.set_postfix({
                    "save": f"{num_collect_episodes}/{self.dagger_config.update_size}",
                    "processed": f"{processed_episodes}/{assigned_total}",
                })

                if processed_episodes >= self.dagger_config.update_size:
                    stop_collect = True

        if os.path.exists(rank_jsonl):
            records = load_records(rank_jsonl)
            if records:
                previous = load_records(rank_json, strict_json=True)
                write_snapshot(rank_json, deduplicate_records(previous + records))

        print(f"rank{self.rank}: save total episodes {num_collect_episodes} time cost {time.time() - start}")


def parse_args():

    parser = argparse.ArgumentParser()

    # VLLM
    parser.add_argument("--vllm_host", type=str, default="localhost",
                        help="vLLM server host")
    parser.add_argument("--vllm_port_start", type=int, default=22001,
                        help="starting port of vLLM servers")
    parser.add_argument("--vllm_num_servers", type=int, default=1,
                        help="number of vLLM servers (ports: port_start ~ port_start+num-1)")
    parser.add_argument("--vllm_model_name", type=str, default="qwen3vl")

    # Habitat
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_dagger.yaml')
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--max-action-history", type=int, default=200)

    # DAgger
    parser.add_argument("--dagger_p",type=float, default=0.9)
    parser.add_argument("--dagger_update_size", type=int, default=1)
    parser.add_argument("--dagger_dataset", type=str, default=DATASET)
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--dagger_bo_n", type=int, default=1,
                        help="best-of-n: if a trajectory is not saved, retry up to n times")
    parser.add_argument("--dagger_output_path",type=str, default="data/dagger")
    parser.add_argument("--dagger_data_path", type=str, default="data/datasets/vln_datasets/{split}.json.gz")
    parser.add_argument("--dagger_gt_annotations_path", type=str, default="data/datasets/vln_datasets/annotations.json")
    parser.add_argument(
        "--annotations_actions",
        type=str,
        default="annotations_actions",
        help="action annotation file prefix: append-only <prefix>_<rank>.jsonl; shard <prefix>_<rank>.json from jsonl; merge <prefix>.json on rank0",
    )
    parser.add_argument(
        "--dagger_save_video",
        action="store_true",
        default=False,
        help="whether to save video during dagger collection",
    )
    parser.add_argument(
        "--collect_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to save per-step RGB images under dagger_output_path/images",
    )
    parser.add_argument(
        "--dagger-save-model-trace",
        action="store_true",
        default=False,
        dest="dagger_save_model_trace",
        help="Write per-episode JSON traces (model vs expert, actions) under dagger_output_path/model_traces/",
    )

    return parser


def _dagger_worker(worker_id, num_workers, args, vllm_urls, vllm_model_name):
    """Worker process for multiprocessing-based DAgger collection."""
    logging.getLogger("habitat").setLevel(logging.ERROR)
    logging.getLogger("habitat_sim").setLevel(logging.ERROR)

    vllm_clients = [OpenAI(base_url=url, api_key="dummy", timeout=600) for url in vllm_urls]
    collector = DAggerCollector(
        args=args, rank=worker_id, world_size=num_workers,
        vllm_clients=vllm_clients, vllm_model_name=vllm_model_name,
    )
    collector.update_dataset()


if __name__ == "__main__":

    parser = parse_args()
    args = parser.parse_args()
    if args.max_action_history < 1 or args.parallel_envs < 1 or args.vllm_num_servers < 1:
        parser.error("history, worker and server counts must be positive")
    if args.dagger_bo_n < 1 or args.dagger_update_size < 1:
        parser.error("DAgger counts must be positive")
    if not 0 <= args.dagger_p <= 1 or args.dagger_data_it < 0:
        parser.error("dagger_p must be in [0, 1] and dagger_data_it must be nonnegative")

    hosts = [h.strip() for h in args.vllm_host.split(",") if h.strip()]
    vllm_urls = []
    for host in hosts:
        parsed = urlsplit(host if "://" in host else "http://" + host)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.port or parsed.path.rstrip("/"):
            parser.error("--vllm_host must contain hostnames, optionally with http:// or https://")
        hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        for i in range(args.vllm_num_servers):
            port = args.vllm_port_start + i
            if not 1 <= port <= 65535:
                parser.error("vLLM ports must be between 1 and 65535")
            vllm_urls.append(f"{parsed.scheme}://{hostname}:{port}/v1")
    if not vllm_urls:
        parser.error("At least one vLLM host is required")

    num_workers = max(1, args.parallel_envs)
    print(f"Using {len(vllm_urls)} vLLM server(s): {vllm_urls}")
    print(f"Spawning {num_workers} worker process(es)")

    if num_workers == 1:
        _dagger_worker(0, 1, args, vllm_urls, args.vllm_model_name)
    else:
        processes = []
        for i in range(num_workers):
            p = mp.Process(
                target=_dagger_worker,
                args=(i, num_workers, args, vllm_urls, args.vllm_model_name),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"DAgger workers failed: {failed}; annotations were not merged")

    merge_annotations(args.dagger_output_path, args.annotations_actions)
