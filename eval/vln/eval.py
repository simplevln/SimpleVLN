"""Evaluate navigation episodes against OpenAI-compatible vLLM servers."""

import argparse
import base64
from io import BytesIO
import json
import multiprocessing as mp
import os
from queue import Empty
import random
import re
import time
from urllib.parse import quote, urlsplit

import cv2
from habitat import Env
import habitat
from habitat.core.agent import Agent
import imageio
import numpy as np
from habitat.utils.visualizations import maps
from habitat_extensions import measures, task
_ = (measures, task)  # Register custom measures and datasets with Habitat.
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image
from tqdm import tqdm
from result_store import append_result, load_result_rows, result_directory_lock
from openai import OpenAI


SYSTEM_PROMPT = "You are a visual language navigation model."

def encode_image_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def seed_all():
    np.random.seed(41)
    random.seed(41)

def evaluate_agent(result_queue, base_url, model_name, config, dataset, result_path,
                    forward_distance, turn_angle, max_action_history, resolution_ratio,
                    save_video, pass_k, done_rows) -> None:

    env = Env(config.habitat, dataset)

    try:
        agent = SimpleVLN_Agent(
            base_url,
            model_name,
            result_path,
            forward_distance,
            turn_angle,
            max_action_history,
            resolution_ratio,
            save_video=save_video)

        episodes_snapshot = list(env.episodes)
        
        EARLY_STOP_ROTATION = 25
        EARLY_STOP_STEPS = 400

        for target_ep in episodes_snapshot:
            for trial_id in range(pass_k):
                episode_start_time = time.time()
                scene_id = target_ep.scene_id.split('/')[-2]
                episode_id = target_ep.episode_id
                episode_instruction = target_ep.instruction.instruction_text
                key = (scene_id, str(episode_id), episode_instruction, trial_id, pass_k)
                if key in done_rows:
                    result_queue.put({"t_episode": 0, "skipped": 1,
                                      "success": float(done_rows[key]["success"])})
                    continue

                env.current_episode = target_ep
                obs = env.reset()
                iter_step = 0
                agent.reset()
                agent.video_name = (
                    f"{quote(scene_id, safe='')}_{quote(str(episode_id), safe='')}_"
                    f"pass{pass_k}_trial{trial_id}.gif"
                )
                t_dict = {"t_episode": 0}
                continuse_rotation_count = 0
                last_dtg = 999
                while not env.episode_over:
                    info = env.get_metrics()

                    if info["distance_to_goal"] != last_dtg:
                        last_dtg = info["distance_to_goal"]
                        continuse_rotation_count=0
                    else:
                        continuse_rotation_count +=1

                    action = agent.act(obs, info, env.current_episode.episode_id)

                    if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                        action = {"action": 0}

                    iter_step+=1
                    obs = env.step(action)

                info = env.get_metrics()
                result = {
                    "scene_id": scene_id,
                    "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
                    "trial_id": trial_id,
                    "trial_total": pass_k,
                    "success": info["success"],
                    "spl": info["spl"],
                    "os": info["oracle_success"],
                    "ne": info["distance_to_goal"],
                    "steps": iter_step,
                    "episode_instruction": episode_instruction
                }
                if "ndtw" in info:
                    result["ndtw"] = info["ndtw"]
                if "sdtw" in info:
                    result["sdtw"] = info["sdtw"]
                if "path_length" in info:
                    result["path_length"] = info["path_length"]
                agent.flush_video()
                t_dict["t_episode"] = time.time() - episode_start_time
                t_dict["success"] = float(info["success"])
                t_dict["result_row"] = result
                result_queue.put(t_dict)
    finally:
        env.close()


class SimpleVLN_Agent(Agent):
    def __init__(self, base_url, model_name, result_path, forward_distance,
                    turn_angle, max_action_history, resolution_ratio,
                    save_video=False):

        self.result_path = result_path
        self.save_video = save_video
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        os.makedirs(self.result_path, exist_ok=True)
        if self.save_video:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        urls = [u.strip() for u in base_url.split(",") if u.strip()]
        self.clients = [OpenAI(base_url=url, api_key="dummy", timeout=600) for url in urls]
        self._client_idx = 0
        self.model = model_name

        self.temperature = 0.3
        self.top_p = 0.95
        self.max_tokens = 512

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        self.topdown_map_list = []
        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def predict_inference(self):
        client = self.clients[self._client_idx % len(self.clients)]
        self._client_idx += 1

        outputs = client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        output_text = outputs.choices[0].message.content
        output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
        output_text = output_text.strip()

        return output_text

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")
        
    def flush_video(self):
        if self.save_video and self.topdown_map_list:
            output_path = os.path.join(self.result_path, "video", self.video_name)
            imageio.mimsave(output_path, self.topdown_map_list)
            self.topdown_map_list = []

    def reset(self):
        self.flush_video()
        self.topdown_map_list = []

        self.pending_action_list = []
        self.rgb_list = []

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        self.rgb_list.append(rgb_)
        # Keep the most recent observations within the configured history window.
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.save_video:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.save_video:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            return {"action": temp_action}

        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation.')
        content.append({"type": "text", "text": item[1]})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        
        if self.save_video:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        
        result = self.extract_multi_result(navigation)

        select_action_idx = 2

        result = result[:select_action_idx]
        for action_index,numeric in result:

            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric/self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list)==0:
                print('random select an action')
                action_index = random.randint(1, 3)
                navigation = self.action_id_to_str(action_index)
                self.pending_action_list.append(action_index)

        return {"action": self.pending_action_list.pop(0)}


def resolve_vllm_urls(args):
    if args.vllm_base_url is not None:
        urls = [url.strip().rstrip("/") for url in args.vllm_base_url.split(",") if url.strip()]
        if not urls:
            raise ValueError("--vllm_base_url must contain at least one URL")
    else:
        if args.vllm_num_servers < 1:
            raise ValueError("--vllm_num_servers must be positive")
        host = args.vllm_host.strip().rstrip("/")
        parsed = urlsplit(host if "://" in host else "http://" + host)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.port or parsed.username:
            raise ValueError("--vllm_host must be a hostname, optionally prefixed with http:// or https://; use --vllm_base_url for full URLs")
        if not (1 <= args.vllm_port_start <= args.vllm_port_start + args.vllm_num_servers - 1 <= 65535):
            raise ValueError("vLLM ports must be between 1 and 65535")
        hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        urls = [f"{parsed.scheme}://{hostname}:{args.vllm_port_start + i}/v1" for i in range(args.vllm_num_servers)]
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Invalid vLLM URL: {url}")
        _ = parsed.port  # validate port syntax/range
    return ",".join(urls)


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="number of evaluation workers")
    parser.add_argument("--resolution-ratio",type=float,help="scale factor for observation images, in (0, 1]",default=1.0)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    # Accept the former no-op flag for existing launch commands; repetitions use pass-k.
    parser.add_argument("--num-generations",type=int,choices=(1,),help=argparse.SUPPRESS)
    parser.add_argument("--pass-k", type=int, default=1, help="run each trajectory k times")
    parser.add_argument("--vllm_base_url", type=str, default=None,
                        help="comma-separated full server URLs; overrides host/port options")
    parser.add_argument("--vllm_host", type=str, default="localhost",
                        help="vLLM server host")
    parser.add_argument("--vllm_port_start", type=int, default=22001,
                        help="starting port of vLLM servers")
    parser.add_argument("--vllm_num_servers", type=int, default=8,
                        help="number of vLLM servers (ports: port_start ~ port_start+num-1)")
    parser.add_argument("--vllm_model_name", type=str, default="qwen3vl")
    parser.add_argument(
        "--save-video", "--save_vedio", dest="save_video",
        action="store_true",
        help="if set, save per-episode top-down+gifs under result-path/video",
    )
    args = parser.parse_args()

    try:
        vllm_base_url = resolve_vllm_urls(args)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("split_num", "pass_k", "forward_distance", "turn_angle", "max_action_history"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    print(f"Using vLLM servers: {vllm_base_url}")

    if not 0 < args.resolution_ratio <= 1:
        parser.error("--resolution-ratio must be in (0, 1]")
    with result_directory_lock(args.result_path):
        run_evaluation(args, vllm_base_url)


def run_evaluation(args, vllm_base_url):
    config = get_config(args.exp_config)
    if abs(config.habitat.simulator.forward_step_size * 100 - args.forward_distance) > 1e-6:
        raise ValueError("--forward-distance (cm) must match simulator.forward_step_size (m)")
    if abs(config.habitat.simulator.turn_angle - args.turn_angle) > 1e-6:
        raise ValueError("--turn-angle must match simulator.turn_angle")
    with habitat.config.read_write(config):
        if args.save_video:
            config.habitat.task.measurements.update(
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
            
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset.config = config.habitat.dataset
    num_episodes = len(dataset.episodes)
    if not num_episodes:
        raise ValueError("The configured dataset contains no episodes")
    split_num = min(args.split_num, num_episodes)
    dataset_splits = dataset.get_splits(split_num, allow_uneven_splits=True)
    for ds in dataset_splits:
        ds.config = config.habitat.dataset
    trajectories = {
        (ep.scene_id.split('/')[-2], str(ep.episode_id), ep.instruction.instruction_text)
        for ep in dataset.episodes
    }
    if len(trajectories) != num_episodes:
        raise ValueError("Dataset contains duplicate episode/instruction identities")
    required_metrics = tuple(name for name in ("path_length", "ndtw", "sdtw")
                             if name in config.habitat.task.measurements)
    done_rows = load_result_rows(args.result_path, pass_k=args.pass_k,
                                trajectories=trajectories, required_metrics=required_metrics)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    try:
        for i in range(split_num):
            worker_args = (
                result_queue,
                vllm_base_url,
                args.vllm_model_name,
                config,
                dataset_splits[i],
                args.result_path,
                args.forward_distance,
                args.turn_angle,
                args.max_action_history,
                args.resolution_ratio,
                args.save_video,
                args.pass_k,
                done_rows,
            )
            p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
            p.start()
            processes.append(p)

        with tqdm(total=num_episodes * args.pass_k, desc="Evaluating") as pbar:
            done_count = 0
            succ_sum = 0.0
            succ_done = 0
            target = num_episodes * args.pass_k
            while done_count < target:
                alive = [p for p in processes if p.is_alive()]
                try:
                    result = result_queue.get(timeout=10)
                    row = result.pop("result_row", None)
                    if row is not None:
                        append_result(args.result_path, row)
                    done_count += 1
                    if "success" in result:
                        succ_sum += float(result["success"])
                        succ_done += 1
                    sr = (succ_sum / succ_done) if succ_done else 0.0
                    pbar.update(1)
                    pbar.set_postfix(sr=f"{sr:.4f}", **result)
                except Empty:
                    if not alive:
                        print(f"All workers exited. Got {done_count}/{target} results.")
                        break
        
        for p in processes:
            p.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        manager.shutdown()

    failed_workers = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
    if failed_workers or done_count != target:
        raise RuntimeError(
            f"Evaluation incomplete: {done_count}/{target} results; "
            f"failed workers (pid, exitcode): {failed_workers}. "
            "Completed episode rows are preserved for resume; no summary was written."
        )

    rows = load_result_rows(args.result_path, pass_k=args.pass_k,
                            trajectories=trajectories, required_metrics=required_metrics)
    if len(rows) != target:
        raise RuntimeError(f"Expected {target} distinct complete records, found {len(rows)}; no summary was written")
    summary = summarize_results(rows.values(), args.pass_k)
    print(json.dumps(summary, ensure_ascii=False))
    append_result(args.result_path, summary)


def summarize_results(rows, pass_k):
    n_run = 0
    s_suc, s_spl, s_os, s_ne = 0.0, 0.0, 0.0, 0.0
    s_ndtw, n_ndtw = 0.0, 0
    s_sdtw, n_sdtw = 0.0, 0
    pl_vals = []
    step_vals = []
    traj_results = {}
    for item in rows:
        success = float(item["success"])
        spl = float(item["spl"])
        os_val = float(item["os"])
        ne = float(item["ne"])
        # Keep every run in core metrics, including nonfinite SPL and NE values.
        ndtw = float(item["ndtw"]) if item.get("ndtw") is not None else None
        ndtw = ndtw if ndtw is not None and np.isfinite(ndtw) else None
        sdtw = float(item["sdtw"]) if item.get("sdtw") is not None else None
        sdtw = sdtw if sdtw is not None and np.isfinite(sdtw) else None
        pl = float(item["path_length"]) if "path_length" in item else None
        if pl is not None and np.isfinite(pl):
            pl_vals.append(pl)
        if np.isfinite(float(item["steps"])):
            step_vals.append(float(item["steps"]))
        n_run += 1
        s_suc += success
        s_spl += spl
        s_os += os_val
        s_ne += ne
        if ndtw is not None and np.isfinite(ndtw):
            s_ndtw += ndtw
            n_ndtw += 1
        if sdtw is not None and np.isfinite(sdtw):
            s_sdtw += sdtw
            n_sdtw += 1
        traj_key = (item["scene_id"], str(item["episode_id"]), item["episode_instruction"])
        if traj_key not in traj_results:
            traj_results[traj_key] = {
                "pass_success": success,
                "pass_spl": spl,
                "pass_os": os_val,
                "pass_ne": ne,
                "pass_ndtw": ndtw,
                "pass_sdtw": sdtw,
            }
        else:
            traj_results[traj_key]["pass_success"] = max(traj_results[traj_key]["pass_success"], success)
            traj_results[traj_key]["pass_spl"] = max(traj_results[traj_key]["pass_spl"], spl)
            traj_results[traj_key]["pass_os"] = max(traj_results[traj_key]["pass_os"], os_val)
            traj_results[traj_key]["pass_ne"] = min(traj_results[traj_key]["pass_ne"], ne)
            if ndtw is not None:
                prev = traj_results[traj_key]["pass_ndtw"]
                traj_results[traj_key]["pass_ndtw"] = ndtw if prev is None else max(prev, ndtw)
            if sdtw is not None:
                prev = traj_results[traj_key]["pass_sdtw"]
                traj_results[traj_key]["pass_sdtw"] = sdtw if prev is None else max(prev, sdtw)

    n_traj = len(traj_results)
    if n_run and n_traj:
        pass_k_value = int(pass_k)
        pass_at_k_key = f"pass@{pass_k_value}"
        avg_k_key = f"avg{pass_k_value}"
        pass_suc = sum(v["pass_success"] for v in traj_results.values()) / n_traj
        pass_spl = sum(v["pass_spl"] for v in traj_results.values()) / n_traj
        pass_os = sum(v["pass_os"] for v in traj_results.values()) / n_traj
        pass_ne = sum(v["pass_ne"] for v in traj_results.values()) / n_traj
        ndtw_vals = [v["pass_ndtw"] for v in traj_results.values() if v["pass_ndtw"] is not None and np.isfinite(v["pass_ndtw"])]
        sdtw_vals = [v["pass_sdtw"] for v in traj_results.values() if v["pass_sdtw"] is not None and np.isfinite(v["pass_sdtw"])]
        avg_success = s_suc / n_run
        avg_spl = s_spl / n_run
        avg_os = s_os / n_run
        avg_ne = s_ne / n_run
        length_val = sum(pl_vals) / len(pl_vals) if pl_vals else None
        steps_avg = sum(step_vals) / len(step_vals) if step_vals else None
        summary = {
            "sucs_all": avg_success,
            "spls_all": avg_spl,
            "oss_all": avg_os,
            "ones_all": avg_ne,
            "length": length_val,
            "steps_avg": steps_avg,
            "pass_k": pass_k_value,
            pass_at_k_key: pass_suc,
            avg_k_key: avg_success,
            f"pass_spl@{pass_k_value}": pass_spl,
            f"pass_os@{pass_k_value}": pass_os,
            f"pass_ne@{pass_k_value}": pass_ne,
        }
        if n_ndtw:
            summary["ndtw"] = s_ndtw / n_ndtw
            summary[f"pass_ndtw@{pass_k_value}"] = sum(ndtw_vals) / len(ndtw_vals) if ndtw_vals else None
        if n_sdtw:
            summary["sdtw"] = s_sdtw / n_sdtw
            summary[f"pass_sdtw@{pass_k_value}"] = sum(sdtw_vals) / len(sdtw_vals) if sdtw_vals else None
    else:
        pass_k_value = int(pass_k)
        pass_at_k_key = f"pass@{pass_k_value}"
        avg_k_key = f"avg{pass_k_value}"
        summary = {
            "sucs_all": None,
            "spls_all": None,
            "oss_all": None,
            "ones_all": None,
            "length": None,
            "steps_avg": None,
            "pass_k": pass_k_value,
            pass_at_k_key: None,
            avg_k_key: None,
            f"pass_spl@{pass_k_value}": None,
            f"pass_os@{pass_k_value}": None,
            f"pass_ne@{pass_k_value}": None,
        }

    return summary

if __name__ == "__main__":
    main()
