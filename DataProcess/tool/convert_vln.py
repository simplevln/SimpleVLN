import argparse
import copy
import os
import random

from tqdm import tqdm
from convert_common import (
    get_video_id, get_episode_id, try_combine,
    uniform_sample_with_ends, list_episode_images, prep_actions_and_images,
    get_instructions, load_episodes, relativize_image_paths, write_samples,
)


SYSTEM_PROMPT = "You are a visual language navigation model."

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations{} and an image of the current observation{}. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

IDM_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given an image of current view{} and an image of the goal view{}. "
    "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

FFS_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You are given a history of observations{}, followed by the current observation{} and the executed action: [{}]. "
    "Then you are given four candidate next observations in order. "
    "Please select the correct option letter from A/B/C/D."
)

MAX_HISTORY = 8
MAX_ACTIONS_PER_SAMPLE = 3
COMBINE_PROB = 0.7


def normalize_pair(images):
    return [images[0], images[0]] if len(images) == 1 else images


def sample_history_plus_current(images, max_history=None):
    max_history = MAX_HISTORY if max_history is None else max_history
    if len(images) <= 1:
        return images
    history = images[:-1]
    if len(history) > max_history:
        history = uniform_sample_with_ends(history, max_history)
    return history + [images[-1]]


def build_vln_prompt(instruction, image_count):
    history_count = max(0, image_count - 1)
    return VLN_PROMPT_TEMPLATE.format("<image>" * history_count, "<image>", instruction)


def finalize_vln_sample(sample, instruction, max_history=None):
    user = sample["messages"][0]
    images = sample_history_plus_current(user["image"], max_history=max_history)
    images = normalize_pair(images)
    user["image"] = images
    user["value"] = build_vln_prompt(instruction, len(images))
    sample["images"] = list(images)
    del user["image"]
    return sample


def finalize_idm_sample(sample):
    user = sample["messages"][0]
    images = list(user["image"])
    user["value"] = IDM_PROMPT_TEMPLATE.format("<image>", "<image>")
    sample["images"] = images
    del user["image"]
    return sample


def convert_episode_vln(episode, image_root, out_samples, max_history=None):
    prepared = prep_actions_and_images(episode, image_root, drop_stop=False)
    if prepared is None:
        return
    action_texts, images = prepared
    instructions = get_instructions(episode, only_first=False)
    if not instructions:
        return
    episode_id = get_episode_id(episode)

    for instruction in instructions:
        sample = {
            "system": SYSTEM_PROMPT,
            "messages": [
                {"from": "user", "value": "", "image": [images[0]]},
                {"from": "assistant", "value": action_texts[0]},
            ],
            "episode_id": episode_id,
            "task type": "vln",
        }
        last_action = action_texts[0]
        pending = []
        for i in range(1, len(action_texts)):
            cur = action_texts[i]
            running = sample["messages"][1]["value"]
            combined = try_combine(running, cur) if cur == last_action else None
            if combined is not None and random.random() <= COMBINE_PROB:
                sample["messages"][1]["value"] = combined
                pending.append(images[i])
            elif running.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
                sample["messages"][1]["value"] = running + ", " + cur
                pending.append(images[i])
            else:
                out_samples.append(finalize_vln_sample(copy.deepcopy(sample), instruction, max_history))
                sample["messages"][0]["image"].extend(pending)
                sample["messages"][0]["image"].append(images[i])
                sample["messages"][1]["value"] = cur
                pending = []
            last_action = cur
        out_samples.append(finalize_vln_sample(copy.deepcopy(sample), instruction, max_history))


def convert_episode_idm(episode, image_root, out_samples):
    prepared = prep_actions_and_images(episode, image_root, drop_stop=True)
    if prepared is None:
        return
    action_texts, images = prepared
    if len(images) < 2:
        return
    instructions = get_instructions(episode, only_first=True)
    if not instructions:
        return
    episode_id = get_episode_id(episode)

    sample = {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"from": "user", "value": "", "image": [images[0], images[1]]},
            {"from": "assistant", "value": action_texts[0]},
        ],
        "episode_id": episode_id,
        "task type": "idm",
    }
    last_action = action_texts[0]
    for i in range(1, len(action_texts)):
        cur = action_texts[i]
        running = sample["messages"][1]["value"]
        next_goal = images[i + 1] if i + 1 < len(images) else images[-1]
        combined = try_combine(running, cur) if cur == last_action else None
        if combined is not None and random.random() <= COMBINE_PROB:
            sample["messages"][1]["value"] = combined
            sample["messages"][0]["image"][-1] = next_goal
        elif running.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
            sample["messages"][1]["value"] = running + ", " + cur
            sample["messages"][0]["image"][-1] = next_goal
        else:
            out_samples.append(finalize_idm_sample(copy.deepcopy(sample)))
            sample["messages"][0]["image"] = [images[i], next_goal]
            sample["messages"][1]["value"] = cur
        last_action = cur
    if sample["messages"][1]["value"] != "stop":
        out_samples.append(finalize_idm_sample(copy.deepcopy(sample)))


def build_scene_frame_pool(episodes, image_root):
    pool = {}
    for episode in tqdm(episodes, desc="ffs scene pool"):
        video_id = get_video_id(episode)
        if not video_id:
            continue
        scene_prefix = str(episode.get("scene_id") or os.path.basename(video_id).split("_")[0])
        prepared = prep_actions_and_images(episode, image_root)
        frames = list_episode_images(image_root, video_id) if prepared is not None else []
        if not frames:
            continue
        pool.setdefault(scene_prefix, {})[video_id] = frames
    return pool


def compose_ffs_action(action_texts, start_idx):
    merged = action_texts[start_idx]
    last_action = action_texts[start_idx]
    end_idx = start_idx
    for i in range(start_idx + 1, len(action_texts)):
        cur = action_texts[i]
        combined = try_combine(merged, last_action) if cur == last_action else None
        if combined is not None and random.random() <= COMBINE_PROB:
            merged = combined
            end_idx = i
        elif merged.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
            merged += ", " + cur
            end_idx = i
        else:
            break
        last_action = cur
    return merged, end_idx


def make_ffs_sample(episode, episode_image_list, action_texts, start_idx, end_idx, action_text, scene_pool):
    if end_idx + 1 >= len(episode_image_list):
        return None
    current_img = episode_image_list[start_idx]
    true_next_img = episode_image_list[end_idx + 1]

    history_imgs = episode_image_list[:start_idx]
    if not history_imgs:
        history_imgs = [current_img]
    if len(history_imgs) > MAX_HISTORY:
        history_imgs = uniform_sample_with_ends(history_imgs, MAX_HISTORY)

    before_window = episode_image_list[max(0, end_idx - 4) : end_idx + 1]
    neg1_candidates = [x for x in before_window if x != true_next_img]
    if not neg1_candidates:
        return None
    neg1 = random.choice(neg1_candidates)
    exclude = {true_next_img, neg1}

    if action_texts[end_idx] == "stop":
        window = episode_image_list[max(0, end_idx - 4) : end_idx + 1]
    else:
        window = episode_image_list[end_idx + 2 : min(len(episode_image_list), end_idx + 6)]
        if not window:
            window = episode_image_list[max(0, end_idx - 4) : end_idx + 1]
    neg2_candidates = [x for x in window if x not in exclude]
    if not neg2_candidates:
        return None
    neg2 = random.choice(neg2_candidates)
    exclude.add(neg2)

    video_id = get_video_id(episode)
    scene_prefix = str(episode.get("scene_id") or os.path.basename(video_id).split("_")[0])
    cross_candidates = []
    siblings = scene_pool.get(scene_prefix, {})
    for other_traj, frames in siblings.items():
        if other_traj == video_id:
            continue
        cross_candidates.extend(f for f in frames if f not in exclude)
    if not cross_candidates:
        for prefix, trajs in scene_pool.items():
            if prefix == scene_prefix:
                continue
            for other_traj, frames in trajs.items():
                cross_candidates.extend(f for f in frames if f not in exclude)
            if cross_candidates:
                break
    if not cross_candidates:
        return None
    neg3 = random.choice(cross_candidates)

    options = [("neg", neg1), ("neg", neg2), ("neg", neg3), ("true", true_next_img)]
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    correct = None
    option_imgs = []
    for idx, (kind, path) in enumerate(options):
        option_imgs.append(path)
        if kind == "true":
            correct = labels[idx]

    history_tokens = "<image>" * len(history_imgs)
    prompt = FFS_PROMPT_TEMPLATE.format(history_tokens, "<image>", action_text)
    prompt += " Option A: <image>, Option B: <image>, Option C: <image>, Option D: <image>."

    images = history_imgs + [current_img] + option_imgs
    return {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": correct},
        ],
        "episode_id": get_episode_id(episode),
        "task type": "ffs",
        "images": images,
    }


def convert_episode_ffs(episode, image_root, scene_pool, out_samples):
    prepared = prep_actions_and_images(episode, image_root, drop_stop=False)
    if prepared is None:
        return
    action_texts, images = prepared
    i = 0
    while i < len(action_texts):
        merged, end_idx = compose_ffs_action(action_texts, i)
        sample = make_ffs_sample(episode, images, action_texts, i, end_idx, merged, scene_pool)
        if sample is not None:
            out_samples.append(sample)
        i = end_idx + 1


def main():
    parser = argparse.ArgumentParser(
        description="Convert R2R / RxR / ScaleVLN trajectory annotations into VLN SFT samples.",
    )
    parser.add_argument("--annotation_path", required=True,
                        help="JSONL (R2R/RxR) or JSON-array (ScaleVLN) of episode annotations.")
    parser.add_argument("--image_root", required=True,
                        help="Directory containing per-episode rendered frames.")
    parser.add_argument("--output_path", required=True,
                        help="Output JSON with the converted samples.")
    parser.add_argument("--relative_to", help="Write image paths relative to this dataset root.")
    parser.add_argument(
        "--task_type",
        nargs="+",
        choices=["vln", "idm", "ffs"],
        default=["vln"],
    )
    parser.add_argument("--max_history", type=int, default=8, help="Number of history frames to sample (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global MAX_HISTORY
    if args.max_history < 1:
        parser.error("--max_history must be positive")
    MAX_HISTORY = args.max_history

    random.seed(args.seed)

    episodes = load_episodes(args.annotation_path)

    scene_pool = None
    if "ffs" in args.task_type:
        scene_pool = build_scene_frame_pool(episodes, args.image_root)

    samples = []
    for task_type in args.task_type:
        if task_type == "vln":
            for episode in tqdm(episodes, desc="vln"):
                convert_episode_vln(episode, args.image_root, samples)
        elif task_type == "idm":
            for episode in tqdm(episodes, desc="idm"):
                convert_episode_idm(episode, args.image_root, samples)
        elif task_type == "ffs":
            for episode in tqdm(episodes, desc="ffs"):
                convert_episode_ffs(episode, args.image_root, scene_pool, samples)

    if not samples:
        parser.error("No valid samples were produced; check annotations and frame paths")
    try:
        relativize_image_paths(samples, args.relative_to)
    except ValueError as exc:
        parser.error(str(exc))
    write_samples(args.output_path, samples)
    print(f"wrote {len(samples)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
