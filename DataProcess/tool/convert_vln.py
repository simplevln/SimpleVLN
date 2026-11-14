import argparse
import copy
import random

from tqdm import tqdm
from convert_common import (
    get_episode_id, try_combine, uniform_sample_with_ends, prep_actions_and_images,
    get_instructions, load_episodes, relativize_image_paths, write_samples,
)


SYSTEM_PROMPT = "You are a visual language navigation model."

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations{} and an image of the current observation{}. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
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


def convert_episode_vln(episode, image_root, out_samples, max_history=None):
    prepared = prep_actions_and_images(episode, image_root)
    if prepared is None:
        return
    action_texts, images = prepared
    instructions = get_instructions(episode)
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
    parser.add_argument("--max_history", type=int, default=8, help="Number of history frames to sample (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global MAX_HISTORY
    if args.max_history < 1:
        parser.error("--max_history must be positive")
    MAX_HISTORY = args.max_history

    random.seed(args.seed)

    episodes = load_episodes(args.annotation_path)

    samples = []
    for episode in tqdm(episodes, desc="vln"):
        convert_episode_vln(episode, args.image_root, samples)

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
