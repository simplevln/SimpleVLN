import argparse
import os
import random

from tqdm import tqdm
from convert_common import (
    action_id_to_str, try_combine, get_instructions, get_episode_id,
    prep_actions_and_images, load_episodes, skip_episode, relativize_image_paths,
    write_samples,
)
from convert_vln import (
    SYSTEM_PROMPT, MAX_ACTIONS_PER_SAMPLE, build_vln_prompt, normalize_pair,
    sample_history_plus_current, convert_episode_vln,
)

MAX_HISTORY = 8


def convert_episode_atomic(episode, image_root, out_samples):
    """Use the same random action merging as the standard VLN converter."""
    convert_episode_vln(episode, image_root, out_samples, max_history=MAX_HISTORY)


def _action_chunks(actions):
    """Bound each label to three compound actions while preserving frame offsets."""
    start, text = 0, actions[0]
    for index in range(1, len(actions)):
        combined = try_combine(text, actions[index])
        if combined is not None:
            text = combined
        elif text.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
            text += ", " + actions[index]
        else:
            yield start, index, text
            start, text = index, actions[index]
    yield start, len(actions), text


def convert_episode_merged(episode, image_root, out_samples):
    groups = episode.get("model_call_groups") or []
    if not groups:
        return
    prepared = prep_actions_and_images(episode, image_root)
    if prepared is None:
        return
    action_texts, all_images = prepared
    instructions = get_instructions(episode)
    if not instructions:
        return
    episode_id = get_episode_id(episode)
    has_stop = False

    def emit(step, text):
        images = normalize_pair(sample_history_plus_current(all_images[:step + 1], max_history=MAX_HISTORY))
        for instruction in instructions:
            out_samples.append({
                "system": SYSTEM_PROMPT,
                "messages": [{"from": "user", "value": build_vln_prompt(instruction, len(images))},
                             {"from": "assistant", "value": text}],
                "episode_id": episode_id, "task type": "vln", "images": list(images),
            })

    try:
        if not isinstance(groups, (list, tuple)):
            raise ValueError("model_call_groups must be a list of objects")
        starts = []
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("model_call_groups must be a list of objects")
            value = group["call_step_id"]
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError("model call indices must be integers")
            starts.append(int(value))
        if any(start < 0 or start >= len(action_texts) for start in starts):
            raise ValueError("model call index is outside the executed trajectory")
        if any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError("model call indices must be strictly increasing")
        # Prepare all groups before emitting to avoid partially accepting a corrupt episode.
        segments = []
        for index, group in enumerate(groups):
            start = starts[index]
            recorded = group.get("executed_actions")
            group_actions = recorded if recorded is not None else group.get("actions", [])
            if not isinstance(group_actions, (list, tuple)):
                raise ValueError("model group actions must be a list")
            count = len(group_actions)
            end = min(start + count, len(action_texts),
                      starts[index + 1] if index + 1 < len(starts) else len(action_texts))
            if recorded is not None:
                expected = [action_id_to_str(a) for a in recorded]
                if end - start != len(expected) or action_texts[start:end] != expected:
                    raise ValueError("model group disagrees with episode executed_actions")
            if end > start:
                segments.append((start, action_texts[start:end]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        skip_episode(episode, str(exc))
        return

    for start, actions in segments:
        for offset, _, text in _action_chunks(actions):
            emit(start + offset, text)
            has_stop = has_stop or "stop" in text
    # STOP belongs to the terminal observation, never the end of the last model call.
    if not has_stop and action_texts[-1] == "stop":
        emit(len(action_texts) - 1, "stop")


def main():
    parser = argparse.ArgumentParser(description="Convert dagger trajectory data to training format")
    parser.add_argument(
        "--annotation_path",
        required=True,
        help="Path to dagger annotations_actions.json or .jsonl",
    )
    parser.add_argument(
        "--image_root",
        required=True,
        help="Root directory containing episode images (dagger output path)",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output JSON path for training data",
    )
    parser.add_argument(
        "--mode",
        choices=["atomic", "merged", "both"],
        default="both",
        help="atomic: random merge of per-step actions; merged: use model_call_groups; both: output both",
    )
    parser.add_argument("--max_history", type=int, default=8)
    parser.add_argument("--relative_to", help="Write image paths relative to this dataset root.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global MAX_HISTORY
    if args.max_history < 1:
        parser.error("--max_history must be positive")
    MAX_HISTORY = args.max_history
    random.seed(args.seed)

    episodes = load_episodes(args.annotation_path)

    print(f"Loaded {len(episodes)} episodes from {args.annotation_path}")

    output_base, _ = os.path.splitext(args.output_path)
    outputs = []

    if args.mode in ("atomic", "both"):
        atomic_samples = []
        for episode in tqdm(episodes, desc="atomic"):
            convert_episode_atomic(episode, args.image_root, atomic_samples)
        out = output_base + "_atomic.json" if args.mode == "both" else args.output_path
        if not atomic_samples:
            parser.error("No valid atomic samples were produced")
        outputs.append(("atomic", out, atomic_samples))

    if args.mode in ("merged", "both"):
        merged_samples = []
        for episode in tqdm(episodes, desc="merged"):
            convert_episode_merged(episode, args.image_root, merged_samples)
        out = output_base + "_merged.json" if args.mode == "both" else args.output_path
        if not merged_samples:
            parser.error("No valid merged samples were produced; model_call_groups are required")
        outputs.append(("merged", out, merged_samples))

    # Validate both conversions and their paths before replacing either output.
    try:
        for _, _, samples in outputs:
            relativize_image_paths(samples, args.relative_to)
    except ValueError as exc:
        parser.error(str(exc))
    for mode, out, samples in outputs:
        write_samples(out, samples)
        print(f"[{mode}] wrote {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
