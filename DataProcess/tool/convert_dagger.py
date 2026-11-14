import argparse
import random

from tqdm import tqdm
from convert_common import load_episodes, relativize_image_paths, write_samples
from convert_vln import convert_episode_vln

MAX_HISTORY = 8


def convert_episode_atomic(episode, image_root, out_samples):
    """Use the same random action merging as the standard VLN converter."""
    convert_episode_vln(episode, image_root, out_samples, max_history=MAX_HISTORY)


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
        choices=["atomic"],
        default="atomic",
        help="Atomic conversion using executed per-step actions (the only supported mode)",
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

    samples = []
    for episode in tqdm(episodes, desc="atomic"):
        convert_episode_atomic(episode, args.image_root, samples)
    if not samples:
        parser.error("No valid atomic samples were produced")
    try:
        relativize_image_paths(samples, args.relative_to)
    except ValueError as exc:
        parser.error(str(exc))
    write_samples(args.output_path, samples)
    print(f"[atomic] wrote {len(samples)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
