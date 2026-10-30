#!/usr/bin/env python3
"""Copy and verify the bundled evaluation data and user-supplied MP3D scenes."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile


BUNDLED_ROOT = Path(__file__).resolve().parents[1] / "data" / "vln_data"
SCENE_PREFIX = PurePosixPath("scene_datasets/mp3d")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, relative):
    """Resolve a package entry without allowing traversal outside its root."""
    relative_path = PurePosixPath(relative)
    if (relative_path.is_absolute() or ".." in relative_path.parts
            or "\\" in relative or ":" in relative
            or str(relative_path) != relative or relative in ("", ".")):
        raise ValueError(f"Invalid package path: {relative!r}")
    path = root.joinpath(*relative_path.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes its root: {relative!r}")
    return path


def checksums(root):
    entries = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or relative in entries:
            raise ValueError("Invalid or duplicate SHA256SUMS entry")
        local_path(root, relative)
        entries[relative] = expected
    if "manifest.json" not in entries or "required_scenes.txt" not in entries:
        raise ValueError("SHA256SUMS is missing package metadata")
    return entries


def verify_file(path, expected):
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def copy_verified(source, destination, expected):
    """Copy bytes atomically; never replace an existing different file."""
    if destination.is_symlink():
        raise ValueError(f"Destination must be a regular file: {destination}")
    if destination.exists():
        verify_file(destination, expected)
        return
    if not source.is_file():
        raise ValueError(f"Missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".copy-", dir=destination.parent)
    temporary = Path(temporary)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {source}")
        temporary.chmod(0o644)
        # Link the temporary copy atomically, without overwriting an existing file.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink():
                raise ValueError(f"Destination must be a regular file: {destination}")
            verify_file(destination, expected)
    finally:
        temporary.unlink(missing_ok=True)


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify_annotations(root):
    entries = checksums(root)
    for relative, expected in entries.items():
        verify_file(local_path(root, relative), expected)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format_version"] != 1 or manifest["split"] != "val_unseen":
        raise ValueError("Unsupported evaluation data manifest")
    for item in manifest["annotation_files"]:
        if entries.get(item["path"]) != item["sha256"]:
            raise ValueError("Annotation manifest and checksums disagree")
    scene_names = set(manifest["scenes"])
    if scene_names != set((root / "required_scenes.txt").read_text().splitlines()):
        raise ValueError("Scene list and manifest disagree")
    total = 0
    for name, dataset in manifest["datasets"].items():
        episodes = load_json_gz(local_path(root, dataset["annotation"]))["episodes"]
        ids = {str(episode["episode_id"]) for episode in episodes}
        if len(episodes) != dataset["episodes"] or len(ids) != len(episodes):
            raise ValueError(f"Wrong episode count or duplicate episode IDs: {name}")
        scenes = set()
        for episode in episodes:
            scene = PurePosixPath(episode["scene_id"])
            if (len(scene.parts) != 3 or scene.parts[0] != "mp3d"
                    or scene.parts[1] not in scene_names
                    or scene.name != scene.parts[1] + ".glb"):
                raise ValueError(f"Invalid relative scene path: {episode['scene_id']}")
            scenes.add(scene.parts[1])
        if len(scenes) != dataset["scene_count"]:
            raise ValueError(f"Scene count mismatch: {name}")
        if "ground_truth" in dataset:
            ground_truth = load_json_gz(local_path(root, dataset["ground_truth"]))
            if set(ground_truth) != ids:
                raise ValueError(f"Episode and reference-trajectory IDs differ: {name}")
            if any(not row.get("locations") for row in ground_truth.values()):
                raise ValueError(f"Empty reference trajectory: {name}")
        total += len(episodes)
    if total != manifest["total_episodes"]:
        raise ValueError("Total episode count mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("VLN_DATA_ROOT", str(BUNDLED_ROOT))),
                        help="Destination data root (default: VLN_DATA_ROOT or bundled data)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mp3d-root", type=Path,
                      help="Existing Habitat MP3D directory containing <scene>/<scene>.glb")
    mode.add_argument("--check-only", action="store_true", help="Verify without copying files")
    parser.add_argument("--annotations-only", action="store_true",
                        help="Verify/copy the small annotation package without scene assets")
    args = parser.parse_args()
    if args.annotations_only and args.mp3d_root:
        parser.error("--annotations-only cannot be combined with --mp3d-root")
    root = args.data_root.resolve()
    try:
        if not args.check_only:
            manifest = verify_annotations(BUNDLED_ROOT)
            if root != BUNDLED_ROOT.resolve():
                entries = checksums(BUNDLED_ROOT)
                entries["SHA256SUMS"] = sha256(BUNDLED_ROOT / "SHA256SUMS")
                for relative, expected in entries.items():
                    copy_verified(local_path(BUNDLED_ROOT, relative),
                                  local_path(root, relative), expected)
            if args.mp3d_root:
                for item in manifest["scene_files"]:
                    relative = str(PurePosixPath(item["path"]).relative_to(SCENE_PREFIX))
                    source = local_path(args.mp3d_root, relative)
                    copy_verified(source, local_path(root, item["path"]), item["sha256"])
                    print(f"Ready: {relative}", flush=True)
        manifest = verify_annotations(root)
        print(f"Annotations OK: {manifest['total_episodes']} episodes; "
              f"{len(manifest['scenes'])} scenes.")
        if args.annotations_only:
            print("Scene assets were not checked. They are required for Habitat evaluation.")
        else:
            for item in manifest["scene_files"]:
                path = local_path(root, item["path"])
                if not path.is_file():
                    raise ValueError(
                        f"Missing scene asset: {path}\n"
                        "Provide --mp3d-root /path/to/mp3d to copy your Habitat scene assets.")
                verify_file(path, item["sha256"])
            print(f"Scene assets OK: {len(manifest['scene_files'])} files. "
                  "Data paths and checksums are ready for evaluation.")
        print(f"Data root: {root}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
