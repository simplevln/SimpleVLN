"""Commit annotations before completion markers; serialize concurrent JSONL writes."""
import fcntl
import json
import os
from pathlib import Path
import tempfile


def deduplicate_records(records, key_name='video', fallback_keys=('episode_id', 'trajectory_id')):
    """Keep the latest record for each trajectory without reordering other records."""
    seen, output = set(), []
    for item in reversed(records):
        key = item.get(key_name)
        if key is None:
            key = tuple(item.get(k) for k in fallback_keys)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return list(reversed(output))


def load_records(path, *, strict_json=False):
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == '.json':
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError, UnicodeError) as exc:
            if strict_json:
                raise ValueError(f'Invalid annotation JSON: {path}') from exc
            return []
        records = value if isinstance(value, list) else [value]
        if strict_json and any(not isinstance(row, dict) for row in records):
            raise ValueError(f'Annotations must contain JSON objects: {path}')
        return records
    records = []
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def write_snapshot(path, records):
    """Atomically replace a JSON snapshot; leave existing annotations intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = stream.name
            json.dump(records, stream, ensure_ascii=False, indent=4)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_annotations(output_path, annotations_prefix='annotations_actions'):
    """Read the latest record per trajectory from saved snapshots and worker journals."""
    root = Path(output_path)
    if not root.exists():
        return []
    target = root / f'{annotations_prefix}.json'
    records = load_records(target, strict_json=True)

    shards = {}
    for path in root.iterdir():
        if (path.is_file() and path.name.startswith(annotations_prefix + '_')
                and path.suffix in ('.json', '.jsonl')):
            shards.setdefault(path.stem, {})[path.suffix] = path
    ordered_shards = sorted(shards.values(), key=lambda paths: (
        max(path.stat().st_mtime_ns for path in paths.values()),
        min(path.name for path in paths.values()),
    ))
    for paths in ordered_shards:
        # A journal may contain only new records after a resume or interrupted migration.
        for suffix in ('.json', '.jsonl'):
            if suffix in paths:
                records.extend(load_records(paths[suffix], strict_json=True))
    return deduplicate_records(records)


def merge_annotations(output_path, annotations_prefix='annotations_actions'):
    """Merge saved annotations and worker journals, retaining records from earlier runs."""
    records = load_annotations(output_path, annotations_prefix)
    if not records:
        return

    def episode_order(record):
        episode_id = record.get('id', record.get('episode_id', ''))
        try:
            return (0, int(episode_id))
        except (TypeError, ValueError, OverflowError):
            return (1, str(episode_id))

    records = sorted(records, key=episode_order)
    target = Path(output_path) / f'{annotations_prefix}.json'
    write_snapshot(target, records)
    print(f'Merged {len(records)} annotations into {target}')


def append_records(path, records):
    if not records:
        return
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode('utf-8')
    with Path(path).open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0, 2)
            if stream.tell():
                stream.seek(-1, 2)
                if stream.read(1) != b'\n':
                    stream.write(b'\n')
            stream.write(data)
            stream.flush()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed_keys(output_path, prefix, require_images=True):
    root = Path(output_path)
    saved = set()
    for annotation in load_annotations(root, prefix):
        keys = (annotation.get('scene_id'), annotation.get('episode_id'), annotation.get('trajectory_id'))
        video, actions = annotation.get('video'), annotation.get('actions')
        if (any(value is None for value in keys) or not isinstance(video, str) or not video
                or not isinstance(actions, list) or not actions):
            continue
        if require_images:
            folder = root / video / 'rgb'
            if not all((folder / f'{i:03d}.jpg').is_file() for i in range(len(actions))):
                continue
        saved.add(tuple(map(str, keys)))
    # result.json uses JSONL even though its suffix is .json.
    result_file = root / 'result.json'
    latest_results = {}
    if not result_file.exists():
        return set()
    with result_file.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not all(k in record for k in ('scene', 'episode_id', 'trajectory_id', 'save')):
                continue
            key = tuple(str(record[k]) for k in ('scene', 'episode_id', 'trajectory_id'))
            latest_results[key] = record['save']
    return {key for key, flag in latest_results.items()
            if isinstance(flag, (int, bool)) and (flag == 0 or (flag == 1 and key in saved))}
