"""Validated JSONL results shared by resume and final aggregation."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path


def result_key(row, required_metrics=()):
    if not isinstance(row, dict):
        return None
    try:
        scene, episode, instruction = (row[k] for k in
                                       ("scene_id", "episode_id", "episode_instruction"))
        if not isinstance(scene, str) or not isinstance(instruction, str):
            return None
        if not isinstance(episode, (str, int)) or isinstance(episode, bool):
            return None
        trial, total = row.get("trial_id", 0), row.get("trial_total", 1)
        if isinstance(trial, bool) or isinstance(total, bool):
            return None
        if int(trial) != trial or int(total) != total or not 0 <= trial < total:
            return None
        # Keep nonfinite SPL/NE for compatibility with the reference evaluator.
        for name in ("success", "spl", "os", "ne", "steps", *required_metrics):
            float(row[name])
        if not all(math.isfinite(float(row[name])) for name in ("success", "os", "steps")):
            return None
        return scene, str(episode), instruction, int(trial), int(total)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def load_result_rows(result_path, *, pass_k=None, trajectories=None, required_metrics=()):
    """Keep the last complete row for each trial, scoped to the current run."""
    path = Path(result_path) / "result.json"
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = result_key(row, required_metrics)
            if key is None or (pass_k is not None and key[4] != pass_k):
                continue
            if trajectories is not None and key[:3] not in trajectories:
                continue
            rows[key] = row
    return rows


def append_result(result_path, row):
    """Called only by the parent process while it owns the directory lock."""
    path = Path(result_path) / "result.json"
    data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("a+b") as stream:
        # Preserve a truncated final line, but don't concatenate the next record
        # onto it. The reader skips invalid rows on resume.
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(data)


@contextmanager
def result_directory_lock(result_path):
    path = Path(result_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".eval.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluator is using {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
