"""Shared annotation, frame and action validation for the converters."""
import json
import os
from pathlib import Path
import re
import tempfile
import warnings

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def first_value(episode, keys):
    for key in keys:
        value = episode.get(key)
        if value is not None and value != '':
            return value
    return ''


def get_video_id(episode):
    return str(first_value(episode, ('video', 'video_id', 'episode_id', 'id'))).replace('\\', '/').rstrip('/')


def get_episode_id(episode):
    return str(first_value(episode, ('id', 'episode_id', 'video_id', 'video')))


def skip_episode(episode, reason):
    warnings.warn(f"Skipping episode {get_episode_id(episode)!r}: {reason}", stacklevel=2)


def load_episodes(path):
    text = Path(path).read_text(encoding='utf-8-sig').lstrip()
    if not text:
        return []
    if text.startswith('['):
        episodes = json.loads(text)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            episodes = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            episodes = [value]
    if not isinstance(episodes, list) or any(not isinstance(ep, dict) for ep in episodes):
        raise ValueError('Annotations must be a JSON array of objects or JSONL objects')
    return episodes


def relativize_image_paths(samples, relative_to):
    """Express image paths relative to a release root, rejecting paths outside it."""
    if relative_to is None:
        return
    root = Path(relative_to).resolve()
    for sample in samples:
        try:
            sample['images'] = [Path(image).resolve().relative_to(root).as_posix()
                                for image in sample['images']]
        except ValueError as exc:
            raise ValueError('An image is outside --relative_to; choose a common dataset root') from exc


def write_samples(path, samples):
    """Replace an output only after the complete JSON has been serialized."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = stream.name
            json.dump(samples, stream, ensure_ascii=False, indent=4)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def list_episode_images(image_root, video_id):
    # Both dataset-root/images/<video>/rgb and image-root/<video>/rgb are supported.
    video = str(video_id).replace('\\', '/').rstrip('/')
    if not video:
        return []
    candidates = [os.path.join(image_root, video)]
    if video.startswith('images/'):
        candidates.append(os.path.join(image_root, video[len('images/'):]))
    elif not os.path.isabs(video):
        candidates.append(os.path.join(image_root, 'images', video))
    for folder in dict.fromkeys(candidates):
        rgb = os.path.join(folder, 'rgb')
        folder = rgb if os.path.isdir(rgb) else folder
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f))]
        if not files:
            continue
        numbers = {f: re.findall(r'\d+', Path(f).stem) for f in files}
        if all(numbers.values()):
            files.sort(key=lambda f: (int(numbers[f][-1]), f))
            indices = [int(numbers[f][-1]) for f in files]
            if any(b != a + 1 for a, b in zip(indices, indices[1:])):
                raise ValueError(f'Frame numbers contain gaps or duplicates for {video!r}')
        elif any(numbers.values()):
            raise ValueError(f'Mixed numbered and unnumbered frames for {video!r}')
        else:
            files.sort()
        return [os.path.join(folder, f) for f in files]
    return []


def action_id_to_str(action_id):
    if isinstance(action_id, bool) or int(action_id) != float(action_id):
        raise ValueError(f'Invalid action id: {action_id}')
    texts = {-1: 'stop', 0: 'stop', 1: 'forward 25 cm',
             2: 'turn left 15 degree', 3: 'turn right 15 degree'}
    try:
        return texts[int(action_id)]
    except KeyError as exc:
        raise ValueError(f'Invalid action id: {action_id}') from exc


def to_action_text(action):
    if isinstance(action, str):
        text = action.strip().lower()
        if re.fullmatch(r'-?\d+', text):
            return action_id_to_str(text)
        if text == 'stop' or re.fullmatch(r'(forward [1-9]\d* cm|turn (left|right) [1-9]\d* degree)', text):
            return text
        raise ValueError(f'Invalid action text: {action!r}')
    return action_id_to_str(action)


def try_combine(running_value, next_action):
    head, separator, tail = running_value.rpartition(', ')
    tail = tail if separator else running_value
    pattern = r'(forward|turn left|turn right) (\d+) (cm|degree)'
    first, second = re.fullmatch(pattern, tail), re.fullmatch(pattern, next_action)
    if not first or not second or first[1] != second[1] or first[3] != second[3]:
        return None
    total = int(first[2]) + int(second[2])
    cap = 75 if first[1] == 'forward' else 45
    if total > cap:
        return None
    return (head + separator if separator else '') + f'{first[1]} {total} {first[3]}'


def uniform_sample_with_ends(data, n):
    if n < 1:
        raise ValueError('History sample count must be positive')
    if len(data) <= n:
        return data
    if n == 1:
        return [data[-1]]
    return [data[round(i * (len(data) - 1) / (n - 1))] for i in range(n)]


def prep_actions_and_images(episode, image_root, drop_stop=False):
    try:
        executed = episode.get('executed_actions')
        raw = executed if executed is not None else episode.get('actions', [])
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError('actions must be a nonempty list')
        actions = [to_action_text(a) for a in raw]
        if executed is None:
            # Legacy annotations have one initial sentinel, not an executed STOP.
            if str(raw[0]).strip() in ('-1', '0') or raw[0] in (-1, 0):
                actions = actions[1:]
            if 'terminal_action' in episode:
                actions.append(to_action_text(episode['terminal_action']))
        if not actions:
            actions = ['stop']
        # Repeated terminal STOPs describe the same state. Movement after STOP is invalid.
        if 'stop' in actions:
            stop = actions.index('stop')
            if any(a != 'stop' for a in actions[stop:]):
                raise ValueError('movement follows STOP')
            actions = actions[:stop + 1]
        elif executed is None and 'terminal_action' not in episode:
            actions.append('stop')  # legacy format omits the final STOP
        images = list_episode_images(image_root, get_video_id(episode))
        if not images:
            raise ValueError('no images found; check image_root and video path')
        has_stop = actions[-1] == 'stop'
        required = len(actions) if has_stop else len(actions) + 1
        if len(images) < required:
            raise ValueError(f'{len(images)} frames for {len(actions)} actions; need at least {required}')
        images = images[:len(actions) + 1]
        if has_stop and len(images) == len(actions):
            images.append(images[-1])  # only STOP can reuse the terminal frame
        if drop_stop and has_stop:
            actions = actions[:-1]
            images = images[:len(actions) + 1]
        if not actions:
            return None
        return actions, images
    except (TypeError, ValueError, OverflowError) as exc:
        skip_episode(episode, str(exc))
        return None


def get_instructions(episode, only_first=False):
    instructions = episode.get('instructions') or episode.get('instruction') or []
    if isinstance(instructions, dict):
        instructions = instructions.get('instruction_text', [])
    if isinstance(instructions, str):
        instructions = [instructions]
    if not isinstance(instructions, (list, tuple)):
        skip_episode(episode, 'instruction must be text, a list of texts, or an instruction_text object')
        return []
    instructions = [s for s in instructions if isinstance(s, str) and s.strip()]
    if not instructions:
        skip_episode(episode, 'no nonempty instruction text')
    return instructions[:1] if only_first else instructions
