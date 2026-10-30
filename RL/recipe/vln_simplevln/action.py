"""SimpleVLN text-action parsing + Habitat atomic-action expansion.

Faithful to the SimpleVLN evaluator (extract_result / extract_multi_result + the
action-chunk expansion): up to ACTIONS_PER_TURN sub-actions per decision, each
expanded to <= MAX_REPEAT atomic Habitat steps. Action ids: 0=stop, 1=forward,
2=turn left, 3=turn right.
"""
import re

FORWARD_DISTANCE = 25  # cm per forward step
TURN_ANGLE = 15        # deg per turn step
ACTIONS_PER_TURN = 2   # eval select_action_idx=2
MAX_REPEAT = 3         # eval min(3, round(numeric/unit))


def extract_result(text: str, forward_distance: int = FORWARD_DISTANCE, turn_angle: int = TURN_ANGLE):
    """One sub-action string -> (action_id, numeric). Mirrors eval.extract_result."""
    m = re.search(r"<answer>(.*?)</answer>", text)
    s = (m.group(1) if m else text).strip().lower()
    if "stop" in s:
        return 0, None
    if "forward" in s:
        n = re.search(r"-?\d+", s)
        return 1, (float(n.group()) if n else float(forward_distance))
    if "left" in s:
        n = re.search(r"-?\d+", s)
        return 2, (float(n.group()) if n else float(turn_angle))
    if "right" in s:
        n = re.search(r"-?\d+", s)
        return 3, (float(n.group()) if n else float(turn_angle))
    return None, None


def parse_simplevln_action(text: str, n: int = ACTIONS_PER_TURN,
                        forward_distance: int = FORWARD_DISTANCE, turn_angle: int = TURN_ANGLE):
    """Model output -> first n (action_id, numeric) sub-actions (split by ', ')."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    subs = text.split(", ")
    return [extract_result(sa, forward_distance, turn_angle) for sa in subs[:n]]


def to_atomic_chunk(parsed: list, forward_distance: int = FORWARD_DISTANCE,
                    turn_angle: int = TURN_ANGLE, max_repeat: int = MAX_REPEAT) -> list[int]:
    """Parsed sub-actions -> flat atomic Habitat action list (one id per env.step).

    None sub-actions are skipped; the caller decides fallback (eval picks a random
    move when nothing valid was produced).
    """
    chunk: list[int] = []
    for action_id, numeric in parsed:
        if action_id == 0:
            chunk.append(0)
        elif action_id == 1:
            chunk += [1] * min(max_repeat, round(numeric / forward_distance))
        elif action_id == 2:
            chunk += [2] * min(max_repeat, round(numeric / turn_angle))
        elif action_id == 3:
            chunk += [3] * min(max_repeat, round(numeric / turn_angle))
    return chunk
