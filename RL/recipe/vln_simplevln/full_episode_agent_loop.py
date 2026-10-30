"""VLN full-episode rollout.

run_episode() drives one Habitat trajectory to done/cap and records every
SimpleVLN decision as a DecisionRecord. The only backend-specific piece is decide():

  Standalone inference (rollout_smoke):
    b64_buffer -> build_simplevln_messages_b64 -> vLLM OpenAI API -> action_text
    decide returns DecisionGen(action_text=...) only (no token ids).

  verl training (VLNFullEpisodeAgentLoop):
    b64_buffer -> apply_chat_template -> server_manager.generate
    decide returns DecisionGen with prompt_ids / response_ids / response_mask / images
    for training. The same episode loop is reused unchanged.
"""
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from recipe.vln_simplevln.action import parse_simplevln_action, to_atomic_chunk
from recipe.vln_simplevln.reward import compute_trajectory_reward

MAX_ACTION_HISTORY = 200  # frame buffer cap (matches eval --max-action-history 200)
_FALLBACK_CHUNK = [1]     # Use one forward step when no valid action was produced.


@dataclass
class DecisionGen:
    """What a backend returns for one LLM call. Training fields are optional for standalone inference."""
    action_text: str
    prompt_ids: Optional[list[int]] = None
    response_ids: Optional[list[int]] = None
    response_logprobs: Optional[list[float]] = None
    response_mask: Optional[list[int]] = None
    images: Optional[list[Any]] = None           # PIL images (rollout inference only)
    mm_processor_kwargs: Optional[dict] = None
    image_indices: Optional[list[int]] = None    # indices into episode b64_buffer
    raw_prompt: Optional[str] = None             # pre-tokenization prompt string
    position_ids: Optional[list[list[int]]] = None  # full prompt+response, axes x sequence


@dataclass
class DecisionRecord:
    turn_id: int
    action_text: str
    parsed_actions: list
    atomic_chunk: list[int]
    env_step_before: int
    env_step_after: int
    is_stop_action: bool
    gen: DecisionGen


@dataclass
class TrajectoryRecord:
    group_uid: str
    trajectory_uid: str
    scene_id: str
    episode_id: str
    instruction: str
    reward: float
    metrics: dict
    decisions: list[DecisionRecord] = field(default_factory=list)
    image_buffer: list[str] = field(default_factory=list)


async def run_episode(
    env,
    extra_info: dict,
    decide: Callable[..., Awaitable[DecisionGen]],
    *,
    group_uid: str,
    trajectory_uid: str,
    max_decisions: int = 64,
    max_env_steps: int = 200,
    progress_coef: float = 0.0,
) -> TrajectoryRecord:
    """Drive one full episode. env must NOT be reset yet; this function resets it.

    decide(instruction, b64_buffer, all_frames) -> DecisionGen
      b64_buffer: list of JPEG base64 strings, [-1] = current frame.
      b64_buffer is accumulated across turns (capped at MAX_ACTION_HISTORY).
    """
    b64 = await env.reset(extra_info)
    all_frames: list[str] = [b64]
    history_window: list[str] = [b64]
    decisions: list[DecisionRecord] = []
    env_steps = 0
    done = env.done

    while not done and len(decisions) < max_decisions and env_steps < max_env_steps:
        gen = await decide(env.instruction, history_window, all_frames)
        parsed = parse_simplevln_action(gen.action_text)
        chunk = to_atomic_chunk(parsed)
        step_before = env_steps

        if not chunk:
            chunk = _FALLBACK_CHUNK
        chunk = chunk[:max_env_steps - env_steps]

        for atomic in chunk:
            resp = await env.step([atomic])
            frame = env.current_jpeg_b64()
            all_frames.append(frame)
            history_window.append(frame)
            if len(history_window) > MAX_ACTION_HISTORY:
                history_window = history_window[1:]
            env_steps += 1
            done = resp["done"]
            if done:
                break

        decisions.append(DecisionRecord(
            turn_id=len(decisions), action_text=gen.action_text,
            parsed_actions=parsed, atomic_chunk=chunk,
            env_step_before=step_before, env_step_after=env_steps,
            is_stop_action=(0 in chunk), gen=gen,
        ))

    metrics = env.metrics()
    metrics["num_decisions"] = len(decisions)
    metrics["env_steps"] = env_steps
    return TrajectoryRecord(
        group_uid=group_uid, trajectory_uid=trajectory_uid,
        scene_id=str(extra_info.get("scene_id", "")),
        episode_id=str(extra_info["episode_id"]),
        instruction=env.instruction,
        reward=compute_trajectory_reward(metrics, progress_coef=progress_coef),
        metrics=metrics,
        decisions=decisions,
        image_buffer=all_frames,
    )
