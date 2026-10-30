"""VLNFullEpisodeAgentLoop — verl AgentLoopBase subclass.

Drives a full Habitat episode inside verl's AgentLoopWorker. Uses
server_manager.generate (colocated vLLM) + apply_chat_template for each
SimpleVLN decision. Stores ALL decisions in extra_fields; returns the last
decision as the 1:1 placeholder AgentLoopOutput. VLNOnlineRolloutManager
extracts + flattens.
"""
import base64
import io
import os
from typing import Any
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopWorker, register
from verl.utils.chat_template import apply_chat_template as verl_apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    normalize_token_ids,
)
from verl.workers.rollout.replica import TokenOutput

from recipe.vln_simplevln.env_pool import VLNEnv
from recipe.vln_simplevln.full_episode_agent_loop import (
    DecisionGen,
    TrajectoryRecord,
    run_episode,
)
from recipe.vln_simplevln.prompt import build_simplevln_messages


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@register("vln_full_episode_agent")
class VLNFullEpisodeAgentLoop(AgentLoopBase):
    """One run() = one full episode. Decisions stored in extra_fields for flatten."""

    def __init__(self, *args, env_server_url: str = "http://127.0.0.1:8002",
                 progress_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_url = os.environ.get("VLN_ENV_SERVER_URL", env_server_url)
        self.progress_coef = float(os.environ.get("VLN_PROGRESS_COEF", progress_coef))
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info", {}) or {}
        episode_id = str(extra.get("episode_id", ""))
        uid = str(kwargs.get("uid", episode_id))

        metrics: dict = {}

        # Build verl-native decide closure (uses server_manager + apply_chat_template)
        async def verl_decide(instruction: str, b64_buffer: list[str], all_frames: list[str]) -> DecisionGen:
            pil_images = [_b64_to_pil(b) for b in b64_buffer]
            messages, images, image_indices = build_simplevln_messages(instruction, pil_images)
            # Convert window-relative indices to all_frames absolute indices
            offset = len(all_frames) - len(b64_buffer)
            image_indices = [idx + offset for idx in image_indices]
            mm_processor_kwargs = self._get_mm_processor_kwargs(None)
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: verl_apply_chat_template(
                    self.processor or self.tokenizer,
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if self.processor is None:
                raise RuntimeError("VLN multimodal rollout requires a model processor to construct mRoPE position ids")
            processor_inputs = await self.loop.run_in_executor(
                None,
                lambda: build_multimodal_processor_inputs(
                    self.processor,
                    text=[raw_prompt],
                    images=images,
                    mm_processor_kwargs=mm_processor_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(processor_inputs)
            if len(prompt_ids) > self.prompt_length:
                raise ValueError(
                    f"VLN prompt has {len(prompt_ids)} tokens, exceeding data.max_prompt_length="
                    f"{self.prompt_length}; increase the prompt and model length limits"
                )
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            full_input_ids = torch.tensor([prompt_ids + list(output.token_ids)], dtype=torch.long)
            full_attention_mask = torch.ones_like(full_input_ids)
            position_ids = AgentLoopWorker._compute_position_ids(
                self,
                full_input_ids,
                full_attention_mask,
                processor_inputs,
            )
            action_text = self.tokenizer.decode(output.token_ids)
            return DecisionGen(
                action_text=action_text,
                prompt_ids=prompt_ids,
                response_ids=output.token_ids,
                response_logprobs=output.log_probs,
                response_mask=[1] * len(output.token_ids),
                images=images,
                mm_processor_kwargs=mm_processor_kwargs,
                image_indices=image_indices,
                raw_prompt=raw_prompt,
                position_ids=position_ids.squeeze(0).tolist(),
            )

        # Drive full episode
        env = VLNEnv(self.env_server_url)
        try:
            traj: TrajectoryRecord = await run_episode(
                env, extra, verl_decide,
                group_uid=uid,
                trajectory_uid=f"{uid}#{uuid4().hex[:8]}",
                progress_coef=self.progress_coef,
            )
        finally:
            await env.close()

        # Pick last decision as the 1:1 placeholder output
        last = traj.decisions[-1] if traj.decisions else None
        if last and last.gen.prompt_ids is not None:
            prompt_ids = last.gen.prompt_ids
            response_ids = last.gen.response_ids or []
            response_mask = last.gen.response_mask or [1] * len(response_ids)
            response_logprobs = last.gen.response_logprobs
            multi_modal_data = {"images": last.gen.images} if last.gen.images else None
            mm_kwargs = last.gen.mm_processor_kwargs
        else:
            # fallback: empty (shouldn't happen)
            prompt_ids = [0]
            response_ids = [0]
            response_mask = [0]
            response_logprobs = None
            multi_modal_data = None
            mm_kwargs = None

        # Serialize full trajectory into extra_fields for the manager to flatten
        trajectory_data = {
            "group_uid": traj.group_uid,
            "trajectory_uid": traj.trajectory_uid,
            "episode_id": traj.episode_id,
            "scene_id": traj.scene_id,
            "instruction": traj.instruction,
            "reward": traj.reward,
            "metrics": traj.metrics,
            "num_decisions": len(traj.decisions),
            "image_buffer": traj.image_buffer,
            "decisions": [
                {
                    "turn_id": d.turn_id,
                    "prompt_ids": d.gen.prompt_ids,
                    "response_ids": d.gen.response_ids,
                    "response_logprobs": d.gen.response_logprobs,
                    "response_mask": d.gen.response_mask,
                    "image_indices": d.gen.image_indices,
                    "raw_prompt": d.gen.raw_prompt,
                    "mm_processor_kwargs": d.gen.mm_processor_kwargs,
                    "position_ids": d.gen.position_ids,
                    "action_text": d.action_text,
                    "is_stop_action": d.is_stop_action,
                }
                for d in traj.decisions
                if d.gen.prompt_ids is not None
            ],
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_kwargs,
            reward_score=float(traj.reward),
            num_turns=len(traj.decisions),
            metrics=metrics,
            extra_fields={"trajectory": trajectory_data},
        )
