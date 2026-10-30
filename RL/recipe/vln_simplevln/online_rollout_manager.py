"""VLNOnlineRolloutManager — replaces AgentLoopManager for full-episode flatten.

Standard flow:  1 row → 1 AgentLoopOutput → 1 training sample.
VLN flow:       1 row → 1 full episode → N decisions → N training samples (flatten).

This manager inherits AgentLoopManager and overrides generate_sequences:
1. Splits rollouts into windows (rollout_window controls concurrency).
2. For each window, calls agent workers → 1:1 DataProto (each row = one episode-trial).
3. Extracts trajectory data (all decisions) from extra_fields["trajectory"].
4. Flattens: each decision → one padded row (prompt/response/mask/rm_scores/position_ids).
5. Returns the flattened DataProto (rows = total decisions across all trajectories).

The trainer receives a batch where each row is an INDEPENDENT SimpleVLN decision, and
uid groups all decisions of the same episode start for trajectory-level GRPO. Validation retains one result per episode.
"""
import asyncio
import os
import time

import numpy as np
import ray
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await

from recipe.vln_simplevln.batch_utils import object_vector, padding_multiple


def _fmt_dur(seconds: float) -> str:
    """Seconds -> 'm:ss' or 'h:mm:ss'."""
    s = int(max(0, seconds))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class VLNOnlineRolloutManager(AgentLoopManager):
    """Full-episode rollout + flatten decisions, with rollout_window concurrency control."""

    def _get_rollout_window(self, total: int) -> int:
        """Read VLN_ROLLOUT_WINDOW from env var; default = total (no windowing)."""
        w = os.environ.get("VLN_ROLLOUT_WINDOW")
        if w is not None:
            try:
                w = int(w)
                if w > 0:
                    return w
            except ValueError:
                pass
        return total

    def _extract_decisions(self, one_to_one: DataProto):
        """Extract decision rows + reward/success from 1:1 rollout output.

        Returns (all_rows: list[dict], traj_rewards: list[float], traj_successes: list[float]).
        """
        trajectories = one_to_one.non_tensor_batch.get("trajectory")
        if trajectories is None:
            return [], [], []

        all_rows = []
        traj_rewards = []
        traj_successes = []
        for traj_data in trajectories:
            if traj_data is None:
                continue
            group_uid = traj_data["group_uid"]
            trajectory_uid = traj_data["trajectory_uid"]
            reward = traj_data["reward"]
            num_decisions = traj_data["num_decisions"]
            decisions = traj_data.get("decisions", [])
            image_buffer = traj_data.get("image_buffer", [])
            image_buffer_ref = ray.put(image_buffer) if image_buffer else None
            traj_rewards.append(reward)
            metrics = traj_data.get("metrics", {})
            traj_successes.append(float(metrics.get("success", 0.0)))

            for dec in decisions:
                all_rows.append({
                    "prompt_ids": dec["prompt_ids"],
                    "response_ids": dec["response_ids"],
                    "response_logprobs": dec.get("response_logprobs"),
                    "response_mask": dec["response_mask"],
                    "image_buffer_ref": image_buffer_ref,
                    "image_indices": dec.get("image_indices"),
                    "raw_prompt": dec.get("raw_prompt"),
                    "mm_processor_kwargs": dec.get("mm_processor_kwargs"),
                    "position_ids": dec.get("position_ids"),
                    "turn_id": dec["turn_id"],
                    "action_text": dec.get("action_text", ""),
                    "is_stop_action": dec.get("is_stop_action", False),
                    "uid": group_uid,
                    "trajectory_uid": trajectory_uid,
                    "trajectory_reward": reward,
                    "trajectory_success": float(metrics.get("success", 0.0)),
                    "decision_loss_weight": 1.0 / max(num_decisions, 1),
                })

        return all_rows, traj_rewards, traj_successes

    def _build_output(self, all_rows, traj_rewards, traj_successes, meta_info) -> DataProto:
        """Build padded DataProto from collected decision rows."""
        prompt_length = self.rollout_config.prompt_length
        response_length = self.rollout_config.response_length

        pad_multiple = padding_multiple(self.config)
        real_decision_count = len(all_rows)
        remainder = len(all_rows) % pad_multiple
        if remainder:
            pad_count = pad_multiple - remainder
            pad_token_id = int(meta_info.get("pad_token_id", 0) or 0)
            for pad_idx in range(pad_count):
                all_rows.append({
                    "prompt_ids": [pad_token_id],
                    "response_ids": [],
                    "response_logprobs": None,
                    "response_mask": [],
                    "trajectory_reward": 0.0,
                    "trajectory_success": 0.0,
                    "decision_loss_weight": 0.0,
                    "turn_id": -1,
                    "action_text": "",
                    "image_buffer_ref": None,
                    "image_indices": [],
                    "raw_prompt": "",
                    "mm_processor_kwargs": {},
                    "position_ids": [[0], [0], [0], [0]],
                    "uid": f"__vln_padding__{pad_idx}",
                    "trajectory_uid": f"__vln_padding__{pad_idx}",
                    "is_stop_action": False,
                    "is_padding": True,
                })

        n = len(all_rows)
        prompts_t = torch.zeros(n, prompt_length, dtype=torch.long)
        responses_t = torch.zeros(n, response_length, dtype=torch.long)
        attention_mask = torch.zeros(n, prompt_length + response_length, dtype=torch.long)
        response_mask = torch.zeros(n, response_length, dtype=torch.long)
        rm_scores = torch.zeros(n, response_length, dtype=torch.float32)
        position_ids = torch.zeros(n, 4, prompt_length + response_length, dtype=torch.long)

        for i, row in enumerate(all_rows):
            original_prompt_ids = row["prompt_ids"]
            original_response_ids = row["response_ids"]
            if len(original_prompt_ids) > prompt_length:
                raise ValueError("Cannot truncate a VLN multimodal prompt during training")
            p_ids = original_prompt_ids
            r_ids = original_response_ids[:response_length]
            r_mask = row["response_mask"][:response_length]

            pad_len = prompt_length - len(p_ids)
            prompts_t[i, pad_len:] = torch.tensor(p_ids, dtype=torch.long)
            attention_mask[i, pad_len:prompt_length] = 1

            responses_t[i, :len(r_ids)] = torch.tensor(r_ids, dtype=torch.long)
            attention_mask[i, prompt_length:prompt_length + len(r_ids)] = 1
            response_mask[i, :len(r_mask)] = torch.tensor(r_mask, dtype=torch.long)

            last_valid = len(r_ids) - 1
            if last_valid >= 0:
                rm_scores[i, last_valid] = float(row["trajectory_reward"])

            row_position_ids = row.get("position_ids")
            if row_position_ids is None:
                raise ValueError(
                    "VLN decision is missing processor-aligned 4-axis position_ids; "
                    f"trajectory_uid={row.get('trajectory_uid')!r}, turn_id={row.get('turn_id')!r}"
                )
            row_position_ids = torch.as_tensor(row_position_ids, dtype=torch.long)
            expected_length = len(original_prompt_ids) + len(original_response_ids)
            if row_position_ids.ndim != 2 or row_position_ids.shape != (4, expected_length):
                raise ValueError(
                    "VLN decision position_ids must have shape "
                    f"(4, prompt+response={expected_length}), got {tuple(row_position_ids.shape)}; "
                    f"trajectory_uid={row.get('trajectory_uid')!r}, turn_id={row.get('turn_id')!r}"
                )

            prompt_start = len(original_prompt_ids) - len(p_ids)
            position_ids[i, :, pad_len:prompt_length] = row_position_ids[
                :, prompt_start : len(original_prompt_ids)
            ]
            position_ids[i, :, prompt_length : prompt_length + len(r_ids)] = row_position_ids[
                :, len(original_prompt_ids) : len(original_prompt_ids) + len(r_ids)
            ]

        input_ids = torch.cat([prompts_t, responses_t], dim=1)

        non_tensor = {
            "uid": np.array([row["uid"] for row in all_rows], dtype=object),
            "trajectory_uid": np.array([row["trajectory_uid"] for row in all_rows], dtype=object),
            "trajectory_reward": np.array([row["trajectory_reward"] for row in all_rows], dtype=np.float32),
            "trajectory_success": np.array([row["trajectory_success"] for row in all_rows], dtype=np.float32),
            "decision_loss_weight": np.array([row["decision_loss_weight"] for row in all_rows], dtype=np.float32),
            "turn_id": np.array([row["turn_id"] for row in all_rows], dtype=np.int32),
            "action_text": np.array([row["action_text"] for row in all_rows], dtype=object),
            "image_buffer_ref": np.array([row.get("image_buffer_ref") for row in all_rows], dtype=object),
            "image_indices": object_vector(row.get("image_indices") for row in all_rows),
            "raw_prompt": np.array([row.get("raw_prompt") or "" for row in all_rows], dtype=object),
            "mm_processor_kwargs": np.array([row.get("mm_processor_kwargs") or {} for row in all_rows], dtype=object),
            # Mark padding rows so downstream metrics (patches.py) can drop them; real
            # decisions never set this key, padding rows in _build_output set is_padding=True.
            "is_padding": np.array([bool(row.get("is_padding", False)) for row in all_rows], dtype=bool),
        }

        from tensordict import TensorDict
        batch = TensorDict({
            "prompts": prompts_t,
            "responses": responses_t,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "rm_scores": rm_scores,
        }, batch_size=n)

        output = DataProto(batch=batch, non_tensor_batch=non_tensor)
        output.meta_info = dict(meta_info)
        output.meta_info["vln_flattened"] = True
        output.meta_info["seqlen_sorted_indices"] = list(range(n))

        if traj_rewards:
            r = np.array(traj_rewards)
            s = np.array(traj_successes)
            print(f"[VLN rollout] {len(traj_rewards)} trajectories, "
                  f"{real_decision_count} decisions (padded {n}), "
                  f"SR={s.mean():.1%}, reward={r.mean():.3f}±{r.std():.3f}")
            output.meta_info["vln_metrics"] = {
                "vln/traj_reward/mean": float(r.mean()),
                "vln/traj_reward/std": float(r.std()),
                "vln/traj_reward/max": float(r.max()),
                "vln/traj_reward/min": float(r.min()),
                "vln/traj_sr": float(s.mean()),
                "vln/traj_count": len(traj_rewards),
                "vln/avg_decisions_per_traj": real_decision_count / len(traj_rewards),
            }

        return output

    async def _dispatch_window(self, prompts: DataProto) -> DataProto:
        """Dispatch nonempty, ordered chunks, including a partial final window."""
        count = min(len(prompts), len(self.agent_loop_workers))
        if count == 0:
            raise ValueError("VLN rollout requires episodes and at least one agent worker")
        width, extra = divmod(len(prompts), count)
        tasks = []
        start = 0
        for i, worker in enumerate(self.agent_loop_workers[:count]):
            end = start + width + (i < extra)
            tasks.append(worker.generate_sequences.remote(prompts[start:end]))
            start = end
        outputs = await asyncio.gather(*tasks)
        metrics = []
        for out in outputs:
            metrics.extend(out.meta_info.pop("metrics", []))
        output = DataProto.concat(outputs)
        if len(output) != len(prompts):
            raise ValueError("Agent workers must return one result per episode")
        output.meta_info = {**prompts.meta_info, **output.meta_info, "metrics": metrics}
        return output

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        total = len(prompts)
        if total == 0:
            raise ValueError("VLN rollout received an empty batch")
        window_size = self._get_rollout_window(total)

        if prompts.meta_info.get("validate", False):
            # The stock validation loop unpads and joins by episode index. Keep every
            # episode in order; a failed validation request must not be silently dropped.
            outputs, metrics = [], []
            for start in range(0, total, window_size):
                out = await self._dispatch_window(prompts[start:start + window_size])
                metrics.extend(out.meta_info.pop("metrics", []))
                outputs.append(out)
            output = DataProto.concat(outputs)
            output.meta_info["timing"] = self._aggregate_timing(metrics)
            return output

        scheduler = os.environ.get("VLN_ROLLOUT_SCHEDULER", "window").lower()
        if scheduler == "sliding":
            return await self._generate_sequences_sliding(prompts, window_size, total)
        if scheduler != "window":
            raise ValueError(f"Unknown VLN rollout scheduler: {scheduler}")

        all_rows, all_rewards, all_successes, metrics = [], [], [], []
        meta_info = dict(prompts.meta_info)
        failed_windows = 0
        num_windows = (total + window_size - 1) // window_size
        for start in range(0, total, window_size):
            end = min(start + window_size, total)
            print(f"[VLN rollout] window {start // window_size + 1}/{num_windows} "
                  f"({end - start} rollouts, range [{start}:{end}])")
            try:
                out = await self._dispatch_window(prompts[start:end])
            except Exception as exc:
                failed_windows += 1
                print(f"[VLN rollout] window failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            metrics.extend(out.meta_info.pop("metrics", []))
            meta_info.update(out.meta_info)
            rows, rewards, successes = self._extract_decisions(out)
            all_rows.extend(rows)
            all_rewards.extend(rewards)
            all_successes.extend(successes)

        if failed_windows:
            print(f"[VLN rollout] {failed_windows}/{num_windows} windows failed and were dropped",
                  flush=True)
        if not all_rows:
            raise RuntimeError("VLN rollout produced no usable decisions; check the environment service")
        output = self._build_output(all_rows, all_rewards, all_successes, meta_info)
        output.meta_info["timing"] = self._aggregate_timing(metrics)
        return output

    async def _generate_sequences_sliding(self, prompts, max_concurrent, total):
        """Bounded rolling queue: index queue + fixed worker loops, no barrier.

        Concurrency = min(VLN_ROLLOUT_WINDOW, num_workers, total).
        Workers pull episodes from a shared index queue; as each finishes,
        the next starts immediately — no window barrier.
        """
        concurrency = min(max_concurrent, len(self.agent_loop_workers), total)
        active_workers = self.agent_loop_workers[:concurrency]

        index_queue = asyncio.Queue()
        for i in range(total):
            index_queue.put_nowait(i)

        results = [None] * total
        all_metrics = []
        output_meta = {}
        completed = [0]
        failed = [0]
        failed_indices = []
        failed_reasons = {}

        async def worker_loop(worker):
            while True:
                try:
                    idx = index_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                single = prompts[idx:idx + 1]
                if hasattr(prompts, "meta_info"):
                    single.meta_info = prompts.meta_info

                # Per-episode fault isolation: one episode's env failure — a transient
                # env_server 503 after reset() burns its retry budget, a worker crash, or
                # a Ray task error — must NOT abort the whole rollout. Before this guard a
                # single raise here propagated out of asyncio.gather() and killed the
                # entire (multi-node) training run. The env_server pool self-heals (workers
                # respawn, a transient scene-loading failure clears), so drop just this episode
                # (results[idx] stays None → excluded from the aggregation below) and keep
                # going. A total wipeout still raises via the `if not all_rows` guard.
                try:
                    out = await worker.generate_sequences.remote(single)
                    rows, rewards, successes = self._extract_decisions(out)
                    results[idx] = (rows, rewards, successes)

                    worker_metrics = out.meta_info.get("metrics", [])
                    all_metrics.extend(worker_metrics)
                    for k, v in out.meta_info.items():
                        if k != "metrics" and k not in output_meta:
                            output_meta[k] = v
                except Exception as exc:
                    # Aggregate failures by cause to keep rollout logs readable.
                    failed[0] += 1
                    failed_indices.append(idx)
                    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
                    cause = [ln for ln in lines if "Error" in ln or "Exception" in ln]
                    reason = cause[-1] if cause else (lines[-1] if lines else type(exc).__name__)
                    failed_reasons[reason[:200]] = failed_reasons.get(reason[:200], 0) + 1

                completed[0] += 1
                if completed[0] % concurrency == 0 or completed[0] == total:
                    el = time.time() - t0
                    rate = completed[0] / el if el > 0 else 0.0
                    eta = (total - completed[0]) / rate if rate > 0 else 0.0
                    print(f"[VLN rollout] sliding: {completed[0]}/{total} episodes "
                          f"| elapsed {_fmt_dur(el)} | ETA {_fmt_dur(eta)} | {rate:.2f} ep/s"
                          + (f" | {failed[0]} failed" if failed[0] else ""))

        print(f"[VLN rollout] sliding mode: {total} episodes, concurrency={concurrency}")
        t0 = time.time()
        await asyncio.gather(*(worker_loop(w) for w in active_workers))

        if failed[0]:
            preview = failed_indices[:20]
            top = sorted(failed_reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
            causes = "; ".join(f"{n}× {r}" for r, n in top)
            print(f"[VLN rollout] sliding: {failed[0]}/{total} episodes failed and were "
                  f"dropped (indices: {preview}{'...' if len(failed_indices) > 20 else ''})"
                  + (f" | top causes: {causes}" if causes else ""),
                  flush=True)

        all_rows, all_rewards, all_successes = [], [], []
        for res in results:
            if res:
                all_rows.extend(res[0])
                all_rewards.extend(res[1])
                all_successes.extend(res[2])

        if not all_rows:
            raise RuntimeError(
                f"[VLN rollout] sliding: no usable decisions — all {total} episodes "
                f"returned empty or failed ({failed[0]} raised errors, e.g. env_server "
                f"unreachable/starved). Check the env_server on this node (curl "
                f"http://127.0.0.1:8002/stats).")

        meta_info = dict(prompts.meta_info) if hasattr(prompts, "meta_info") else {}
        meta_info.update(output_meta)
        output = self._build_output(all_rows, all_rewards, all_successes, meta_info)
        output.meta_info["timing"] = self._aggregate_timing(all_metrics)
        return output

    def _aggregate_timing(self, all_metrics):
        """Aggregate per-episode metrics into timing dict (mirrors AgentLoopManager._performance_metrics)."""
        timing = {}
        if not all_metrics:
            return timing

        flat = [m for chunk in all_metrics for m in (chunk if isinstance(chunk, list) else [chunk])]
        if not flat:
            return timing

        t_gen = np.array([m["generate_sequences"] for m in flat])
        t_tool = np.array([m["tool_calls"] for m in flat])
        t_score = np.array([m["compute_score"] for m in flat])
        num_preempted = np.array([m["num_preempted"] for m in flat])

        for prefix, arr in [
            ("agent_loop/generate_sequences", t_gen),
            ("agent_loop/tool_calls", t_tool),
            ("agent_loop/compute_score", t_score),
            ("agent_loop/num_preempted", num_preempted),
        ]:
            timing[f"{prefix}/min"] = float(arr.min())
            timing[f"{prefix}/max"] = float(arr.max())
            timing[f"{prefix}/mean"] = float(arr.mean())

        slowest = int(np.argmax(t_gen + t_tool + t_score))
        timing["agent_loop/slowest/generate_sequences"] = float(t_gen[slowest])
        timing["agent_loop/slowest/tool_calls"] = float(t_tool[slowest])
        timing["agent_loop/slowest/compute_score"] = float(t_score[slowest])
        timing["agent_loop/slowest/num_preempted"] = float(num_preempted[slowest])

        return timing
