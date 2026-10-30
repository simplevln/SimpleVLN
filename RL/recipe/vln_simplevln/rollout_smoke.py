"""Standalone rollout smoke test: drive full episodes via env_server + vLLM OpenAI API.

Validates env_server session API + SimpleVLN prompt + action parser + run_episode
end-to-end without pulling in verl. Useful for debugging the episode loop.

Launch env_server (host, VLN conda env):
  python -m recipe.vln_simplevln.env_server.launch \\
    --config recipe/vln_simplevln/config/vln_datasets.yaml --port 8002 --pool-size 8

Launch vLLM (any GPU host):
  vllm serve <model> --port 8000 --max-model-len 8192 ...

Run:
  python -m recipe.vln_simplevln.rollout_smoke \\
    --env-url http://127.0.0.1:8002 --vllm-url http://127.0.0.1:8000 \\
    --model <served-name> --n-episodes 8 --concurrency 8 --out /tmp/trace.json
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

from recipe.vln_simplevln.env_pool import VLNEnv
from recipe.vln_simplevln.full_episode_agent_loop import DecisionGen, run_episode
from recipe.vln_simplevln.prompt import build_simplevln_messages_b64


def episode_request(key: str) -> dict:
    parts = key.split("::")
    if len(parts) == 2:
        scene_id, episode_id = parts
        return {"episode_id": episode_id, "scene_id": scene_id}
    return {"episode_id": key}


def make_decide(client: httpx.AsyncClient, model: str, temperature: float,
                top_p: float, max_tokens: int):
    async def decide(instruction: str, b64_buffer: list, _all_frames: list) -> DecisionGen:
        messages = build_simplevln_messages_b64(instruction, b64_buffer)
        r = await client.post("/v1/chat/completions", json={
            "model": model, "messages": messages,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        })
        r.raise_for_status()
        return DecisionGen(action_text=r.json()["choices"][0]["message"]["content"])
    return decide


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default="http://127.0.0.1:8002")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--episode-ids", help="Comma-separated episode IDs or scene::episode keys")
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-decisions", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("logs/rollout_smoke.json"))
    args = ap.parse_args()
    if min(args.n_episodes, args.concurrency, args.max_decisions) < 1:
        ap.error("Episode count, concurrency, and decision limit must be positive")

    # Fetch compound episode identities, or use an explicit list.
    episode_ids: list[str] = []
    if args.episode_ids:
        episode_ids = [e.strip() for e in args.episode_ids.split(",") if e.strip()]
    async with httpx.AsyncClient(base_url=args.env_url, timeout=30) as ec:
        if not episode_ids:
            r = await ec.get("/v1/episodes", params={"limit": args.n_episodes})
            r.raise_for_status()
            episode_ids = r.json()["episode_ids"]
    episode_ids = episode_ids[: args.n_episodes]
    if not episode_ids:
        raise RuntimeError("No matching episodes were returned by the environment service")

    vclient = httpx.AsyncClient(base_url=args.vllm_url, timeout=120.0)
    decide = make_decide(vclient, args.model, args.temperature, args.top_p, args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(ep: str):
        async with sem:
            env = VLNEnv(args.env_url)
            try:
                traj = await run_episode(
                    env, episode_request(ep),
                    decide, group_uid=ep, trajectory_uid=f"{ep}#0",
                    max_decisions=args.max_decisions,
                )
            finally:
                await env.close()
            print(f"ep={ep} SR={traj.reward} decisions={traj.metrics['num_decisions']} "
                  f"steps={traj.metrics['env_steps']} ne={traj.metrics.get('distance_to_goal',0):.2f}",
                  flush=True)
            return traj

    try:
        trajs = await asyncio.gather(*[one(ep) for ep in episode_ids])
    finally:
        await vclient.aclose()

    sr = sum(t.reward for t in trajs) / max(len(trajs), 1)
    print(f"\n=== {len(trajs)} episodes | SR={sr:.3f} ===")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump([{
            "episode_id": t.episode_id, "reward": t.reward, "metrics": t.metrics,
            "decisions": [{"turn_id": d.turn_id, "action_text": d.action_text,
                           "parsed": d.parsed_actions, "atomic": d.atomic_chunk}
                          for d in t.decisions],
        } for t in trajs], f, indent=2)
    print(f"trace -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
