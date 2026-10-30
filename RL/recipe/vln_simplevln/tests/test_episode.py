import asyncio

import httpx
import pytest

from recipe.vln_simplevln.action import parse_simplevln_action, to_atomic_chunk
from recipe.vln_simplevln.env_pool import VLNEnv
from recipe.vln_simplevln.full_episode_agent_loop import DecisionGen, run_episode


class Environment:
    instruction = 'Go forward'

    def __init__(self, done=False):
        self.done = done
        self.actions = []

    async def reset(self, extra):
        return 'frame0'

    async def step(self, actions):
        self.actions.extend(actions)
        return {'done': self.done}

    def current_jpeg_b64(self):
        return f'frame{len(self.actions)}'

    def metrics(self):
        return {'success': 0.0}


async def decide(*args):
    return DecisionGen(action_text='forward 75, right 45')


def test_action_chunk_respects_remaining_step_budget():
    env = Environment()
    traj = asyncio.run(run_episode(env, {'episode_id': '1'}, decide,
                                  group_uid='g', trajectory_uid='t', max_env_steps=4))
    assert env.actions == [1, 1, 1, 3]
    assert traj.metrics['env_steps'] == 4
    assert len(traj.image_buffer) == 5
    assert traj.decisions[0].atomic_chunk == env.actions


def test_reset_that_is_already_done_does_not_take_an_action():
    env = Environment(done=True)
    traj = asyncio.run(run_episode(env, {'episode_id': '1'}, decide,
                                  group_uid='g', trajectory_uid='t'))
    assert not traj.decisions and not env.actions


def test_reasoning_is_not_interpreted_as_a_navigation_action():
    text = '<think>stop, turn left\nfirst</think><answer>forward 25, right 15</answer>'
    assert to_atomic_chunk(parse_simplevln_action(text)) == [1, 3]


def test_lost_step_response_is_not_retried():
    async def check():
        calls = []

        def transport(request):
            calls.append(request)
            raise httpx.ReadTimeout('reply lost after action', request=request)

        env = VLNEnv('http://env.example')
        await env._client.aclose()
        env._client = httpx.AsyncClient(base_url='http://env.example', transport=httpx.MockTransport(transport))
        env.session_id = 's'
        try:
            with pytest.raises(httpx.ReadTimeout):
                await env.step([1])
            assert len(calls) == 1
        finally:
            await env._client.aclose()

    asyncio.run(check())
