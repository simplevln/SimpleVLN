import asyncio
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from pydantic import ValidationError
from fastapi import HTTPException
import pytest

from recipe.vln_simplevln.env_server import server, worker
from recipe.vln_simplevln.env_server.schemas import StepRequest
from recipe.vln_simplevln.rollout_smoke import episode_request


def test_routing_stays_within_requested_dataset_and_reports_executed_actions(monkeypatch):
    r2r = SimpleNamespace(scene_id='mp3d/scene/scene.glb')
    rxr = SimpleNamespace(scene_id='mp3d/scene/scene.glb')

    class Env:
        episode_over = False
        current_episode = None

        def reset(self):
            self.episode_over = False
            return {'rgb': np.zeros((2, 2, 3), dtype=np.uint8),
                    'instruction': {'text': 'rxr' if self.current_episode is rxr else 'r2r'}}

        def step(self, action):
            obs = self.reset()
            self.episode_over = True
            return obs

        def get_metrics(self):
            return {'distance_to_goal': 4, 'oracle_navigation_error': 1}

        def close(self):
            pass

    env = Env()
    full = {'r2r::scene::1': r2r, 'rxr::scene::1': rxr}
    bare = {'r2r::1': r2r, 'rxr::1': rxr}
    monkeypatch.setattr(worker, '_load_env', lambda _: (env, full, bare, 'r2r'))
    commands, responses = queue.Queue(), queue.Queue()
    for cmd in [
        {'op': 'reset', 'episode_id': '1', 'scene_id': 'scene', 'dataset_name': 'missing'},
        {'op': 'reset', 'episode_id': '1', 'dataset_name': 'rxr'},
        {'op': 'step', 'actions': [1, 1, 1]},
        {'op': 'episodes', 'dataset_name': 'rxr', 'limit': 10},
        {'op': 'close'},
    ]:
        commands.put(cmd)
    worker.worker_loop(0, 'config.yaml', commands, responses)
    assert responses.get()['ok']  # Ready.
    assert not responses.get()['ok']  # Unknown dataset cannot fall back to R2R.
    reset = responses.get()
    assert reset['instruction'] == 'rxr'
    assert reset['metrics']['oracle_navigation_error'] == 1
    assert responses.get()['executed_actions'] == [1]
    assert episode_request(responses.get()['episode_ids'][0]) == {
        'episode_id': '1', 'scene_id': 'scene', 'dataset_name': 'rxr'}


def test_timeout_retires_worker_before_it_can_serve_another_session():
    handle = server.WorkerHandle.__new__(server.WorkerHandle)
    handle._call_lock = threading.RLock()
    handle._closed = False
    handle.worker_id = 0
    handle.state = 'busy'
    handle.session_id = 's'
    handle.cmd_q = Mock()
    handle.resp_q = Mock()
    handle.resp_q.get.side_effect = queue.Empty
    handle.proc = Mock()
    handle.proc.is_alive.return_value = True
    with pytest.raises(queue.Empty):
        handle.call({'op': 'step', 'actions': [1]}, timeout=0.01)
    handle.proc.terminate.assert_called_once()
    assert handle.state == 'unhealthy'
    pool = server.WorkerPool.__new__(server.WorkerPool)
    pool._lock = threading.Lock()
    pool._session_to_worker = {'s': handle}
    pool.release('s')
    assert handle.state == 'unhealthy'


def test_ttl_does_not_reassign_a_worker_with_a_request_in_progress():
    handle = SimpleNamespace(_call_lock=threading.Lock(), last_active=time.time() - 100)
    handle._call_lock.acquire()
    pool = server.WorkerPool.__new__(server.WorkerPool)
    pool._lock = threading.Lock()
    pool._session_to_worker = {'s': handle}
    pool.session_ttl_sec = 1
    pool.release = Mock()
    pool.reap_stale()
    pool.release.assert_not_called()


def test_empty_action_request_is_rejected():
    with pytest.raises(ValidationError):
        StepRequest(actions=[])


def test_smoke_rejects_conflicting_dataset_name():
    with pytest.raises(ValueError):
        episode_request('rxr::scene::1', 'r2r')


class FakeWorker:
    def __init__(self, worker_id, config_path, gpu_id):
        self.worker_id = worker_id
        self.state = 'idle'
        self.session_id = None
        self.last_active = time.time()
        self._call_lock = threading.RLock()
        self.alive = True
        self.call = Mock(return_value={'ok': True})

    def wait_ready(self, **kwargs):
        pass

    def is_alive(self):
        return self.alive

    def close(self):
        self.alive = False


def test_queued_request_cannot_act_on_reassigned_worker(monkeypatch):
    monkeypatch.setattr(server, 'WorkerHandle', FakeWorker)
    pool = server.WorkerPool('config.yaml', 1, 10)
    worker = pool.claim('old')
    fetched, proceed = threading.Event(), threading.Event()
    original_get = pool.get

    def delayed_get(sid):
        result = original_get(sid)
        if threading.current_thread().name == 'old-request':
            fetched.set()
            assert proceed.wait(2)
        return result

    monkeypatch.setattr(pool, 'get', delayed_get)
    errors = []

    def request():
        try:
            pool.call_session('old', {'op': 'step', 'actions': [1]})
        except HTTPException as exc:
            errors.append(exc.status_code)

    t = threading.Thread(target=request, name='old-request')
    t.start()
    try:
        assert fetched.wait(2)
        assert pool.release('old')
        assert pool.claim('new') is worker
    finally:
        proceed.set()
        t.join(2)
    assert not t.is_alive()
    assert errors == [404]
    worker.call.assert_not_called()
    assert pool.get('new') is worker


def test_release_waits_for_active_request(monkeypatch):
    monkeypatch.setattr(server, 'WorkerHandle', FakeWorker)
    pool = server.WorkerPool('config.yaml', 1, 10)
    worker = pool.claim('old')
    attempting = threading.Event()
    lock = threading.RLock()

    class ObservedLock:
        def __enter__(self):
            attempting.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    worker._call_lock = ObservedLock()
    lock.acquire()  # An in-flight RPC owns this lock.
    t = threading.Thread(target=pool.release, args=('old',))
    t.start()
    try:
        assert attempting.wait(2)
        assert pool.claim('new') is None
        assert pool.get('old') is worker
    finally:
        lock.release()
        t.join(2)
    assert not t.is_alive()
    assert pool.claim('new') is worker


@pytest.mark.parametrize('failure', ['spawn', 'ready'])
def test_failed_pool_startup_closes_started_workers(monkeypatch, failure):
    started = []

    class FailingWorker(FakeWorker):
        def __init__(self, worker_id, *args, **kwargs):
            if worker_id == 1 and failure == 'spawn':
                raise RuntimeError('spawn failed')
            super().__init__(worker_id, *args, **kwargs)
            started.append(self)

        def wait_ready(self, **kwargs):
            if self.worker_id == 1:
                raise RuntimeError('ready failed')

    monkeypatch.setattr(server, 'WorkerHandle', FailingWorker)
    with pytest.raises(RuntimeError, match='failed'):
        server.WorkerPool('config.yaml', 2, 10)
    assert started and all(not w.alive for w in started)


def test_shutdown_does_not_respawn_workers(monkeypatch):
    factory = Mock(side_effect=FakeWorker)
    monkeypatch.setattr(server, 'WorkerHandle', factory)
    pool = server.WorkerPool('config.yaml', 1, 10)
    pool.close()
    pool.respawn_dead()
    assert factory.call_count == 1
    assert pool.claim('new') is None


def test_shutdown_cancels_worker_replacement(monkeypatch):
    monkeypatch.setattr(server, 'WorkerHandle', FakeWorker)
    pool = server.WorkerPool('config.yaml', 1, 10)
    pool.workers[0].alive = False
    waiting = threading.Event()
    replacements = []

    class Replacement(FakeWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            replacements.append(self)

        def wait_ready(self, stop_event):
            waiting.set()
            assert stop_event.wait(2)
            raise RuntimeError('pool closed during initialization')

    monkeypatch.setattr(server, 'WorkerHandle', Replacement)
    t = threading.Thread(target=pool.respawn_dead)
    t.start()
    try:
        assert waiting.wait(2)
    finally:
        pool.close()
        t.join(2)
    assert not t.is_alive()
    assert replacements and all(not w.alive for w in replacements)
    assert not pool._pending


def test_ttl_rechecks_activity_after_acquiring_worker_lock(monkeypatch):
    monkeypatch.setattr(server, 'WorkerHandle', FakeWorker)
    pool = server.WorkerPool('config.yaml', 1, 1)
    worker = pool.claim('active')
    worker.last_active = time.time() - 100
    real_lock = worker._call_lock

    class RecentlyUsedLock:
        def acquire(self, **kwargs):
            # A request finished after the reaper took its session snapshot.
            worker.last_active = time.time()
            return real_lock.acquire(**kwargs)

        def release(self):
            real_lock.release()

    worker._call_lock = RecentlyUsedLock()
    pool.reap_stale()
    assert pool.get('active') is worker


def test_lifespan_closes_pool_when_application_raises(monkeypatch):
    monkeypatch.setattr(server, 'WorkerHandle', FakeWorker)
    monkeypatch.setattr(server, '_config_path', 'config.yaml')
    monkeypatch.setattr(server, '_pool_size', 1)
    pools = []

    async def run():
        async with server.lifespan(server.app):
            pools.append(server._pool)
            raise RuntimeError('application error')

    with pytest.raises(RuntimeError, match='application error'):
        asyncio.run(run())
    assert server._pool is None
    assert pools[0]._stop_event.is_set()
    assert all(not w.alive for w in pools[0].workers)
