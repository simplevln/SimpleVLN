"""FastAPI env server — session-based Habitat env pool.

Each session = one Habitat episode on one dedicated worker process.
TTL background task reclaims stale sessions (worker crash / client forgot to delete).
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from recipe.vln_simplevln.env_server.schemas import (
    CreateSessionRequest, CreateSessionResponse,
    DeleteSessionResponse, EpisodeInfo, HealthResponse,
    Metrics, MetricsResponse, Observation,
    StatsResponse, StepRequest, StepResponse,
)
from recipe.vln_simplevln.env_server.worker import worker_loop

_START_TIME = time.time()


# ── Worker handle ─────────────────────────────────────────────────────────────

class WorkerHandle:
    def __init__(self, worker_id: int, config_path: str, gpu_id: int = -1):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        ctx = mp.get_context("spawn")
        self.cmd_q: mp.Queue = ctx.Queue()
        self.resp_q: mp.Queue = ctx.Queue()
        self._call_lock = threading.RLock()
        self._closed = False
        self.proc = ctx.Process(
            target=worker_loop,
            args=(worker_id, config_path, self.cmd_q, self.resp_q, gpu_id),
            daemon=True,
        )
        try:
            self.proc.start()
        except Exception:
            self.cmd_q.close()
            self.resp_q.close()
            raise
        # Do NOT block on the ready signal here: that would serialize the whole pool's
        # startup (each worker waiting for the previous to finish Habitat init). Call
        # wait_ready() after every proc is started so they initialize in parallel.
        self.state: str = "idle"
        self.session_id: Optional[str] = None
        self.last_active: float = time.time()

    def wait_ready(self, timeout: float = 300.0, stop_event=None):
        deadline = time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Worker pool is closing")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"worker {self.worker_id} initialization timed out")
            try:
                ready = self.resp_q.get(timeout=min(1.0, remaining))
                break
            except queue.Empty:
                if not self.is_alive():
                    raise RuntimeError(f"worker {self.worker_id} exited during initialization")
        if not ready.get("ok"):
            raise RuntimeError(f"worker {self.worker_id} failed: {ready.get('error')}")

    def call(self, cmd: dict, timeout: float = 60.0) -> dict:
        with self._call_lock:
            self.last_active = time.time()
            if self.state == "unhealthy" or not self.is_alive():
                raise RuntimeError(f"worker {self.worker_id} is unavailable")
            self.cmd_q.put(cmd)
            try:
                return self.resp_q.get(timeout=timeout)
            except Exception:
                # A late response cannot safely be assigned to the next command.
                # Retire this process; the pool will start a replacement.
                self.state = "unhealthy"
                self.proc.terminate()
                self.proc.join(timeout=1)
                raise
            finally:
                self.last_active = time.time()

    def is_alive(self) -> bool:
        return not self._closed and self.proc.is_alive()

    def close(self):
        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.proc.is_alive():
                    self.cmd_q.put({"op": "close"})
                    self.proc.join(timeout=10)
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=5)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join(timeout=5)
                if not self.proc.is_alive():
                    self.proc.close()
            finally:
                for channel in (self.cmd_q, self.resp_q):
                    channel.cancel_join_thread()
                    channel.close()


# ── Worker pool ───────────────────────────────────────────────────────────────

class WorkerPool:
    def __init__(self, config_path: str, pool_size: int, session_ttl_sec: float,
                 gpu_ids: Optional[list[int]] = None):
        self.pool_size = pool_size
        self.session_ttl_sec = session_ttl_sec
        if gpu_ids is None:
            gpu_ids = [-1] * pool_size
        elif len(gpu_ids) < pool_size:
            gpu_ids = gpu_ids * ((pool_size // len(gpu_ids)) + 1)
            gpu_ids = gpu_ids[:pool_size]
        # Remember construction params so dead workers can be respawned later.
        self._config_path = config_path
        self._gpu_ids = gpu_ids
        self._session_to_worker: dict[str, WorkerHandle] = {}
        self._lock = threading.Lock()
        self._respawn_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending: list[WorkerHandle] = []
        # Start all worker processes first (cheap), then wait for their ready signals —
        # this lets the Habitat workers initialize in parallel instead of one-by-one.
        self.workers: list[WorkerHandle] = []
        try:
            for i in range(pool_size):
                self.workers.append(WorkerHandle(i, config_path, gpu_id=gpu_ids[i]))
            for w in self.workers:
                w.wait_ready(stop_event=self._stop_event)
        except Exception:
            self.close()
            raise

    def claim(self, session_id: str) -> Optional[WorkerHandle]:
        with self._lock:
            if self._stop_event.is_set():
                return None
            for w in self.workers:
                if w.state == "idle" and w.is_alive():
                    w.state = "busy"
                    w.session_id = session_id
                    w.last_active = time.time()
                    self._session_to_worker[session_id] = w
                    return w
        return None

    def get(self, session_id: str) -> Optional[WorkerHandle]:
        with self._lock:
            return self._session_to_worker.get(session_id)

    def call_session(self, session_id: str, cmd: dict, timeout: float = 60.0) -> dict:
        w = self.get(session_id)
        if w is None:
            raise HTTPException(404, detail=f"session {session_id} not found")
        # Never wait for a worker lock while holding the pool lock. Recheck ownership
        # after acquiring it: DELETE/TTL may have reassigned this worker in between.
        with w._call_lock:
            with self._lock:
                if self._session_to_worker.get(session_id) is not w:
                    raise HTTPException(404, detail=f"session {session_id} not found")
            return w.call(cmd, timeout=timeout)

    def _release_locked(self, session_id: str, w: WorkerHandle) -> bool:
        """Caller holds both the worker and pool locks."""
        if self._session_to_worker.get(session_id) is not w:
            return False
        self._session_to_worker.pop(session_id)
        if w.state != "unhealthy":
            w.state = "idle"
        w.session_id = None
        return True

    def release(self, session_id: str) -> bool:
        w = self.get(session_id)
        if w is None:
            return False
        with w._call_lock:
            with self._lock:
                return self._release_locked(session_id, w)

    def reap_stale(self):
        """Release idle sessions, checking activity and ownership under both locks."""
        with self._lock:
            sessions = list(self._session_to_worker.items())
        for sid, w in sessions:
            if not w._call_lock.acquire(blocking=False):
                continue
            try:
                with self._lock:
                    if time.time() - w.last_active > self.session_ttl_sec:
                        self._release_locked(sid, w)
            finally:
                w._call_lock.release()

    def respawn_dead(self):
        """Initialize replacements in parallel; shutdown cancels pending readiness."""
        if not self._respawn_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                if self._stop_event.is_set():
                    return
                dead = [(i, w) for i, w in enumerate(self.workers) if not w.is_alive()]
                for sid, w in list(self._session_to_worker.items()):
                    if not w.is_alive():
                        self._session_to_worker.pop(sid)
            replacements = []
            for i, old in dead:
                try:
                    with self._lock:
                        if self._stop_event.is_set():
                            break
                        new = WorkerHandle(i, self._config_path, gpu_id=self._gpu_ids[i])
                        self._pending.append(new)
                    replacements.append((i, old, new))
                except Exception as exc:
                    print(f"[env_server] respawn worker {i} spawn failed: {exc!r}", flush=True)
            for i, old, new in replacements:
                installed = False
                try:
                    new.wait_ready(stop_event=self._stop_event)
                    with self._lock:
                        if not self._stop_event.is_set() and self.workers[i] is old:
                            self.workers[i] = new
                            installed = True
                except Exception as exc:
                    if not self._stop_event.is_set():
                        print(f"[env_server] respawn worker {i} failed: {exc!r}", flush=True)
                finally:
                    with self._lock:
                        self._pending.remove(new)
                    if installed:
                        old.close()
                        print(f"[env_server] respawned worker {i}", flush=True)
                    else:
                        new.close()
        finally:
            self._respawn_lock.release()

    @property
    def free_workers(self) -> int:
        return sum(1 for w in self.workers if w.state == "idle" and w.is_alive())

    @property
    def unhealthy_workers(self) -> int:
        return sum(1 for w in self.workers if not w.is_alive())

    def close(self):
        with self._lock:
            self._stop_event.set()
            self._session_to_worker.clear()
            workers = [*self.workers, *self._pending]
        for w in workers:
            try:
                w.close()
            except Exception as exc:
                print(f"[env_server] closing worker {w.worker_id} failed: {exc!r}", flush=True)


# ── App state ─────────────────────────────────────────────────────────────────

_pool: Optional[WorkerPool] = None
_config_path: Optional[str] = None
_pool_size: int = 1
_session_ttl_sec: float = 1800.0
_gpu_ids: Optional[list[int]] = None


def _ttl_reaper(pool: WorkerPool):
    while not pool._stop_event.wait(30):
        try:
            pool.reap_stale()
            pool.respawn_dead()
        except Exception as exc:
            print(f"[env_server] pool maintenance failed: {exc!r}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pool
    assert _config_path, "call set_config() before starting uvicorn"
    pool = WorkerPool(_config_path, _pool_size, _session_ttl_sec, gpu_ids=_gpu_ids)
    _pool = pool
    t = threading.Thread(target=_ttl_reaper, args=(pool,), daemon=True)
    t.start()
    try:
        yield
    finally:
        pool.close()
        t.join()
        _pool = None


app = FastAPI(lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs(d: dict) -> Observation:
    o = d["obs"]
    return Observation(rgb_jpeg_base64=o["rgb_jpeg_base64"],
                       width=o["width"], height=o["height"], step=o["step"])

def _metrics(d: dict) -> Metrics:
    m = d["metrics"]
    return Metrics(**m)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(ok=True)


@app.get("/stats", response_model=StatsResponse)
def stats():
    assert _pool
    return StatsResponse(
        pool_size=_pool.pool_size,
        free_workers=_pool.free_workers,
        active_sessions=len(_pool._session_to_worker),
        unhealthy_workers=_pool.unhealthy_workers,
        uptime_sec=time.time() - _START_TIME,
    )


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    assert _pool
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    w = _pool.claim(session_id)
    if w is None:
        raise HTTPException(503, detail={"error": {"code": "NO_FREE_WORKER",
            "message": "No free Habitat worker", "retryable": True}})
    try:
        r = _pool.call_session(session_id, {"op": "reset", "episode_id": req.episode_id, "scene_id": req.scene_id,
                    "instruction": req.instruction, "session_id": session_id}, timeout=120)
    except Exception:
        _pool.release(session_id)
        raise
    if not r.get("ok"):
        _pool.release(session_id)
        raise HTTPException(400, detail=r.get("error"))
    return CreateSessionResponse(
        session_id=session_id,
        worker_id=w.worker_id,
        episode=EpisodeInfo(scene_id=r["scene_id"], episode_id=r["episode_id"],
                            instruction=r["instruction"]),
        observation=_obs(r),
        metrics=_metrics(r),
        done=r["done"],
    )


@app.post("/v1/sessions/{session_id}/step", response_model=StepResponse)
def step(session_id: str, req: StepRequest):
    assert _pool
    for a in req.actions:
        if a not in (0, 1, 2, 3):
            raise HTTPException(400, detail=f"invalid action {a}")
    try:
        r = _pool.call_session(session_id, {"op": "step", "actions": req.actions}, timeout=60)
    except Exception:
        _pool.release(session_id)
        raise
    if not r.get("ok"):
        _pool.release(session_id)
        raise HTTPException(500, detail=r.get("error"))
    if r["done"] and req.stop_on_done:
        _pool.release(session_id)
    return StepResponse(
        session_id=session_id,
        executed_actions=r["executed_actions"],
        observation=_obs(r),
        metrics=_metrics(r),
        done=r["done"],
        done_reason=r.get("done_reason"),
    )


@app.get("/v1/sessions/{session_id}/metrics", response_model=MetricsResponse)
def get_metrics(session_id: str):
    assert _pool
    try:
        r = _pool.call_session(session_id, {"op": "metrics"}, timeout=10)
    except Exception:
        _pool.release(session_id)
        raise
    if not r.get("ok"):
        raise HTTPException(500, detail=r.get("error"))
    return MetricsResponse(session_id=session_id, metrics=_metrics(r), done=r["done"])


@app.delete("/v1/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    assert _pool
    released = _pool.release(session_id)
    return DeleteSessionResponse(session_id=session_id, released=released)


@app.get("/v1/episodes")
def list_episodes(limit: int = 200):
    """Convenience: return episode ids from the first worker's dataset."""
    assert _pool
    if limit < 1:
        raise HTTPException(400, detail="limit must be positive")
    r = _pool.workers[0].call({"op": "episodes", "limit": limit}, timeout=15)
    if not r.get("ok"):
        raise HTTPException(500, detail=r.get("error"))
    return {"episode_ids": r["episode_ids"]}


def set_config(config_path: str, pool_size: int = 1, session_ttl_sec: float = 1800.0,
               gpu_ids: Optional[list[int]] = None):
    global _config_path, _pool_size, _session_ttl_sec, _gpu_ids
    if pool_size < 1 or session_ttl_sec <= 0:
        raise ValueError("Pool size and session TTL must be positive")
    if gpu_ids is not None and (not gpu_ids or any(gpu < 0 for gpu in gpu_ids)):
        raise ValueError("GPU IDs must be a nonempty list of nonnegative integers")
    _config_path = config_path
    _pool_size = pool_size
    _session_ttl_sec = session_ttl_sec
    _gpu_ids = gpu_ids
