"""Pydantic schemas for the VLN Habitat env server API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    rgb_jpeg_base64: str
    width: int
    height: int
    step: int


class Metrics(BaseModel):
    distance_to_goal: float
    success: float
    spl: float
    oracle_success: float
    oracle_navigation_error: float
    collisions: float


class EpisodeInfo(BaseModel):
    scene_id: str
    episode_id: str
    instruction: str


# ── POST /v1/sessions ─────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    episode_id: str
    scene_id: Optional[str] = None   # composite key (scene::episode_id) to disambiguate splits
    dataset_name: Optional[str] = None   # None → env_server's primary (first) dataset


class CreateSessionResponse(BaseModel):
    session_id: str
    worker_id: int
    episode: EpisodeInfo
    observation: Observation
    metrics: Metrics
    done: bool
    done_reason: Optional[str] = None


# ── POST /v1/sessions/{session_id}/step ───────────────────────────────────────

class StepRequest(BaseModel):
    actions: list[int] = Field(min_length=1)  # 0=stop 1=fwd 2=left 3=right
    stop_on_done: bool = True


class StepResponse(BaseModel):
    session_id: str
    executed_actions: list[int]
    observation: Observation
    metrics: Metrics
    done: bool
    done_reason: Optional[str] = None  # episode_over|stop_action|max_episode_steps|worker_error|timeout


# ── GET /v1/sessions/{session_id}/metrics ─────────────────────────────────────

class MetricsResponse(BaseModel):
    session_id: str
    metrics: Metrics
    done: bool


# ── DELETE /v1/sessions/{session_id} ─────────────────────────────────────────

class DeleteSessionResponse(BaseModel):
    session_id: str
    released: bool


# ── GET /health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    ok: bool
    service: str = "vln-habitat-env-server"
    version: str = "0.1.0"


# ── GET /stats ────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    pool_size: int
    free_workers: int
    active_sessions: int
    unhealthy_workers: int
    uptime_sec: float
