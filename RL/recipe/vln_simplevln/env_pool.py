"""Async HTTP client for one Habitat trajectory.

Session lifecycle: reset() -> step() x N -> close(). Observations retain the
server's JPEG encoding. Reset retries are bounded; actions are sent once.
"""
import asyncio

import httpx


class VLNEnv:
    def __init__(self, base_url: str, dataset_name: str | None = None,
                 reset_timeout_s: float = 600.0, reset_retry_s: float = 2.0,
                 timeout: float = 120.0):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._dataset_name = dataset_name
        self._reset_timeout_s = reset_timeout_s
        self._reset_retry_s = reset_retry_s
        self.session_id: str | None = None
        self.instruction: str | None = None
        self._cur_b64: str | None = None
        self._last_metrics: dict | None = None
        self._done: bool = False

    async def reset(self, extra_info: dict) -> str:
        """Start one trajectory. Returns the first frame's JPEG base64."""
        body = {"episode_id": str(extra_info["episode_id"]),
                "scene_id": str(extra_info.get("scene_id", "")),
                "dataset_name": extra_info.get("dataset_name", self._dataset_name)}
        deadline = asyncio.get_event_loop().time() + self._reset_timeout_s
        while True:
            try:
                r = await self._client.post("/v1/sessions", json=body)
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                    httpx.TimeoutException):
                if asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(self._reset_retry_s)
                    continue
                raise
            if r.status_code in (500, 502, 503) and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(self._reset_retry_s)
                continue
            r.raise_for_status()
            break
        d = r.json()
        self.session_id = d["session_id"]
        self.instruction = d["episode"]["instruction"]
        self._cur_b64 = d["observation"]["rgb_jpeg_base64"]
        self._last_metrics = d["metrics"]
        self._done = d["done"]
        return self._cur_b64

    async def step(self, actions: list[int]) -> dict:
        """Execute actions once; a lost reply must not cause the actions to repeat."""
        r = await self._client.post(
            f"/v1/sessions/{self.session_id}/step",
            json={"actions": actions, "stop_on_done": True},
        )
        r.raise_for_status()
        d = r.json()
        self._cur_b64 = d["observation"]["rgb_jpeg_base64"]
        self._last_metrics = d["metrics"]
        self._done = d["done"]
        return d

    def current_jpeg_b64(self) -> str:
        return self._cur_b64

    @property
    def done(self) -> bool:
        return self._done

    def metrics(self) -> dict:
        return self._last_metrics or {}

    async def close(self) -> None:
        try:
            if self.session_id is not None:
                await self._client.delete(f"/v1/sessions/{self.session_id}")
        except httpx.HTTPError:
            # The server TTL reclaims a session if its connection has been lost.
            pass
        finally:
            await self._client.aclose()
