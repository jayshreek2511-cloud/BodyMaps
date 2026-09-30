"""Per-user nnInteractive remote sessions.

The model stays on the GPU service. This module owns the HTTP client sessions,
their source volume, mutable target buffer, prompt replay log, and idle cleanup.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from services.interactive_mock import FakeInteractiveSession


class InteractiveError(RuntimeError):
    status_code = 400
    code = "interactive_error"


class InteractiveNotFound(InteractiveError):
    status_code = 404
    code = "session_not_found"


class InteractiveCapacity(InteractiveError):
    status_code = 503
    code = "server_at_capacity"


class InteractiveTimeout(InteractiveError):
    status_code = 504
    code = "server_timeout"


class InteractiveExpired(InteractiveError):
    status_code = 410
    code = "session_expired"


@dataclass
class InteractiveSession:
    session_id: str
    owner_id: str
    dataset: str
    case_id: str
    image: np.ndarray
    target: np.ndarray
    remote: Any
    seed_target: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.uint8))
    label_id: int | None = None
    room_id: str | None = None
    charged_users: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)


def make_remote_session():
    if os.environ.get("NNINTERACTIVE_MOCK", "false").lower() == "true":
        return FakeInteractiveSession()
    try:
        from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
    except ImportError as exc:
        raise InteractiveError(
            "nninteractive-client is not installed; install flask-server requirements"
        ) from exc
    kwargs = {"server_url": os.environ.get("NNINTERACTIVE_SERVER_URL", "http://127.0.0.1:1527")}
    api_key = os.environ.get("NNINTERACTIVE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return nnInteractiveRemoteInferenceSession(**kwargs)


def _is_expired(exc: BaseException) -> bool:
    return type(exc).__name__ == "SessionExpiredError"


def _is_capacity(exc: BaseException) -> bool:
    return type(exc).__name__ == "ServerAtCapacityError"


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or type(exc).__name__.lower() in {
        "timeoutexception", "readtimeout", "writetimeout", "connecttimeout", "pooltimeout"
    }


def _snapshot_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    result = dict(prompt)
    if isinstance(result.get("crop"), np.ndarray):
        result["crop"] = result["crop"].astype(np.uint8).tolist()
    return result


def _replay(remote, image: np.ndarray, target: np.ndarray, prompts: list[dict[str, Any]], seed_target: np.ndarray | None = None) -> None:
    remote.set_image(image[np.newaxis, ...])
    remote.set_target_buffer(target)
    if seed_target is not None and seed_target.size and seed_target.any():
        target[...] = seed_target
        remote.add_initial_seg_interaction(seed_target, run_prediction=False)
    for item in prompts:
        kind = item["type"]
        include = bool(item.get("include", True))
        if kind == "point":
            remote.add_point_interaction(item["coordinates"], include_interaction=include)
        elif kind == "bbox":
            remote.add_bbox_interaction(item["bounds"], include_interaction=include)
        elif kind == "scribble":
            remote.add_scribble_interaction(np.asarray(item["crop"], dtype=bool), include_interaction=include,
                                            interaction_bbox=item["interaction_bbox"])
        elif kind == "lasso":
            remote.add_lasso_interaction(np.asarray(item["crop"], dtype=bool), include_interaction=include,
                                         interaction_bbox=item["interaction_bbox"])


class InteractiveSessionManager:
    def __init__(self):
        self._sessions: dict[str, InteractiveSession] = {}
        self._lock = threading.RLock()
        self._stop_reaper = threading.Event()
        self._reaper = threading.Thread(target=self._reaper_loop, name="nninteractive-idle-reaper", daemon=True)
        self._reaper.start()

    def _reaper_loop(self) -> None:
        while not self._stop_reaper.wait(30):
            self.reap_idle()

    def active_count(self, owner_id: str) -> int:
        with self._lock:
            return sum(item.owner_id == owner_id for item in self._sessions.values())

    def create(self, owner_id: str, dataset: str, case_id: str, image: np.ndarray, room_id: str | None = None,
               initial_mask: np.ndarray | None = None, label_id: int | None = None) -> InteractiveSession:
        self.reap_idle()
        seed_target = np.zeros(image.shape, dtype=np.uint8) if initial_mask is None else np.ascontiguousarray(initial_mask, dtype=np.uint8)
        if seed_target.shape != image.shape:
            raise InteractiveError("Initial organ mask does not match the CT volume")
        target = seed_target.copy()
        try:
            remote = make_remote_session()
        except Exception as exc:
            if _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            if _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            raise
        try:
            remote.set_image(image[np.newaxis, ...])
            remote.set_target_buffer(target)
            if seed_target.any():
                remote.add_initial_seg_interaction(seed_target, run_prediction=False)
        except Exception as exc:
            try:
                remote.close()
            except Exception:
                pass
            if _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            if _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            raise
        item = InteractiveSession(str(uuid.uuid4()), owner_id, dataset, case_id, image, target, remote,
                                  seed_target=seed_target, label_id=label_id,
                                  room_id=room_id, charged_users={owner_id})
        with self._lock:
            self._sessions[item.session_id] = item
        return item

    def get(self, session_id: str, owner_id: str, room_id: str | None = None) -> InteractiveSession:
        with self._lock:
            item = self._sessions.get(session_id)
        if item is None or (item.owner_id != owner_id and (not room_id or item.room_id != room_id)):
            raise InteractiveNotFound("Interactive session not found")
        return item

    def find_room(self, room_id: str) -> InteractiveSession | None:
        with self._lock:
            return next((item for item in self._sessions.values() if item.room_id == room_id), None)

    def _replace_remote(self, item: InteractiveSession) -> None:
        old_remote = item.remote
        try:
            if old_remote is not None:
                old_remote.close()
        except Exception:
            pass
        prior_target = item.target.copy()
        replacement_target = prior_target.copy()
        replacement = None
        try:
            replacement = make_remote_session()
            _replay(replacement, item.image, replacement_target, item.prompts, item.seed_target)
        except Exception as exc:
            if replacement is not None:
                try:
                    replacement.close()
                except Exception:
                    pass
            item.remote = None
            item.target[...] = prior_target
            if _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            if _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            raise InteractiveExpired("nnInteractive session expired; replay recovery failed") from exc
        item.remote = replacement
        item.target[...] = replacement_target

    def _ensure_remote(self, item: InteractiveSession) -> None:
        if item.remote is None:
            self._replace_remote(item)

    def apply(self, item: InteractiveSession, kind: str, prompt: dict[str, Any]) -> list[list[int]]:
        self._ensure_remote(item)
        methods = {
            "point": lambda: item.remote.add_point_interaction(
                prompt["coordinates"], include_interaction=prompt["include"]
            ),
            "bbox": lambda: item.remote.add_bbox_interaction(
                prompt["bounds"], include_interaction=prompt["include"]
            ),
            "scribble": lambda: item.remote.add_scribble_interaction(
                prompt["crop"], include_interaction=prompt["include"],
                interaction_bbox=prompt["interaction_bbox"]
            ),
            "lasso": lambda: item.remote.add_lasso_interaction(
                prompt["crop"], include_interaction=prompt["include"],
                interaction_bbox=prompt["interaction_bbox"]
            ),
        }
        previous = item.target.copy()
        try:
            changed = methods[kind]()
        except Exception as exc:
            if _is_expired(exc):
                try:
                    self._replace_remote(item)
                    changed = methods[kind]()
                except Exception as retry_exc:
                    if _is_capacity(retry_exc):
                        raise InteractiveCapacity("nnInteractive server is at capacity") from retry_exc
                    if _is_timeout(retry_exc):
                        raise InteractiveTimeout("nnInteractive server timed out") from retry_exc
                    raise InteractiveExpired("nnInteractive session expired; replay recovery failed") from retry_exc
            elif _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            elif _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            else:
                raise
        item.prompts.append(_snapshot_prompt(prompt))
        item.last_used = time.time()
        if isinstance(changed, (list, tuple)) and len(changed) == 3:
            return [[int(pair[0]), int(pair[1])] for pair in changed]
        return _tight_bbox(previous != item.target)

    def undo(self, item: InteractiveSession) -> list[list[int]] | None:
        self._ensure_remote(item)
        before = item.target.copy()
        try:
            ok = item.remote.undo()
        except Exception as exc:
            if _is_expired(exc):
                self._replace_remote(item)
                raise InteractiveExpired("nnInteractive session expired; the session was restored. Retry the action.") from exc
            elif _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            elif _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            else:
                raise
        if ok and item.prompts:
            item.prompts.pop()
        item.last_used = time.time()
        return _tight_bbox(before != item.target) if ok else None

    def reset(self, item: InteractiveSession) -> None:
        self._ensure_remote(item)
        try:
            item.remote.reset_interactions()
        except Exception as exc:
            if _is_expired(exc):
                self._replace_remote(item)
                item.remote.reset_interactions()
            elif _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            elif _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            else:
                raise
        item.target[...] = item.seed_target
        if item.seed_target.any():
            item.remote.add_initial_seg_interaction(item.seed_target, run_prediction=False)
        item.prompts.clear()
        item.last_used = time.time()

    def select_target(self, item: InteractiveSession, label_id: int, seed: np.ndarray | None) -> None:
        self._ensure_remote(item)
        seed_target = np.zeros(item.image.shape, dtype=np.uint8) if seed is None else np.ascontiguousarray(seed, dtype=np.uint8)
        if seed_target.shape != item.image.shape:
            raise InteractiveError("Initial organ mask does not match the CT volume")
        try:
            item.remote.reset_interactions()
            item.target[...] = seed_target
            if seed_target.any():
                item.remote.add_initial_seg_interaction(seed_target, run_prediction=False)
        except Exception as exc:
            if _is_expired(exc):
                self._replace_remote(item)
                item.remote.reset_interactions()
                item.target[...] = seed_target
                if seed_target.any():
                    item.remote.add_initial_seg_interaction(seed_target, run_prediction=False)
            elif _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            elif _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            else:
                raise
        item.seed_target = seed_target
        item.label_id = label_id
        item.prompts.clear()
        item.last_used = time.time()

    def accept(self, item: InteractiveSession) -> None:
        self._ensure_remote(item)
        """Promote the current proposal to this organ's next refinement seed."""
        accepted_target = item.target.copy()
        try:
            item.remote.reset_interactions()
            item.target[...] = accepted_target
            item.seed_target = accepted_target
            item.prompts.clear()
            if item.seed_target.any():
                item.remote.add_initial_seg_interaction(item.seed_target, run_prediction=False)
        except Exception as exc:
            if _is_expired(exc):
                self._replace_remote(item)
                item.remote.reset_interactions()
                item.target[...] = accepted_target
                item.seed_target = accepted_target
                item.prompts.clear()
                if item.seed_target.any():
                    item.remote.add_initial_seg_interaction(item.seed_target, run_prediction=False)
            elif _is_capacity(exc):
                raise InteractiveCapacity("nnInteractive server is at capacity") from exc
            elif _is_timeout(exc):
                raise InteractiveTimeout("nnInteractive server timed out") from exc
            else:
                raise
        item.last_used = time.time()

    def close(self, session_id: str, owner_id: str, room_id: str | None = None) -> None:
        item = self.get(session_id, owner_id, room_id)
        with self._lock:
            self._sessions.pop(session_id, None)
        try:
            item.remote.close()
        finally:
            item.image = np.empty((0,), dtype=np.int16)
            item.target = np.empty((0,), dtype=np.uint8)

    def reap_idle(self) -> int:
        idle = max(30, int(os.environ.get("NNINTERACTIVE_IDLE_SECONDS", "900")))
        now = time.time()
        with self._lock:
            expired = [key for key, item in self._sessions.items() if now - item.last_used > idle]
            items = [self._sessions.pop(key) for key in expired]
        for item in items:
            try:
                item.remote.close()
            except Exception:
                pass
            item.image = np.empty((0,), dtype=np.int16)
            item.target = np.empty((0,), dtype=np.uint8)
        return len(items)

    def close_all(self) -> None:
        self._stop_reaper.set()
        with self._lock:
            items = list(self._sessions.values())
            self._sessions.clear()
        for item in items:
            try:
                item.remote.close()
            except Exception:
                pass


def _tight_bbox(changed: np.ndarray) -> list[list[int]] | None:
    points = np.argwhere(changed)
    if not len(points):
        return None
    return [[int(points[:, axis].min()), int(points[:, axis].max()) + 1] for axis in range(3)]


def encode_region(target: np.ndarray, bbox: list[list[int]] | None) -> dict[str, Any] | None:
    if bbox is None:
        return None
    slices = tuple(slice(a, b) for a, b in bbox)
    region = np.ascontiguousarray(target[slices].astype(np.uint8))
    return {
        "bbox": bbox,
        "shape": list(region.shape),
        "encoding": "zlib-base64-uint8",
        "data": base64.b64encode(zlib.compress(region.tobytes(), level=3)).decode("ascii"),
    }


def decode_crop(value: Any, bbox: list[list[int]], shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise InteractiveError("crop must be a 3D array for a single 2D slice")
    expected = tuple(end - start for start, end in bbox)
    if any(size < 1 for size in expected) or sum(size == 1 for size in expected) != 1:
        raise InteractiveError("interaction_bbox must describe a 2D region on one slice")
    if any(bbox[d][0] < 0 or bbox[d][1] > shape[d] for d in range(3)):
        raise InteractiveError("interaction_bbox is outside the volume")
    try:
        crop = np.asarray(value, dtype=np.uint8)
    except (ValueError, TypeError) as exc:
        raise InteractiveError("crop must contain only zero or one values") from exc
    if crop.shape != expected or crop.size > 1_000_000 or np.any(crop > 1):
        raise InteractiveError("crop dimensions or values are invalid")
    return crop.astype(bool)
