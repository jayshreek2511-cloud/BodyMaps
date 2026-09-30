"""Filesystem-backed state for temporary, account-free Live Rooms.

Flask and the async WebSocket process both use this module.  Every mutation is
guarded by a process-local lock plus an advisory file lock, so sequence numbers
and snapshots remain consistent across the two processes without Redis or a
database.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import hmac
import html
import io
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import uuid
import weakref
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import nibabel as nib
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from services.live_quiz import (
    ALLOWED_TIMERS,
    QUIZ_PACK_ID,
    QuizPackError,
    QuizPackRegistry,
    choice_ids,
    consistency_for,
    correct_choice,
    get_registry,
    public_question,
    reveal_for,
)
from services.quiz_telemetry import QuizTelemetry

try:  # Linux production path; tests on other platforms still keep thread safety.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


ROOM_TTL = timedelta(hours=24)
MAX_EVENTS = 50_000
MAX_EVENT_LOG_BYTES = 512 * 1024 * 1024
MAX_ROOM_BYTES = 8 * 1024 * 1024 * 1024
MAX_EXPORT_BYTES = 4 * 1024 * 1024 * 1024
MAX_TEXT_NOTE = 4_000
MAX_TEXT_CHAT = 2_000
MAX_NAME = 32
MAX_CHAT_MESSAGES = 500
MAX_MASK_RANGES_PER_EVENT = 50_000
MAX_MASK_VOXELS_PER_EVENT = 8_000_000
MAX_PARTICIPANT_IDENTITIES = 128
SNAPSHOT_MASK_EVENT_INTERVAL = 50
SNAPSHOT_TIME_INTERVAL = timedelta(minutes=5)
MAX_REPLAY_EVENTS = 500
MAX_REPLAY_BYTES = 2 * 1024 * 1024
LIFECYCLE_EVENT_RESERVE = 64
LIFECYCLE_EVENT_BYTE_RESERVE = 1024 * 1024

logger = logging.getLogger(__name__)

MEASUREMENT_TOOLS = {
    "Length",
    "Probe",
    "RectangleROI",
    "EllipticalROI",
    "PlanarFreehandROI",
    "Bidirectional",
    "Angle",
    "CobbAngle",
    "ArrowAnnotate",
}
DURABLE_TYPES = {
    "measurement.upsert",
    "measurement.delete",
    "mask.patch",
    "note.upsert",
    "note.delete",
    "chat.add",
    "interactive.prompt",
    "interactive.accept",
}
QUIZ_EVENT_TYPES = {
    "quiz.started",
    "quiz.closed",
    "quiz.revealed",
    "quiz.advanced",
    "quiz.host_paused",
    "quiz.host_resumed",
    "quiz.host_promoted",
}


class LiveRoomError(RuntimeError):
    status_code = 400
    code = "invalid_request"


class RoomNotFound(LiveRoomError):
    status_code = 404
    code = "room_not_found"


class RoomUnauthorized(LiveRoomError):
    status_code = 401
    code = "invalid_room_key"


class RoomExpired(LiveRoomError):
    status_code = 410
    code = "room_expired"


class RoomFull(LiveRoomError):
    status_code = 507
    code = "room_event_log_full"


class ParticipantUnauthorized(LiveRoomError):
    status_code = 401
    code = "invalid_participant_resume"


class ParticipantLimit(LiveRoomError):
    status_code = 409
    code = "participant_identity_limit"


@dataclass(frozen=True)
class CaseFiles:
    ct_path: Path
    mask_path: Path
    geometry_hash: str
    dimensions: tuple[int, int, int]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def hash_room_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _clean_text(value: Any, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise LiveRoomError("Text value must be a string")
    # React renders these values as text.  Remove control characters as an extra
    # barrier for JSON/CSV/report exports while preserving ordinary whitespace.
    cleaned = "".join(ch for ch in value if ch in "\n\t" or ord(ch) >= 32).strip()
    if required and not cleaned:
        raise LiveRoomError("Text value cannot be empty")
    if len(cleaned) > limit:
        raise LiveRoomError(f"Text value exceeds {limit} characters")
    return cleaned


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


class LiveRoomStore:
    def __init__(
        self,
        sessions_dir: str | os.PathLike[str],
        pants_path: str | os.PathLike[str] | None = None,
        *,
        now: Callable[[], datetime] = utcnow,
        quiz_registry: QuizPackRegistry | None = None,
    ) -> None:
        self.root = Path(sessions_dir).resolve() / "live_rooms"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.pants_path = Path(pants_path).resolve() if pants_path else None
        self._now = now
        self.quiz_registry = quiz_registry or get_registry()
        self.quiz_telemetry = QuizTelemetry(sessions_dir)
        self._locks: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
        self._locks_guard = threading.Lock()
        self._event_id_cache: dict[str, tuple[int, set[str]]] = {}

    @staticmethod
    def validate_room_id(room_id: str) -> str:
        try:
            parsed = uuid.UUID(str(room_id))
        except (ValueError, AttributeError) as exc:
            raise RoomNotFound("Room not found") from exc
        canonical = str(parsed)
        if canonical != str(room_id).lower():
            raise RoomNotFound("Room not found")
        return canonical

    def _room_dir(self, room_id: str) -> Path:
        canonical = self.validate_room_id(room_id)
        path = self.root / canonical
        # UUID validation above prevents traversal.  Resolve check also guards a
        # mistakenly-created symlink inside live_rooms.
        if path.exists() and path.resolve().parent != self.root:
            raise RoomNotFound("Room not found")
        return path

    def _thread_lock(self, room_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(room_id, threading.RLock())

    @contextmanager
    def _locked(self, room_id: str, *, require_exists: bool = True) -> Iterator[Path]:
        room_id = self.validate_room_id(room_id)
        room_dir = self._room_dir(room_id)
        if require_exists and not (room_dir / "metadata.json").is_file():
            raise RoomNotFound("Room not found")
        with self._thread_lock(room_id):
            if require_exists and not (room_dir / "metadata.json").is_file():
                raise RoomNotFound("Room not found")
            room_dir.mkdir(parents=True, exist_ok=True)
            room_dir.chmod(0o700)
            for child in room_dir.iterdir():
                if child.is_file() and not child.is_symlink():
                    child.chmod(0o600)
            lock_path = room_dir / ".lock"
            with lock_path.open("a+b") as lock_file:
                os.chmod(lock_path, 0o600)
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield room_dir
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except FileNotFoundError as exc:
            raise RoomNotFound("Room not found") from exc
        if not isinstance(value, dict):
            raise LiveRoomError(f"Invalid room file: {path.name}")
        return value

    def _load_metadata(self, room_dir: Path, *, allow_expired: bool = False) -> dict[str, Any]:
        metadata = self._read_json(room_dir / "metadata.json")
        metadata.setdefault("mode", "review")
        metadata = self._recover_from_event_log(room_dir, metadata)
        if not allow_expired and parse_time(metadata["expires_at"]) <= self._now():
            raise RoomExpired("Room expired and was deleted")
        return metadata

    def _recover_from_event_log(self, room_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Replay committed events if a process stopped between event fsync and JSON replace."""
        self._truncate_incomplete_event_tail(room_dir)
        pending_path = room_dir / ".pending_event.json"
        if pending_path.is_file():
            pending = self._read_json(pending_path)
            event = pending.get("event")
            target_metadata = pending.get("metadata")
            derived = pending.get("derived")
            logged = (
                isinstance(event, dict)
                and isinstance(target_metadata, dict)
                and isinstance(derived, dict)
                and self._find_event(room_dir, str(event.get("event_id", ""))) is not None
            )
            if logged:
                for filename, value in derived.items():
                    if filename not in {"state.json", "quiz.json", "quiz_answers.json"} or not isinstance(value, dict):
                        raise LiveRoomError("Invalid pending room transaction")
                    _atomic_json(room_dir / filename, value)
                metadata = target_metadata
                if pending.get("rebuild_mask"):
                    self._rebuild_mask_caches_locked(room_dir, metadata)
                _atomic_json(room_dir / "metadata.json", metadata)
                self._invalidate_exports(room_dir)
                self._event_id_cache.pop(str(metadata.get("room_id", "")), None)
            pending_path.unlink(missing_ok=True)

        last_event = self._last_event(room_dir)
        latest_logged = int(last_event["seq"]) if last_event else 0
        latest_metadata = int(metadata.get("latest_seq", 0))
        if latest_logged <= latest_metadata:
            return metadata
        events = list(self._iter_events(room_dir))
        state = self._read_json(room_dir / "state.json")
        state.setdefault("measurements", {})
        state.setdefault("notes", {})
        state.setdefault("chat", [])
        state.setdefault("undone_event_ids", [])
        nvoxels = int(np.prod(metadata["dimensions"], dtype=np.int64))
        for event in events:
            if int(event["seq"]) <= latest_metadata:
                continue
            payload = event.get("payload") or {}
            event_type = event.get("type")
            if event_type == "measurement.upsert" and payload.get("measurement"):
                item = payload["measurement"]
                state["measurements"][item["id"]] = item
            elif event_type == "measurement.delete" and payload.get("id"):
                state["measurements"].pop(payload["id"], None)
            elif event_type == "note.upsert" and payload.get("note"):
                item = payload["note"]
                state["notes"][item["id"]] = item
            elif event_type == "note.delete" and payload.get("id"):
                state["notes"].pop(payload["id"], None)
            elif event_type == "chat.add" and payload.get("message"):
                if not any(item.get("id") == payload["message"]["id"] for item in state["chat"]):
                    state["chat"].append(payload["message"])
                    state["chat"] = state["chat"][-MAX_CHAT_MESSAGES:]
            elif event_type == "mask.patch":
                metadata["mask_events_since_snapshot"] = int(metadata.get("mask_events_since_snapshot", 0)) + 1
            if event.get("undo_of") and event["undo_of"] not in state["undone_event_ids"]:
                state["undone_event_ids"].append(event["undo_of"])
        metadata["latest_seq"] = latest_logged
        writers = self._writer_map(room_dir, nvoxels)
        writers[:] = 0
        for event in events:
            if event.get("type") != "mask.patch":
                continue
            for item in (event.get("payload") or {}).get("ranges", []):
                start = int(item["start"])
                writers[start : start + int(item["length"])] = int(event["seq"])
        writers.flush()
        del writers
        metadata["mask_values_seq"] = -1
        values = self._mask_value_map(room_dir, metadata)
        del values
        metadata["recent_event_ids"] = [event["event_id"] for event in events[-10_000:]]
        _atomic_json(room_dir / "state.json", state)
        _atomic_json(room_dir / "metadata.json", metadata)
        self._event_id_cache.pop(str(metadata.get("room_id", "")), None)
        return metadata

    @staticmethod
    def _truncate_incomplete_event_tail(room_dir: Path) -> None:
        """Discard only an unterminated final append before another writer continues."""
        path = room_dir / "events.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            return
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) == b"\n":
                return
            position = stream.tell() - 1
            while position > 0:
                size = min(64 * 1024, position)
                position -= size
                stream.seek(position)
                chunk = stream.read(size)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    stream.truncate(position + newline + 1)
                    stream.flush()
                    os.fsync(stream.fileno())
                    return
            stream.truncate(0)
            stream.flush()
            os.fsync(stream.fileno())

    def _rebuild_mask_caches_locked(self, room_dir: Path, metadata: dict[str, Any]) -> None:
        nvoxels = int(np.prod(metadata["dimensions"], dtype=np.int64))
        writers = self._writer_map(room_dir, nvoxels)
        writers[:] = 0
        for event in self._iter_events(room_dir):
            if event.get("type") != "mask.patch":
                continue
            for item in (event.get("payload") or {}).get("ranges", []):
                start = int(item["start"])
                writers[start : start + int(item["length"])] = int(event["seq"])
        writers.flush()
        del writers
        metadata["mask_values_seq"] = -1
        values = self._mask_value_map(room_dir, metadata)
        del values

    @staticmethod
    def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "room_id",
            "case_id",
            "resolution",
            "created_at",
            "expires_at",
            "geometry_hash",
            "dimensions",
            "latest_seq",
            "mode",
            "quiz_pack_id",
            "quiz_pack_version",
            "quiz_playlist_id",
            "quiz_timer_seconds",
        }
        public = {key: metadata[key] for key in allowed if key in metadata}
        public.setdefault("mode", "review")
        return public

    def resolve_participant_identity(
        self,
        room_id: str,
        room_key: str,
        participant_id: str = "",
        resume_credential: str = "",
        host_claim: str = "",
    ) -> tuple[str, str | None, bool]:
        """Resume or mint an identity while retaining a host claim until ready is acknowledged."""
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            identities = metadata.setdefault("participant_credentials", {})
            if not isinstance(identities, dict):
                raise LiveRoomError("Participant credential store is invalid")
            # Protocol 1 rooms created by the previous release used this exact
            # field. Keep the fallback only for their 24-hour room lifetime and
            # migrate it to the claim/identity model on the first valid hello.
            claim_hash = metadata.get("quiz_host_claim_hash") or metadata.get("quiz_host_secret_hash")
            claiming_host = bool(host_claim)
            if claiming_host and (
                metadata.get("mode") != "quiz"
                or not claim_hash
                or not hmac.compare_digest(hash_room_key(host_claim), str(claim_hash))
            ):
                error = RoomUnauthorized("Invalid or already-consumed quiz host claim")
                error.code = "invalid_host_claim"
                raise error
            assigned_claim_id = metadata.get("quiz_host_claim_participant_id")
            if not assigned_claim_id and claiming_host:
                current_host = metadata.get("quiz_host_participant_id")
                if current_host in identities:
                    assigned_claim_id = current_host
            if participant_id or resume_credential:
                if not participant_id or not resume_credential:
                    raise ParticipantUnauthorized("Participant ID and resume credential are both required")
                try:
                    canonical = str(uuid.UUID(participant_id))
                except (ValueError, AttributeError) as exc:
                    raise ParticipantUnauthorized("Participant resume credential is invalid") from exc
                record = identities.get(canonical)
                expected_hash = record.get("resume_hash", "") if isinstance(record, dict) else ""
                if not expected_hash or not hmac.compare_digest(
                    hash_room_key(resume_credential), expected_hash
                ):
                    raise ParticipantUnauthorized("Participant resume credential is invalid")
                credential = None
            elif claiming_host and assigned_claim_id in identities:
                canonical = str(assigned_claim_id)
                credential = secrets.token_urlsafe(32)
                identities[canonical]["resume_hash"] = hash_room_key(credential)
            else:
                identity_limit = MAX_PARTICIPANT_IDENTITIES
                if metadata.get("mode") == "quiz" and not claiming_host and not metadata.get("quiz_host_participant_id"):
                    identity_limit -= 1
                if len(identities) >= identity_limit:
                    raise ParticipantLimit("Room has reached its participant identity limit")
                canonical = str(uuid.uuid4())
                credential = secrets.token_urlsafe(32)
                identities[canonical] = {
                    "resume_hash": hash_room_key(credential),
                    "created_at": isoformat(self._now()),
                    "quiz_host": False,
                }
            record = identities[canonical]
            if claiming_host:
                claim_owner = metadata.get("quiz_host_claim_participant_id")
                current_host = metadata.get("quiz_host_participant_id")
                if claim_owner not in {None, canonical} or current_host not in {None, canonical}:
                    error = RoomUnauthorized("The quiz host claim belongs to another participant")
                    error.code = "invalid_host_claim"
                    raise error
                record["quiz_host"] = True
                metadata["quiz_host_participant_id"] = canonical
                metadata["quiz_host_claim_participant_id"] = canonical
                metadata["quiz_host_claim_hash"] = str(claim_hash)
                metadata.pop("quiz_host_secret_hash", None)
            _atomic_json(room_dir / "metadata.json", metadata)
            is_host = bool(record.get("quiz_host")) and metadata.get("quiz_host_participant_id") == canonical
            return canonical, credential, is_host

    def acknowledge_quiz_host_claim(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        lease_id: str,
    ) -> bool:
        """Consume the creator claim only after its room.ready reached the active host."""
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_host(metadata, participant_id, lease_id)
            claim_owner = metadata.get("quiz_host_claim_participant_id")
            if claim_owner not in {None, participant_id}:
                error = RoomUnauthorized("The quiz host claim belongs to another participant")
                error.code = "invalid_host_claim"
                raise error
            changed = bool(
                metadata.get("quiz_host_claim_hash")
                or metadata.get("quiz_host_secret_hash")
                or claim_owner
            )
            metadata["quiz_host_claim_hash"] = None
            metadata.pop("quiz_host_secret_hash", None)
            metadata.pop("quiz_host_claim_participant_id", None)
            _atomic_json(room_dir / "metadata.json", metadata)
            return changed

    def quiz_host_identity(self, room_id: str, room_key: str) -> str | None:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            return metadata.get("quiz_host_participant_id")

    def quiz_host_claim_pending(self, room_id: str, room_key: str) -> bool:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            return bool(
                metadata.get("quiz_host_claim_hash")
                or metadata.get("quiz_host_secret_hash")
            )

    def _resolve_case(self, case_id: str, resolution: str) -> CaseFiles:
        if not re.fullmatch(r"\d+", str(case_id)):
            raise LiveRoomError("case_id must be numeric")
        if resolution not in {"low", "full"}:
            raise LiveRoomError("resolution must be 'low' or 'full'")
        if self.pants_path is None:
            raise LiveRoomError("PANTS_PATH is not configured")
        pants_id = f"PanTS_{int(case_id):08d}"
        ct_full = self.pants_path / "image_only" / pants_id / "ct.nii.gz"
        mask_full = self.pants_path / "mask_only" / pants_id / "combined_labels.nii.gz"
        if not ct_full.is_file() or not mask_full.is_file():
            raise LiveRoomError("Dataset CT or segmentation is missing")
        ct_path, mask_path = ct_full, mask_full
        if resolution == "low":
            ct_low = ct_full.with_name("ct_lowres.nii.gz")
            if ct_low.is_file():
                ct_path = ct_low

        try:
            ct_img = nib.load(str(ct_path))
            mask_img = nib.load(str(mask_path))
        except Exception as exc:
            raise LiveRoomError("Dataset CT or segmentation could not be read") from exc
        if resolution == "full":
            if tuple(ct_img.shape[:3]) != tuple(mask_img.shape[:3]):
                raise LiveRoomError("CT and segmentation geometry do not match")
            if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
                raise LiveRoomError("CT and segmentation geometry do not match")
        else:
            # Fast rooms use a smaller CT texture but keep the editable labelmap on its
            # original grid. Cornerstone overlays independent volumes in world space.
            # Validate orientation, origin, and physical extent instead of requiring
            # identical voxel dimensions.
            ct_basis = np.asarray(ct_img.affine[:3, :3], dtype=float)
            mask_basis = np.asarray(mask_img.affine[:3, :3], dtype=float)
            ct_spacing = np.linalg.norm(ct_basis, axis=0)
            mask_spacing = np.linalg.norm(mask_basis, axis=0)
            if np.any(ct_spacing == 0) or np.any(mask_spacing == 0):
                raise LiveRoomError("CT and segmentation geometry do not match")
            if not np.allclose(ct_basis / ct_spacing, mask_basis / mask_spacing, atol=1e-4):
                raise LiveRoomError("CT and segmentation geometry do not match")
            if not np.allclose(ct_img.affine[:3, 3], mask_img.affine[:3, 3], atol=1e-3):
                raise LiveRoomError("CT and segmentation geometry do not match")
            ct_extent = (np.asarray(ct_img.shape[:3], dtype=float) - 1) * ct_spacing
            mask_extent = (np.asarray(mask_img.shape[:3], dtype=float) - 1) * mask_spacing
            extent_tolerance = float(max(np.max(ct_spacing), np.max(mask_spacing))) + 1e-3
            if not np.allclose(ct_extent, mask_extent, atol=extent_tolerance, rtol=0):
                raise LiveRoomError("CT and segmentation geometry do not match")
        geometry = {
            "shape": list(mask_img.shape[:3]),
            "affine": np.asarray(mask_img.affine).round(6).tolist(),
        }
        geometry_hash = hashlib.sha256(
            json.dumps(geometry, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return CaseFiles(
            ct_path=ct_path,
            mask_path=mask_path,
            geometry_hash=geometry_hash,
            dimensions=tuple(int(v) for v in mask_img.shape[:3]),
        )

    def _create_room(
        self,
        case_id: str,
        resolution: str = "low",
        *,
        mode: str = "review",
        quiz_pack_id: str | None = None,
        quiz_playlist_id: str | None = None,
        quiz_playlist_seed: str | None = None,
        quiz_exclude_pack_ids: Iterable[str] = (),
        quiz_timer_seconds: int | None = 30,
    ) -> tuple[dict[str, Any], str, str | None]:
        if mode not in {"review", "quiz"}:
            raise LiveRoomError("mode must be 'review' or 'quiz'")
        host_claim: str | None = None
        pack: dict[str, Any] | None = None
        if mode == "quiz":
            if bool(quiz_pack_id) == bool(quiz_playlist_id):
                raise LiveRoomError("Provide exactly one of quiz_pack_id or quiz_playlist_id")
            try:
                if quiz_playlist_id:
                    pack = self.quiz_registry.select(
                        quiz_playlist_id,
                        seed=str(quiz_playlist_seed or "default"),
                        exclude_pack_ids=quiz_exclude_pack_ids,
                    )
                else:
                    pack = self.quiz_registry.get(str(quiz_pack_id))
            except QuizPackError as exc:
                raise LiveRoomError(str(exc)) from exc
            if quiz_timer_seconds not in ALLOWED_TIMERS:
                raise LiveRoomError("Quiz timer must be untimed, 15, 30, or 60 seconds")
            if not quiz_playlist_id and case_id:
                try:
                    requested_case_id = str(int(case_id))
                except (TypeError, ValueError) as exc:
                    raise LiveRoomError("case_id must be numeric") from exc
                if requested_case_id != str(int(pack["case_id"])):
                    raise LiveRoomError("Quiz pack does not match the requested case")
            case_id = str(pack["case_id"])
            quiz_pack_id = str(pack["pack_id"])
            host_claim = secrets.token_urlsafe(32)
        case_files = self._resolve_case(str(case_id), resolution)
        room_id = str(uuid.uuid4())
        room_key = secrets.token_urlsafe(32)
        created = self._now()
        metadata = {
            "room_id": room_id,
            "key_hash": hash_room_key(room_key),
            "case_id": str(int(case_id)),
            "resolution": resolution,
            "mode": mode,
            "created_at": isoformat(created),
            "expires_at": isoformat(created + ROOM_TTL),
            "geometry_hash": case_files.geometry_hash,
            "dimensions": list(case_files.dimensions),
            "base_ct_path": str(case_files.ct_path),
            "base_mask_path": str(case_files.mask_path),
            "latest_seq": 0,
            "mask_snapshot_seq": 0,
            "mask_events_since_snapshot": 0,
            "mask_values_seq": 0,
            "last_snapshot_at": isoformat(created),
            "recent_event_ids": [],
        }
        if mode == "quiz":
            metadata.update({
                "quiz_pack_id": quiz_pack_id,
                "quiz_pack_version": pack["version"] if pack else None,
                "quiz_playlist_id": quiz_playlist_id,
                "quiz_timer_seconds": quiz_timer_seconds,
                "quiz_host_claim_hash": hash_room_key(host_claim or ""),
                "quiz_host_participant_id": None,
                "quiz_host_lease_id": None,
            })
        state = {
            "measurements": {},
            "notes": {},
            "chat": [],
            "undone_event_ids": [],
        }
        with self._locked(room_id, require_exists=False) as room_dir:
            _atomic_json(room_dir / "state.json", state)
            (room_dir / "events.jsonl").touch(mode=0o600)
            (room_dir / "events.jsonl").chmod(0o600)
            if mode == "quiz":
                if pack is None:  # pragma: no cover - resolved above
                    raise LiveRoomError("Quiz pack could not be resolved")
                pack_path = room_dir / "quiz_pack.json"
                _atomic_json(pack_path, pack)
                pack_path.chmod(0o600)
                quiz = {
                    "phase": "lobby",
                    "question_index": -1,
                    "question_count": len(pack["questions"]),
                    "eligible": [],
                    "eligible_history": {},
                    "question_elapsed_ms": {},
                    "response_count": 0,
                    "deadline_at": None,
                    "remaining_seconds": quiz_timer_seconds,
                    "timer_paused": False,
                    "elapsed_ms": 0,
                    "resumed_at": None,
                    "reveal": None,
                    "reveals": {},
                    "leaderboard": [],
                    "consistency_summary": {"consistent": 0, "inconsistent": 0, "incomplete": 0},
                    "round_completed": False,
                    "host_connected": False,
                    "revision": 0,
                }
                _atomic_json(room_dir / "quiz.json", quiz)
                answers_path = room_dir / "quiz_answers.json"
                _atomic_json(answers_path, {"submissions": {}})
                answers_path.chmod(0o600)
            # Metadata is readiness marker. Requests cannot observe partially
            # initialized room if process stops during multi-file creation.
            _atomic_json(room_dir / "metadata.json", metadata)
        return self.public_metadata(metadata), room_key, host_claim

    def create_room(self, case_id: str, resolution: str = "low") -> tuple[dict[str, Any], str]:
        metadata, room_key, _ = self._create_room(case_id, resolution, mode="review")
        return metadata, room_key

    def create_quiz_room(
        self,
        case_id: str,
        resolution: str = "low",
        *,
        quiz_pack_id: str = QUIZ_PACK_ID,
        quiz_playlist_id: str | None = None,
        quiz_playlist_seed: str | None = None,
        quiz_exclude_pack_ids: Iterable[str] = (),
        quiz_timer_seconds: int | None = 30,
    ) -> tuple[dict[str, Any], str, str]:
        metadata, room_key, host_claim = self._create_room(
            case_id,
            resolution,
            mode="quiz",
            quiz_pack_id=None if quiz_playlist_id else quiz_pack_id,
            quiz_playlist_id=quiz_playlist_id,
            quiz_playlist_seed=quiz_playlist_seed,
            quiz_exclude_pack_ids=quiz_exclude_pack_ids,
            quiz_timer_seconds=quiz_timer_seconds,
        )
        if host_claim is None:  # pragma: no cover - guarded by mode above
            raise LiveRoomError("Quiz host claim could not be created")
        return metadata, room_key, host_claim

    def verify_key(self, room_id: str, room_key: str, *, allow_expired: bool = False) -> dict[str, Any]:
        if not isinstance(room_key, str) or not room_key:
            raise RoomUnauthorized("Room key required")
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir, allow_expired=allow_expired)
            supplied = hash_room_key(room_key)
            if not hmac.compare_digest(supplied, metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            return self.public_metadata(metadata)

    @staticmethod
    def _public_quiz_state(quiz: dict[str, Any]) -> dict[str, Any]:
        public = {
            "revision": int(quiz.get("revision", 0)),
            "phase": quiz.get("phase", "lobby"),
            "question_index": int(quiz.get("question_index", -1)),
            "question_count": int(quiz.get("question_count", 0)),
            "deadline_at": quiz.get("deadline_at"),
            "remaining_seconds": quiz.get("remaining_seconds"),
            "timer_paused": bool(quiz.get("timer_paused", False)),
            "response_count": int(quiz.get("response_count", 0)),
            "eligible_count": len(quiz.get("eligible") or []),
            "reveal": quiz.get("reveal"),
            "leaderboard": quiz.get("leaderboard") or [],
            "consistency_summary": quiz.get("consistency_summary") or {
                "consistent": 0,
                "inconsistent": 0,
                "incomplete": 0,
            },
            "round_completed": bool(quiz.get("round_completed", False)),
            "host_connected": bool(quiz.get("host_connected", False)),
        }
        index = public["question_index"]
        if 0 <= index < public["question_count"]:
            public["current_question"] = quiz.get("current_question")
        else:
            public["current_question"] = None
        return public

    def _load_quiz_locked(self, room_dir: Path) -> dict[str, Any]:
        quiz = self._read_json(room_dir / "quiz.json")
        if "revision" not in quiz:
            # Rooms survive rolling deploys for 24 hours. Persist the baseline so
            # all subsequent mutations use the same monotonic counter.
            quiz["revision"] = 0
            _atomic_json(room_dir / "quiz.json", quiz)
        revision = quiz.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise LiveRoomError("Quiz revision is invalid")
        return quiz

    @staticmethod
    def _bump_quiz_revision(quiz: dict[str, Any]) -> None:
        quiz["revision"] = int(quiz.get("revision", 0)) + 1

    def _pack_locked(self, room_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        path = room_dir / "quiz_pack.json"
        if path.is_file():
            return self._read_json(path)
        # Backward compatibility for rooms created before frozen pack snapshots.
        try:
            return self.quiz_registry.get(str(metadata.get("quiz_pack_id", QUIZ_PACK_ID)))
        except QuizPackError as exc:
            raise LiveRoomError(str(exc)) from exc

    @staticmethod
    def _quiz_elapsed_ms(quiz: dict[str, Any], now: datetime) -> int:
        elapsed = int(quiz.get("elapsed_ms", 0))
        resumed_at = quiz.get("resumed_at")
        if resumed_at and not quiz.get("timer_paused"):
            elapsed += max(0, int((now - parse_time(resumed_at)).total_seconds() * 1000))
        return elapsed

    @staticmethod
    def _authorize_locked(metadata: dict[str, Any], room_key: str) -> None:
        if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
            raise RoomUnauthorized("Invalid room key")

    @staticmethod
    def _require_quiz(metadata: dict[str, Any]) -> None:
        if metadata.get("mode", "review") != "quiz":
            raise LiveRoomError("This Live Room is not a quiz")

    @staticmethod
    def _require_host(metadata: dict[str, Any], participant_id: str, lease_id: str) -> None:
        identities = metadata.get("participant_credentials") or {}
        record = identities.get(participant_id) if isinstance(identities, dict) else None
        if (
            not lease_id
            or metadata.get("quiz_host_participant_id") != participant_id
            or metadata.get("quiz_host_lease_id") != lease_id
            or not isinstance(record, dict)
            or not record.get("quiz_host")
        ):
            error = RoomUnauthorized("This connection is not the active quiz host")
            error.code = "invalid_host_lease"
            raise error

    def _event_ids_locked(self, room_dir: Path, metadata: dict[str, Any]) -> set[str]:
        room_id = str(metadata["room_id"])
        latest_seq = int(metadata.get("latest_seq", 0))
        cached = self._event_id_cache.get(room_id)
        if cached and cached[0] == latest_seq:
            return cached[1]
        event_ids = {
            str(event["event_id"])
            for event in self._iter_events(room_dir)
            if isinstance(event.get("event_id"), str)
        }
        self._event_id_cache[room_id] = (latest_seq, event_ids)
        return event_ids

    def _new_event_id_locked(self, room_dir: Path, metadata: dict[str, Any]) -> str:
        event_ids = self._event_ids_locked(room_dir, metadata)
        while True:
            event_id = str(uuid.uuid4())
            if event_id not in event_ids:
                return event_id

    @staticmethod
    def _room_file_bytes(room_dir: Path) -> int:
        return sum(path.stat().st_size for path in room_dir.iterdir() if path.is_file())

    def _ensure_event_capacity_locked(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        serialized_bytes: int = 0,
        *,
        additional_room_bytes: int = 0,
        lifecycle: bool = False,
    ) -> None:
        event_path = room_dir / "events.jsonl"
        event_bytes = event_path.stat().st_size
        event_limit = MAX_EVENTS if lifecycle else max(0, MAX_EVENTS - LIFECYCLE_EVENT_RESERVE)
        event_byte_limit = (
            MAX_EVENT_LOG_BYTES
            if lifecycle
            else max(0, MAX_EVENT_LOG_BYTES - LIFECYCLE_EVENT_BYTE_RESERVE)
        )
        room_byte_limit = (
            MAX_ROOM_BYTES
            if lifecycle
            else max(0, MAX_ROOM_BYTES - LIFECYCLE_EVENT_BYTE_RESERVE)
        )
        if (
            int(metadata.get("latest_seq", 0)) >= event_limit
            or event_bytes + serialized_bytes + 1 > event_byte_limit
            or self._room_file_bytes(room_dir) + serialized_bytes + additional_room_bytes + 1 > room_byte_limit
        ):
            raise RoomFull("Room event log is full; export remains available")

    def _append_event_locked(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        event: dict[str, Any],
        *,
        derived: dict[str, dict[str, Any]],
        lifecycle: bool = False,
        rebuild_mask: bool = False,
        after_append: Callable[[], None] | None = None,
    ) -> None:
        event_id = str(event.get("event_id", ""))
        event_ids = self._event_ids_locked(room_dir, metadata)
        if not event_id or event_id in event_ids:
            raise LiveRoomError("event_id must be unique for the lifetime of the room")
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        serialized_bytes = len(serialized.encode("utf-8"))
        final_metadata = copy.deepcopy(metadata)
        final_metadata["latest_seq"] = int(event["seq"])
        recent = list(final_metadata.get("recent_event_ids", [])) + [event_id]
        final_metadata["recent_event_ids"] = recent[-10_000:]
        pending = {
            "event": event,
            "metadata": final_metadata,
            "derived": derived,
            "rebuild_mask": rebuild_mask,
        }
        pending_bytes = len(json.dumps(pending, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        derived_growth = sum(
            max(
                0,
                len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                - ((room_dir / filename).stat().st_size if (room_dir / filename).exists() else 0),
            )
            for filename, value in derived.items()
        )
        self._ensure_event_capacity_locked(
            room_dir,
            metadata,
            serialized_bytes,
            additional_room_bytes=pending_bytes + derived_growth,
            lifecycle=lifecycle,
        )
        _atomic_json(room_dir / ".pending_event.json", pending)
        event_path = room_dir / "events.jsonl"
        with event_path.open("a", encoding="utf-8") as stream:
            os.chmod(event_path, 0o600)
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        for filename, value in derived.items():
            _atomic_json(room_dir / filename, value)
        if after_append:
            after_append()
        _atomic_json(room_dir / "metadata.json", final_metadata)
        (room_dir / ".pending_event.json").unlink(missing_ok=True)
        directory_fd = os.open(room_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        metadata.clear()
        metadata.update(final_metadata)
        event_ids.add(event_id)
        self._event_id_cache[str(metadata["room_id"])] = (int(event["seq"]), event_ids)
        self._invalidate_exports(room_dir)

    def _append_quiz_event_locked(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        quiz: dict[str, Any],
        event_type: str,
        *,
        additional_derived: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if event_type not in QUIZ_EVENT_TYPES:
            raise LiveRoomError("Unsupported quiz event type")
        self._bump_quiz_revision(quiz)
        seq = int(metadata.get("latest_seq", 0)) + 1
        event = {
            "seq": seq,
            "event_id": self._new_event_id_locked(room_dir, metadata),
            "type": event_type,
            "participant_id": "server",
            "name": "Quiz server",
            "created_at": isoformat(self._now()),
            "payload": {"quiz": self._public_quiz_state(quiz)},
        }
        derived = {"quiz.json": quiz, **(additional_derived or {})}
        self._append_event_locked(
            room_dir,
            metadata,
            event,
            derived=derived,
            lifecycle=event_type in {
                "quiz.host_paused",
                "quiz.host_resumed",
                "quiz.host_promoted",
            },
        )
        return event

    def _reconcile_quiz_response_count_locked(
        self,
        room_dir: Path,
        quiz: dict[str, Any],
        answers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question_id = (quiz.get("current_question") or {}).get("id")
        if not question_id:
            return answers or {"submissions": {}}
        answers = answers or self._read_json(room_dir / "quiz_answers.json")
        response_count = len((answers.get("submissions") or {}).get(question_id, {}))
        if int(quiz.get("response_count", 0)) != response_count:
            quiz["response_count"] = response_count
            self._bump_quiz_revision(quiz)
            _atomic_json(room_dir / "quiz.json", quiz)
        return answers

    def quiz_context(
        self,
        room_id: str,
        room_key: str,
        participant_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            if metadata.get("mode", "review") != "quiz":
                return None
            quiz = self._load_quiz_locked(room_dir)
            answers = self._read_json(room_dir / "quiz_answers.json")
            self._reconcile_quiz_response_count_locked(room_dir, quiz, answers)
            own = {}
            if participant_id:
                for question_id, submissions in (answers.get("submissions") or {}).items():
                    if participant_id in submissions:
                        own[question_id] = dict(submissions[participant_id])
            eligible = participant_id in {
                item.get("participant_id") for item in (quiz.get("eligible") or [])
            }
            return {
                "state": self._public_quiz_state(quiz),
                "own_submissions": own,
                "eligible": eligible,
            }

    def quiz_host_assigned(self, room_id: str, room_key: str) -> bool:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            return bool(metadata.get("quiz_host_participant_id"))

    def connect_quiz_host(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        lease_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_quiz(metadata)
            identities = metadata.get("participant_credentials") or {}
            record = identities.get(participant_id) if isinstance(identities, dict) else None
            if (
                metadata.get("quiz_host_participant_id") != participant_id
                or not isinstance(record, dict)
                or not record.get("quiz_host")
                or not lease_id
            ):
                error = RoomUnauthorized("Participant does not own quiz host authority")
                error.code = "invalid_host_lease"
                raise error
            metadata["quiz_host_lease_id"] = lease_id
            quiz = self._load_quiz_locked(room_dir)
            event = None
            was_connected = bool(quiz.get("host_connected"))
            quiz["host_connected"] = True
            resumed_timer = quiz.get("phase") == "question_open" and quiz.get("timer_paused")
            if resumed_timer:
                now = self._now()
                remaining = quiz.get("remaining_seconds")
                quiz["timer_paused"] = False
                quiz["resumed_at"] = isoformat(now)
                quiz["deadline_at"] = (
                    isoformat(now + timedelta(seconds=float(remaining)))
                    if remaining is not None else None
                )
                try:
                    event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.host_resumed")
                except RoomFull:
                    _atomic_json(room_dir / "quiz.json", quiz)
                    _atomic_json(room_dir / "metadata.json", metadata)
                    logger.warning("Resumed host without event because room is full room=%s", room_id)
            if event is None:
                if not was_connected and not resumed_timer:
                    self._bump_quiz_revision(quiz)
                _atomic_json(room_dir / "quiz.json", quiz)
                _atomic_json(room_dir / "metadata.json", metadata)
            return self._public_quiz_state(quiz), event

    def disconnect_quiz_host(
        self,
        room_id: str,
        participant_id: str,
        lease_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if (
                metadata.get("mode", "review") != "quiz"
                or metadata.get("quiz_host_participant_id") != participant_id
                or metadata.get("quiz_host_lease_id") != lease_id
            ):
                return None
            quiz = self._load_quiz_locked(room_dir)
            quiz["host_connected"] = False
            metadata["quiz_host_lease_id"] = None
            if quiz.get("phase") == "question_open" and not quiz.get("timer_paused"):
                now = self._now()
                quiz["elapsed_ms"] = self._quiz_elapsed_ms(quiz, now)
                deadline = quiz.get("deadline_at")
                quiz["remaining_seconds"] = (
                    max(0.0, (parse_time(deadline) - now).total_seconds()) if deadline else None
                )
                quiz["timer_paused"] = True
                quiz["deadline_at"] = None
                quiz["resumed_at"] = None
            try:
                event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.host_paused")
            except RoomFull:
                event = None
                _atomic_json(room_dir / "quiz.json", quiz)
                _atomic_json(room_dir / "metadata.json", metadata)
                logger.warning("Paused host without event because room is full room=%s", room_id)
            return self._public_quiz_state(quiz), event

    def reconcile_stale_host_leases(self) -> int:
        """Pause quiz hosts left connected by a previous websocket process."""
        reconciled = 0
        for child in list(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                room_id = self.validate_room_id(child.name)
                with self._locked(room_id) as room_dir:
                    metadata = self._load_metadata(room_dir)
                    if metadata.get("mode") != "quiz":
                        continue
                    quiz = self._load_quiz_locked(room_dir)
                    if not metadata.get("quiz_host_lease_id") and not quiz.get("host_connected"):
                        continue
                    metadata["quiz_host_lease_id"] = None
                    quiz["host_connected"] = False
                    if quiz.get("phase") == "question_open" and not quiz.get("timer_paused"):
                        now = self._now()
                        quiz["elapsed_ms"] = self._quiz_elapsed_ms(quiz, now)
                        deadline = quiz.get("deadline_at")
                        quiz["remaining_seconds"] = (
                            max(0.0, (parse_time(deadline) - now).total_seconds()) if deadline else None
                        )
                        quiz["timer_paused"] = True
                        quiz["deadline_at"] = None
                        quiz["resumed_at"] = None
                    try:
                        self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.host_paused")
                    except RoomFull:
                        _atomic_json(room_dir / "quiz.json", quiz)
                        _atomic_json(room_dir / "metadata.json", metadata)
                        logger.warning("Cleared stale host lease without event because room is full room=%s", room_id)
                    reconciled += 1
            except (LiveRoomError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Could not reconcile stale Live Room host room=%s error=%s", child.name, exc)
        return reconciled

    def promote_quiz_host(
        self,
        room_id: str,
        participant_id: str,
        lease_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._require_quiz(metadata)
            quiz = self._load_quiz_locked(room_dir)
            if quiz.get("host_connected"):
                raise LiveRoomError("Quiz host is already connected")
            identities = metadata.get("participant_credentials") or {}
            candidate = identities.get(participant_id) if isinstance(identities, dict) else None
            if not isinstance(candidate, dict) or not lease_id:
                raise ParticipantUnauthorized("Promotion candidate identity is invalid")
            previous_host = metadata.get("quiz_host_participant_id")
            if previous_host and isinstance(identities.get(previous_host), dict):
                identities[previous_host]["quiz_host"] = False
            candidate["quiz_host"] = True
            metadata["quiz_host_participant_id"] = participant_id
            metadata["quiz_host_lease_id"] = lease_id
            quiz["host_connected"] = True
            if quiz.get("phase") == "question_open" and quiz.get("timer_paused"):
                now = self._now()
                remaining = quiz.get("remaining_seconds")
                quiz["timer_paused"] = False
                quiz["resumed_at"] = isoformat(now)
                quiz["deadline_at"] = (
                    isoformat(now + timedelta(seconds=float(remaining)))
                    if remaining is not None else None
                )
            try:
                event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.host_promoted")
            except RoomFull:
                # Authority must remain recoverable even after all reserved log
                # capacity is exhausted. Persist the identity/lease transition;
                # clients also receive presence and quiz-state broadcasts.
                event = None
                _atomic_json(room_dir / "quiz.json", quiz)
                _atomic_json(room_dir / "metadata.json", metadata)
                logger.warning("Promoted host without event because room is full room=%s", room_id)
            return self._public_quiz_state(quiz), event, previous_host

    def rollback_quiz_host_promotion(
        self,
        room_id: str,
        participant_id: str,
        lease_id: str,
        previous_host: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        """Rollback only the exact undelivered promotion lease."""
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if (
                metadata.get("quiz_host_participant_id") != participant_id
                or metadata.get("quiz_host_lease_id") != lease_id
            ):
                return None
            identities = metadata.get("participant_credentials") or {}
            if isinstance(identities.get(participant_id), dict):
                identities[participant_id]["quiz_host"] = False
            if previous_host and isinstance(identities.get(previous_host), dict):
                identities[previous_host]["quiz_host"] = True
            metadata["quiz_host_participant_id"] = previous_host
            metadata["quiz_host_lease_id"] = None
            quiz = self._load_quiz_locked(room_dir)
            quiz["host_connected"] = False
            if quiz.get("phase") == "question_open" and not quiz.get("timer_paused"):
                now = self._now()
                quiz["elapsed_ms"] = self._quiz_elapsed_ms(quiz, now)
                deadline = quiz.get("deadline_at")
                quiz["remaining_seconds"] = (
                    max(0.0, (parse_time(deadline) - now).total_seconds()) if deadline else None
                )
                quiz["timer_paused"] = True
                quiz["deadline_at"] = None
                quiz["resumed_at"] = None
            try:
                event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.host_paused")
            except RoomFull:
                event = None
                _atomic_json(room_dir / "quiz.json", quiz)
                _atomic_json(room_dir / "metadata.json", metadata)
                logger.warning("Rolled back host promotion without event because room is full room=%s", room_id)
            return self._public_quiz_state(quiz), event

    def _open_question_locked(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        quiz: dict[str, Any],
        question_index: int,
        participants: list[dict[str, str]],
    ) -> dict[str, Any]:
        pack = self._pack_locked(room_dir, metadata)
        questions = pack["questions"]
        if not 0 <= question_index < len(questions):
            raise LiveRoomError("Question index is out of range")
        now = self._now()
        eligible = [
            {"participant_id": str(item["participant_id"]), "name": _clean_text(item["name"], MAX_NAME)}
            for item in participants
        ]
        question_id = questions[question_index]["id"]
        timer = metadata.get("quiz_timer_seconds")
        quiz.update({
            "phase": "question_open",
            "question_index": question_index,
            "eligible": eligible,
            "response_count": 0,
            "remaining_seconds": timer,
            "timer_paused": False,
            "elapsed_ms": 0,
            "resumed_at": isoformat(now),
            "deadline_at": isoformat(now + timedelta(seconds=timer)) if timer is not None else None,
            "reveal": None,
            "current_question": public_question(pack, question_index, choice_seed=metadata["room_id"]),
        })
        quiz.setdefault("eligible_history", {})[question_id] = eligible
        quiz.setdefault("question_elapsed_ms", {})
        return quiz

    def start_quiz(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        lease_id: str,
        participants: list[dict[str, str]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_quiz(metadata)
            self._require_host(metadata, participant_id, lease_id)
            quiz = self._load_quiz_locked(room_dir)
            if quiz.get("phase") != "lobby" or quiz.get("round_completed"):
                raise LiveRoomError("This quiz round has already started")
            self._open_question_locked(room_dir, metadata, quiz, 0, participants)
            event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.started")
            pack = self._pack_locked(room_dir, metadata)
            self.quiz_telemetry.record(
                "pack_started",
                pack_id=pack["pack_id"],
                case_id=pack["case_id"],
                mode="race",
            )
            return self._public_quiz_state(quiz), event

    def _close_question_locked(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        quiz: dict[str, Any],
    ) -> dict[str, Any]:
        if quiz.get("phase") != "question_open":
            raise LiveRoomError("No quiz question is open")
        now = self._now()
        elapsed = self._quiz_elapsed_ms(quiz, now)
        pack = self._pack_locked(room_dir, metadata)
        question_id = pack["questions"][int(quiz["question_index"])]["id"]
        quiz.setdefault("question_elapsed_ms", {})[question_id] = elapsed
        quiz.update({
            "phase": "question_closed",
            "deadline_at": None,
            "remaining_seconds": 0,
            "timer_paused": False,
            "elapsed_ms": elapsed,
            "resumed_at": None,
        })
        return quiz

    def answer_quiz(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        name: str,
        choice_id: str,
        connected_participant_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_quiz(metadata)
            quiz = self._load_quiz_locked(room_dir)
            if quiz.get("phase") != "question_open" or quiz.get("timer_paused"):
                raise LiveRoomError("This question is not accepting answers")
            now = self._now()
            deadline = quiz.get("deadline_at")
            if deadline and now >= parse_time(deadline):
                raise LiveRoomError("The question deadline has passed")
            eligible_ids = {item["participant_id"] for item in quiz.get("eligible") or []}
            if participant_id not in eligible_ids:
                raise LiveRoomError("You become eligible on the next question")
            pack = self._pack_locked(room_dir, metadata)
            question_id = pack["questions"][int(quiz["question_index"])]["id"]
            if choice_id not in choice_ids(pack, question_id):
                raise LiveRoomError("Invalid answer choice")
            answers = self._read_json(room_dir / "quiz_answers.json")
            submissions = answers.setdefault("submissions", {}).setdefault(question_id, {})
            if participant_id in submissions:
                error = LiveRoomError("Your answer is already locked")
                error.code = "duplicate_quiz_answer"
                raise error
            submission = {
                "choice_id": choice_id,
                "answered_at": isoformat(now),
                "response_ms": self._quiz_elapsed_ms(quiz, now),
                "name": _clean_text(name, MAX_NAME),
            }
            submissions[participant_id] = submission
            quiz["response_count"] = len(submissions)
            active_eligible = eligible_ids & set(connected_participant_ids)
            event = None
            if active_eligible <= set(submissions):
                self._close_question_locked(room_dir, metadata, quiz)
                # The final answer and automatic close are one retryable commit.
                # Quota is checked before either derived file becomes visible.
                event = self._append_quiz_event_locked(
                    room_dir,
                    metadata,
                    quiz,
                    "quiz.closed",
                    additional_derived={"quiz_answers.json": answers},
                )
            else:
                self._bump_quiz_revision(quiz)
                _atomic_json(room_dir / "quiz_answers.json", answers)
                _atomic_json(room_dir / "quiz.json", quiz)
            return self._public_quiz_state(quiz), submission, event

    def close_quiz_question(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        lease_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_host(metadata, participant_id, lease_id)
            quiz = self._load_quiz_locked(room_dir)
            self._reconcile_quiz_response_count_locked(room_dir, quiz)
            self._close_question_locked(room_dir, metadata, quiz)
            event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.closed")
            return self._public_quiz_state(quiz), event

    def close_quiz_if_due(self, room_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if metadata.get("mode") != "quiz":
                return None
            quiz = self._load_quiz_locked(room_dir)
            self._reconcile_quiz_response_count_locked(room_dir, quiz)
            deadline = quiz.get("deadline_at")
            if quiz.get("phase") != "question_open" or quiz.get("timer_paused") or not deadline:
                return None
            if self._now() < parse_time(deadline):
                return None
            self._close_question_locked(room_dir, metadata, quiz)
            event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.closed")
            return self._public_quiz_state(quiz), event

    def close_quiz_if_all_answered(
        self,
        room_id: str,
        connected_participant_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if metadata.get("mode") != "quiz":
                return None
            quiz = self._load_quiz_locked(room_dir)
            self._reconcile_quiz_response_count_locked(room_dir, quiz)
            if quiz.get("phase") != "question_open":
                return None
            pack = self._pack_locked(room_dir, metadata)
            question_id = pack["questions"][int(quiz["question_index"])]["id"]
            submissions = (self._read_json(room_dir / "quiz_answers.json").get("submissions") or {}).get(question_id, {})
            eligible = {item["participant_id"] for item in quiz.get("eligible") or []}
            if not (eligible & connected_participant_ids) <= set(submissions):
                return None
            self._close_question_locked(room_dir, metadata, quiz)
            event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.closed")
            return self._public_quiz_state(quiz), event

    @staticmethod
    def _quiz_leaderboard(
        quiz: dict[str, Any],
        answers: dict[str, Any],
        pack: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        history = quiz.get("eligible_history") or {}
        people: dict[str, str] = {}
        for roster in history.values():
            for item in roster:
                people[item["participant_id"]] = item["name"]
        submissions = answers.get("submissions") or {}
        question_elapsed = quiz.get("question_elapsed_ms") or {}
        rows: list[dict[str, Any]] = []
        summary = {"consistent": 0, "inconsistent": 0, "incomplete": 0}
        for participant_id, name in people.items():
            selected: dict[str, str] = {}
            score = 0
            total_response_ms = 0
            for question in pack["questions"]:
                question_id = question["id"]
                roster_ids = {item["participant_id"] for item in history.get(question_id, [])}
                if participant_id not in roster_ids:
                    continue
                submission = (submissions.get(question_id) or {}).get(participant_id)
                if submission:
                    selected[question_id] = submission["choice_id"]
                    total_response_ms += int(submission.get("response_ms", 0))
                    if submission["choice_id"] == correct_choice(pack, question_id):
                        score += 1
                else:
                    total_response_ms += int(question_elapsed.get(question_id, 0))
            consistency = consistency_for(selected, pack)
            summary[consistency["status"]] += 1
            rows.append({
                "participant_id": participant_id,
                "name": name,
                "score": score,
                "max_score": len(pack["questions"]),
                "total_response_ms": total_response_ms,
                "consistency": consistency,
            })
        rows.sort(key=lambda item: (-item["score"], item["total_response_ms"], item["name"].casefold(), item["participant_id"]))
        for index, row in enumerate(rows, 1):
            row["rank"] = index
        return rows, summary

    def reveal_quiz_question(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        lease_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_host(metadata, participant_id, lease_id)
            quiz = self._load_quiz_locked(room_dir)
            self._reconcile_quiz_response_count_locked(room_dir, quiz)
            if quiz.get("phase") != "question_closed":
                raise LiveRoomError("Close the question before revealing it")
            question_index = int(quiz["question_index"])
            pack = self._pack_locked(room_dir, metadata)
            question_id = pack["questions"][question_index]["id"]
            answers = self._read_json(room_dir / "quiz_answers.json")
            submissions = (answers.get("submissions") or {}).get(question_id, {})
            revealed = reveal_for(pack, question_id, submissions)
            quiz["phase"] = "question_revealed"
            quiz["reveal"] = revealed
            quiz.setdefault("reveals", {})[question_id] = revealed
            leaderboard, summary = self._quiz_leaderboard(quiz, answers, pack)
            quiz["leaderboard"] = leaderboard
            quiz["consistency_summary"] = summary
            event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.revealed")
            self.quiz_telemetry.record(
                "answer_distribution",
                pack_id=pack["pack_id"],
                case_id=pack["case_id"],
                mode="race",
                payload={"question_id": question_id, "distribution": revealed["distribution"]},
            )
            return self._public_quiz_state(quiz), event

    def advance_quiz(
        self,
        room_id: str,
        room_key: str,
        participant_id: str,
        lease_id: str,
        participants: list[dict[str, str]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_host(metadata, participant_id, lease_id)
            quiz = self._load_quiz_locked(room_dir)
            pack = self._pack_locked(room_dir, metadata)
            if quiz.get("phase") != "question_revealed":
                raise LiveRoomError("Reveal the result before advancing")
            next_index = int(quiz["question_index"]) + 1
            if next_index >= len(pack["questions"]):
                quiz["phase"] = "completed"
                quiz["round_completed"] = True
                event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.advanced")
                self.quiz_telemetry.record(
                    "pack_completed",
                    pack_id=pack["pack_id"],
                    case_id=pack["case_id"],
                    mode="race",
                )
            else:
                self._open_question_locked(room_dir, metadata, quiz, next_index, participants)
                event = self._append_quiz_event_locked(room_dir, metadata, quiz, "quiz.advanced")
            return self._public_quiz_state(quiz), event

    def get_metadata(self, room_id: str, room_key: str) -> dict[str, Any]:
        return self.verify_key(room_id, room_key)

    def get_snapshot(self, room_id: str, room_key: str) -> dict[str, Any]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            state = self._read_json(room_dir / "state.json")
            snapshot = {
                "room": self.public_metadata(metadata),
                "latest_seq": metadata["latest_seq"],
                "state": state,
            }
            if metadata.get("mode") == "quiz":
                snapshot["quiz"] = self._public_quiz_state(self._load_quiz_locked(room_dir))
            return snapshot

    def events_after(self, room_id: str, room_key: str, sequence: int) -> list[dict[str, Any]]:
        events, resync_required = self.replay_after(room_id, room_key, sequence)
        if resync_required:
            raise LiveRoomError("Requested event replay exceeds the replay budget")
        return events

    def replay_after(
        self,
        room_id: str,
        room_key: str,
        sequence: int,
        *,
        budget: int = MAX_REPLAY_EVENTS,
        byte_budget: int = MAX_REPLAY_BYTES,
    ) -> tuple[list[dict[str, Any]], bool]:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            error = LiveRoomError("last_seq must be a non-negative integer")
            error.code = "invalid_last_seq"
            raise error
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            latest = int(metadata.get("latest_seq", 0))
            if sequence > latest:
                error = LiveRoomError("last_seq is ahead of the room sequence")
                error.code = "invalid_last_seq"
                raise error
            if latest - sequence > budget:
                return [], True
            events: list[dict[str, Any]] = []
            replay_bytes = 0
            path = room_dir / "events.jsonl"
            if not path.exists():
                return events, False
            with path.open("rb") as stream:
                event_index = 0
                for line in stream:
                    if not line.strip():
                        continue
                    event_index += 1
                    if event_index <= sequence:
                        continue
                    replay_bytes += len(line)
                    if replay_bytes > byte_budget:
                        return [], True
                    event = json.loads(line)
                    events.append(event)
            return events, False

    @staticmethod
    def _iter_events(room_dir: Path) -> Iterator[dict[str, Any]]:
        path = room_dir / "events.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as stream:
            pending: str | None = None
            for line in stream:
                if pending is not None and pending.strip():
                    yield json.loads(pending)
                pending = line
            if pending and pending.strip():
                try:
                    yield json.loads(pending)
                except json.JSONDecodeError:
                    # Power loss can leave only the final append incomplete. Earlier
                    # malformed lines still raise through the look-ahead branch above.
                    return

    @staticmethod
    def _last_event(room_dir: Path) -> dict[str, Any] | None:
        """Read only the tail event during normal metadata loads.

        Rooms can contain tens of thousands of events. Scanning the complete JSONL
        file on every commit made ordinary edits progressively slower even when no
        recovery was needed.
        """
        path = room_dir / "events.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            return None
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            buffer = b""
            while position > 0:
                size = min(64 * 1024, position)
                position -= size
                stream.seek(position)
                buffer = stream.read(size) + buffer
                lines = buffer.splitlines()
                if len(lines) < 2 and position > 0:
                    continue
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(value, dict):
                        return value
                if position == 0:
                    break
        return None

    @staticmethod
    def _find_event(room_dir: Path, event_id: str) -> dict[str, Any] | None:
        for event in LiveRoomStore._iter_events(room_dir):
            if event.get("event_id") == event_id:
                return event
        return None

    @staticmethod
    def _validate_point(point: Any) -> list[float]:
        if not isinstance(point, list) or len(point) != 3:
            raise LiveRoomError("World coordinate must contain three numbers")
        values = [float(value) for value in point]
        if not all(np.isfinite(values)):
            raise LiveRoomError("World coordinate contains a non-finite value")
        return values

    def _sanitize_measurement(self, raw: Any, current: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise LiveRoomError("measurement must be an object")
        measurement_id = str(raw.get("id", ""))
        if not measurement_id or len(measurement_id) > 128:
            raise LiveRoomError("Invalid measurement id")
        tool = str(raw.get("tool", ""))
        if tool not in MEASUREMENT_TOOLS:
            raise LiveRoomError("Unsupported measurement tool")
        points_raw = raw.get("points") or []
        polyline_raw = raw.get("polyline") or []
        if len(points_raw) > 64 or len(polyline_raw) > 20_000:
            raise LiveRoomError("Measurement contains too many points")
        points = [self._validate_point(point) for point in points_raw]
        polyline = [self._validate_point(point) for point in polyline_raw]
        return {
            "id": measurement_id,
            "tool": tool,
            "points": points,
            "polyline": polyline,
            "text": _clean_text(raw.get("text", ""), MAX_TEXT_NOTE, required=False),
            "label": _clean_text(raw.get("label", ""), 256, required=False),
            "frame_of_reference": str(raw.get("frame_of_reference", ""))[:256],
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            "revision": int(current.get("revision", 0) + 1) if current else 1,
        }

    def _mask_values_at_seq_locked(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        through_seq: int,
    ) -> np.ndarray:
        image = nib.load(str(metadata["base_mask_path"]))
        data = np.array(np.asanyarray(image.dataobj), dtype=np.uint8, order="F")
        flat = data.reshape(-1, order="F")
        for event in self._iter_events(room_dir):
            if int(event.get("seq", 0)) > through_seq:
                break
            if event.get("type") != "mask.patch":
                continue
            for item in (event.get("payload") or {}).get("ranges", []):
                start = int(item["start"])
                flat[start : start + int(item["length"])] = int(item["after"])
        return flat

    def _mask_value_map(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
    ) -> np.memmap:
        nvoxels = int(np.prod(metadata["dimensions"], dtype=np.int64))
        path = room_dir / "mask_values.bin"
        latest_seq = int(metadata.get("latest_seq", 0))
        cache_seq = int(metadata.get("mask_values_seq", -1))
        if not path.exists() or path.stat().st_size != nvoxels or cache_seq != latest_seq:
            values = self._mask_values_at_seq_locked(room_dir, metadata, latest_seq)
            temporary = room_dir / ".mask_values.tmp.bin"
            output = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=(nvoxels,))
            output[:] = values
            output.flush()
            del output
            os.replace(temporary, path)
            path.chmod(0o600)
            metadata["mask_values_seq"] = latest_seq
        return np.memmap(path, dtype=np.uint8, mode="r+", shape=(nvoxels,))

    def _validate_mask_patch(
        self,
        room_dir: Path,
        payload: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LiveRoomError("mask patch must be an object")
        if payload.get("geometry_hash") != metadata["geometry_hash"]:
            raise LiveRoomError("Mask geometry hash does not match this room")
        if payload.get("resolution") != metadata["resolution"]:
            raise LiveRoomError("Mask patch resolution does not match this room")
        segment_label = int(payload.get("segment_label", -1))
        if not 0 <= segment_label <= 255:
            raise LiveRoomError("Invalid segment label")
        ranges_raw = payload.get("ranges")
        if not isinstance(ranges_raw, list) or not ranges_raw:
            raise LiveRoomError("Mask patch ranges are required")
        if len(ranges_raw) > MAX_MASK_RANGES_PER_EVENT:
            raise LiveRoomError("Mask patch contains too many ranges")
        nvoxels = int(np.prod(metadata["dimensions"], dtype=np.int64))
        changed = 0
        ranges: list[dict[str, int]] = []
        previous_end = -1
        values = self._mask_value_map(room_dir, metadata)
        for raw in ranges_raw:
            if not isinstance(raw, dict):
                raise LiveRoomError("Invalid mask range")
            start = int(raw.get("start", -1))
            length = int(raw.get("length", 0))
            after = int(raw.get("after", -1))
            if start < 0 or length <= 0 or start + length > nvoxels:
                raise LiveRoomError("Mask range is out of bounds")
            if start < previous_end:
                raise LiveRoomError("Mask ranges must be sorted and non-overlapping")
            if not 0 <= after <= 255:
                raise LiveRoomError("Mask labels must be uint8 values")
            changed += length
            if changed > MAX_MASK_VOXELS_PER_EVENT:
                raise LiveRoomError("Mask patch is too large")
            previous_end = start + length
            current = np.asarray(values[start : start + length], dtype=np.uint8)
            boundaries = np.concatenate((
                np.array([0], dtype=np.int64),
                np.flatnonzero(np.diff(current)) + 1,
                np.array([length], dtype=np.int64),
            ))
            for run_start, run_end in zip(boundaries[:-1], boundaries[1:]):
                authoritative_before = int(current[int(run_start)])
                if authoritative_before == after:
                    continue
                ranges.append({
                    "start": start + int(run_start),
                    "length": int(run_end - run_start),
                    "before": authoritative_before,
                    "after": after,
                })
                if len(ranges) > MAX_MASK_RANGES_PER_EVENT:
                    del values
                    raise LiveRoomError("Mask patch is too fragmented after conflict resolution")
        del values
        return {
            "operation_id": str(payload.get("operation_id", ""))[:128],
            "geometry_hash": metadata["geometry_hash"],
            "resolution": metadata["resolution"],
            "segment_label": segment_label,
            "ranges": ranges,
        }

    @staticmethod
    def _writer_map(room_dir: Path, nvoxels: int) -> np.memmap:
        path = room_dir / "mask_writers.bin"
        expected32 = nvoxels * np.dtype(np.uint32).itemsize
        expected64 = nvoxels * np.dtype(np.uint64).itemsize
        if path.exists() and path.stat().st_size == expected64:
            # Keep existing rooms readable across this optimization. New rooms use
            # uint32 because sequence numbers are capped at 50,000.
            return np.memmap(path, dtype=np.uint64, mode="r+", shape=(nvoxels,))
        if not path.exists() or path.stat().st_size != expected32:
            with path.open("wb") as stream:
                stream.truncate(expected32)
            path.chmod(0o600)
        return np.memmap(path, dtype=np.uint32, mode="r+", shape=(nvoxels,))

    def _apply_event(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        state: dict[str, Any],
        event_type: str,
        payload: dict[str, Any],
        seq: int,
        name: str,
    ) -> tuple[dict[str, Any], Any]:
        before: Any = None
        if event_type == "measurement.upsert":
            raw = payload.get("measurement")
            measurement_id = str(raw.get("id", "")) if isinstance(raw, dict) else ""
            before = state["measurements"].get(measurement_id)
            measurement = self._sanitize_measurement(raw, before)
            measurement["_seq"] = seq
            state["measurements"][measurement["id"]] = measurement
            payload = {"measurement": measurement}
        elif event_type == "measurement.delete":
            measurement_id = str(payload.get("id", ""))
            before = state["measurements"].pop(measurement_id, None)
            if not measurement_id:
                raise LiveRoomError("Measurement id required")
            payload = {"id": measurement_id}
        elif event_type == "note.upsert":
            raw = payload.get("note")
            if not isinstance(raw, dict):
                raise LiveRoomError("note must be an object")
            note_id = str(raw.get("id", ""))
            if not note_id or len(note_id) > 128:
                raise LiveRoomError("Invalid note id")
            before = state["notes"].get(note_id)
            note = {
                "id": note_id,
                "author": name,
                "text": _clean_text(raw.get("text", ""), MAX_TEXT_NOTE),
                "world": self._validate_point(raw.get("world")),
                "plane": str(raw.get("plane", "axial"))[:32],
                "organ_label": _clean_text(raw.get("organ_label", ""), 128, required=False),
                "revision": int(before.get("revision", 0) + 1) if before else 1,
                "_seq": seq,
            }
            state["notes"][note_id] = note
            payload = {"note": note}
        elif event_type == "note.delete":
            note_id = str(payload.get("id", ""))
            before = state["notes"].pop(note_id, None)
            if not note_id:
                raise LiveRoomError("Note id required")
            payload = {"id": note_id}
        elif event_type == "chat.add":
            raw = payload.get("message")
            if not isinstance(raw, dict):
                raise LiveRoomError("message must be an object")
            message_id = str(raw.get("id", ""))
            if not message_id or len(message_id) > 128:
                raise LiveRoomError("Invalid chat message id")
            if any(item.get("id") == message_id for item in state["chat"]):
                raise LiveRoomError("Duplicate chat message id")
            message = {
                "id": message_id,
                "author": name,
                "text": _clean_text(raw.get("text", ""), MAX_TEXT_CHAT),
                "timestamp": isoformat(self._now()),
                "_seq": seq,
            }
            state["chat"].append(message)
            state["chat"] = state["chat"][-MAX_CHAT_MESSAGES:]
            payload = {"message": message}
        elif event_type == "mask.patch":
            payload = self._validate_mask_patch(room_dir, payload, metadata)
        elif event_type == "interactive.accept":
            accept_id = str(payload.get("accept_id", ""))
            chunk_index, chunk_count = payload.get("chunk_index"), payload.get("chunk_count")
            data_chunk = payload.get("data_chunk", "")
            bbox, shape = payload.get("bbox"), payload.get("shape")
            dimensions = metadata.get("dimensions") or []
            if (not accept_id or len(accept_id) > 128
                    or not isinstance(chunk_index, int) or isinstance(chunk_index, bool)
                    or not isinstance(chunk_count, int) or isinstance(chunk_count, bool)
                    or chunk_count < 1 or chunk_count > 4096 or chunk_index < 0 or chunk_index >= chunk_count
                    or not isinstance(data_chunk, str) or len(data_chunk) > 360_000
                    or not re.fullmatch(r"[A-Za-z0-9+/=]*", data_chunk)
                    or not isinstance(bbox, list) or len(bbox) != 3
                    or not isinstance(shape, list) or len(shape) != 3):
                raise LiveRoomError("Invalid interactive accept event")
            normalized_bbox, normalized_shape = [], []
            for axis, pair in enumerate(bbox):
                if (not isinstance(pair, list) or len(pair) != 2
                        or not isinstance(pair[0], int) or isinstance(pair[0], bool)
                        or not isinstance(pair[1], int) or isinstance(pair[1], bool)
                        or pair[0] < 0 or pair[1] <= pair[0] or axis >= len(dimensions)
                        or pair[1] > int(dimensions[axis])
                        or not isinstance(shape[axis], int) or isinstance(shape[axis], bool)
                        or shape[axis] != pair[1] - pair[0]):
                    raise LiveRoomError("Invalid interactive accept geometry")
                normalized_bbox.append(pair)
                normalized_shape.append(shape[axis])
            label_id = payload.get("label_id")
            organ_label = payload.get("organ_label", "")
            prompt_count = payload.get("prompt_count", 0)
            prompt_log = payload.get("prompt_log", [])
            if (not isinstance(label_id, int) or isinstance(label_id, bool) or not 1 <= label_id <= 65535
                    or not isinstance(organ_label, str) or len(organ_label) > 128
                    or not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count < 0
                    or not isinstance(prompt_log, list) or len(prompt_log) > 10000
                    or any(not isinstance(entry, str) or len(entry) > 128 for entry in prompt_log)
                    or not isinstance(payload.get("allow_overwrite", False), bool)):
                raise LiveRoomError("Invalid accepted organ provenance")
            payload = {"accept_id": accept_id, "bbox": normalized_bbox, "shape": normalized_shape,
                       "encoding": "zlib-base64-uint8", "chunk_index": chunk_index,
                       "chunk_count": chunk_count, "data_chunk": data_chunk, "label_id": label_id,
                       "organ_label": organ_label, "prompt_count": prompt_count,
                       "prompt_log": prompt_log if chunk_index == 0 else [],
                       "allow_overwrite": payload.get("allow_overwrite", False)}
        elif event_type == "interactive.prompt":
            prompt_id = str(payload.get("prompt_id", ""))
            prompt_type = str(payload.get("prompt_type", ""))
            if not prompt_id or len(prompt_id) > 128 or prompt_type not in {"point", "bbox", "scribble", "lasso"}:
                raise LiveRoomError("Invalid interactive prompt provenance")
            chunk_index, chunk_count = payload.get("chunk_index"), payload.get("chunk_count")
            data_chunk = payload.get("data_chunk", "")
            if (not isinstance(chunk_index, int) or not isinstance(chunk_count, int)
                    or chunk_count < 1 or chunk_count > 4096 or chunk_index < 0 or chunk_index >= chunk_count):
                raise LiveRoomError("Invalid interactive delta chunk sequence")
            if not isinstance(data_chunk, str) or len(data_chunk) > 360_000 or not re.fullmatch(r"[A-Za-z0-9+/=]*", data_chunk):
                raise LiveRoomError("Invalid interactive delta chunk")
            bbox = payload.get("bbox")
            shape = payload.get("shape")
            dimensions = metadata.get("dimensions") or []
            if (not isinstance(bbox, list) or len(bbox) != 3 or not isinstance(shape, list) or len(shape) != 3
                    or any(not isinstance(pair, list) or len(pair) != 2 for pair in bbox)):
                raise LiveRoomError("Invalid interactive delta geometry")
            normalized_bbox = []
            normalized_shape = []
            for axis, pair in enumerate(bbox):
                start, end = pair
                if (not isinstance(start, int) or not isinstance(end, int) or start < 0
                        or end <= start or axis >= len(dimensions) or end > int(dimensions[axis])):
                    raise LiveRoomError("Interactive delta is outside the room volume")
                size = end - start
                if not isinstance(shape[axis], int) or shape[axis] != size:
                    raise LiveRoomError("Interactive delta shape does not match its bounds")
                normalized_bbox.append([start, end])
                normalized_shape.append(size)
            raw_prompt = payload.get("prompt")
            if chunk_index == 0 and (not isinstance(raw_prompt, dict) or raw_prompt.get("type") != prompt_type):
                raise LiveRoomError("The first interactive delta chunk requires its prompt")
            prompt = None
            if chunk_index == 0:
                prompt = {"type": prompt_type}
                if prompt_type == "point":
                    coords = raw_prompt.get("coordinates")
                    if (not isinstance(coords, list) or len(coords) != 3
                            or any(not isinstance(value, int) or value < 0 or value >= int(dimensions[i])
                                   for i, value in enumerate(coords))):
                        raise LiveRoomError("Invalid interactive point provenance")
                    prompt["coordinates"] = coords
                elif prompt_type == "bbox":
                    bounds = raw_prompt.get("bounds")
                    if not isinstance(bounds, list) or len(bounds) != 3:
                        raise LiveRoomError("Invalid interactive box provenance")
                    prompt["bounds"] = bounds
                else:
                    raw_bbox = raw_prompt.get("interaction_bbox")
                    points = raw_prompt.get("points")
                    if (not isinstance(raw_bbox, list) or len(raw_bbox) != 3
                            or any(not isinstance(pair, list) or len(pair) != 2
                                   or not isinstance(pair[0], int) or not isinstance(pair[1], int)
                                   or pair[0] < 0 or pair[1] <= pair[0] or pair[1] > int(dimensions[axis])
                                   for axis, pair in enumerate(raw_bbox))
                            or sum(pair[1] - pair[0] == 1 for pair in raw_bbox) != 1
                            or not isinstance(points, list) or not 1 <= len(points) <= 5000
                            or any(not isinstance(point, list) or len(point) != 3
                                   or any(not isinstance(value, int) or value < 0 or value >= int(dimensions[axis])
                                          for axis, value in enumerate(point))
                                   for point in points)):
                        raise LiveRoomError("Invalid interactive stroke provenance")
                    prompt["interaction_bbox"] = raw_bbox
                    prompt["points"] = points
            slice_axis = payload.get("slice_axis", 0)
            slice_index = payload.get("slice_index", 0)
            include = payload.get("include", True)
            if (not isinstance(slice_axis, int) or slice_axis not in (0, 1, 2)
                    or not isinstance(slice_index, int) or slice_index < 0 or slice_index >= int(dimensions[slice_axis])
                    or not isinstance(include, bool)):
                raise LiveRoomError("Invalid interactive slice prompt")
            label_id = payload.get("label_id")
            organ_label = payload.get("organ_label", "")
            prompt_count = payload.get("prompt_count", 0)
            if (label_id is not None and (not isinstance(label_id, int) or isinstance(label_id, bool) or not 1 <= label_id <= 65535)
                    or not isinstance(organ_label, str) or len(organ_label) > 128
                    or not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count < 0):
                raise LiveRoomError("Invalid interactive organ provenance")
            payload = {
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "prompt": prompt if chunk_index == 0 else None,
                "slice_axis": slice_axis,
                "slice_index": slice_index,
                "include": include,
                "bbox": normalized_bbox,
                "shape": normalized_shape,
                "encoding": "zlib-base64-uint8",
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "data_chunk": data_chunk,
                "label_id": label_id,
                "organ_label": organ_label,
                "prompt_count": prompt_count,
            }
        else:
            raise LiveRoomError("Unsupported durable event type")
        return payload, before

    def commit_event(
        self,
        room_id: str,
        room_key: str,
        *,
        event_id: str,
        event_type: str,
        participant_id: str,
        name: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if event_type not in DURABLE_TYPES:
            raise LiveRoomError("Unsupported durable event type")
        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
            raise LiveRoomError("event_id is required")
        if not isinstance(participant_id, str) or not participant_id or len(participant_id) > 128:
            raise LiveRoomError("participant_id is required")
        name = _clean_text(name, MAX_NAME)
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            if metadata.get("mode") == "quiz":
                quiz = self._load_quiz_locked(room_dir)
                if quiz.get("phase") in {"question_open", "question_closed"}:
                    raise LiveRoomError("Collaboration is locked until the question is revealed")
            if event_id in self._event_ids_locked(room_dir, metadata):
                existing = self._find_event(room_dir, event_id)
                if existing:
                    if existing.get("type") != event_type or existing.get("participant_id") != participant_id:
                        raise LiveRoomError("event_id was already used for a different event")
                    return existing, True
            self._ensure_event_capacity_locked(room_dir, metadata)
            state = self._read_json(room_dir / "state.json")
            seq = int(metadata["latest_seq"]) + 1
            normalized_payload, before = self._apply_event(
                room_dir, metadata, state, event_type, payload, seq, name
            )
            event = {
                "seq": seq,
                "event_id": event_id,
                "type": event_type,
                "participant_id": participant_id,
                "name": name,
                "created_at": isoformat(self._now()),
                "payload": normalized_payload,
            }
            if before is not None:
                event["before"] = before
            after_append = None
            rebuild_mask = event_type == "mask.patch" and bool(normalized_payload["ranges"])
            if rebuild_mask:
                metadata["mask_events_since_snapshot"] += 1
                metadata["mask_values_seq"] = seq
                after_append = lambda: self._apply_mask_caches(
                    room_dir, metadata, normalized_payload, seq
                )
            self._append_event_locked(
                room_dir,
                metadata,
                event,
                derived={"state.json": state},
                rebuild_mask=rebuild_mask,
                after_append=after_append,
            )
            if event_type == "mask.patch" and self._snapshot_due(metadata):
                self._materialize_mask_locked(room_dir, metadata)
            return event, False

    def _apply_mask_caches(
        self,
        room_dir: Path,
        metadata: dict[str, Any],
        payload: dict[str, Any],
        seq: int,
    ) -> None:
        nvoxels = int(np.prod(metadata["dimensions"], dtype=np.int64))
        values = self._mask_value_map(room_dir, metadata)
        writers = self._writer_map(room_dir, nvoxels)
        for item in payload.get("ranges", []):
            start = int(item["start"])
            end = start + int(item["length"])
            values[start:end] = int(item["after"])
            writers[start:end] = seq
        values.flush()
        writers.flush()
        del values
        del writers
        metadata["mask_values_seq"] = seq

    def _snapshot_due(self, metadata: dict[str, Any]) -> bool:
        return (
            int(metadata.get("mask_events_since_snapshot", 0)) >= SNAPSHOT_MASK_EVENT_INTERVAL
            or self._now() - parse_time(metadata["last_snapshot_at"]) >= SNAPSHOT_TIME_INTERVAL
        )

    @staticmethod
    def _invalidate_exports(room_dir: Path) -> None:
        for name in ("export.zip", "report.pdf", "report.html"):
            try:
                (room_dir / name).unlink()
            except FileNotFoundError:
                pass

    def _materialize_mask_locked(self, room_dir: Path, metadata: dict[str, Any]) -> Path:
        snapshot_path = room_dir / "mask_snapshot.nii.gz"
        snapshot_seq = int(metadata.get("mask_snapshot_seq", 0))
        source = snapshot_path if snapshot_path.exists() else Path(metadata["base_mask_path"])
        image = nib.load(str(source))
        data = np.array(np.asanyarray(image.dataobj), dtype=np.uint8, order="F")
        flat = data.reshape(-1, order="F")
        latest_applied = snapshot_seq
        for event in self._iter_events(room_dir):
            seq = int(event["seq"])
            if seq <= snapshot_seq or event.get("type") != "mask.patch":
                continue
            for item in event["payload"]["ranges"]:
                flat[item["start"] : item["start"] + item["length"]] = item["after"]
            latest_applied = max(latest_applied, seq)
        output = nib.Nifti1Image(flat.reshape(data.shape, order="F"), image.affine, header=image.header)
        output.set_data_dtype(np.uint8)
        temp_path = room_dir / ".mask_snapshot.tmp.nii.gz"
        nib.save(output, str(temp_path))
        os.replace(temp_path, snapshot_path)
        snapshot_path.chmod(0o600)
        metadata["mask_snapshot_seq"] = latest_applied
        metadata["mask_events_since_snapshot"] = 0
        metadata["last_snapshot_at"] = isoformat(self._now())
        _atomic_json(room_dir / "metadata.json", metadata)
        return snapshot_path

    def _quiz_mask_locked(self, room_dir: Path, metadata: dict[str, Any], *, revealed: bool = False) -> Path:
        path = room_dir / ("quiz_reveal_mask.nii.gz" if revealed else "quiz_blank_mask.nii.gz")
        if path.exists():
            return path
        image = nib.load(str(metadata["base_mask_path"]))
        source = np.asanyarray(image.dataobj)
        output_data = np.zeros(image.shape[:3], dtype=np.uint8)
        if revealed:
            pack = self._pack_locked(room_dir, metadata)
            descriptor = pack["ground_truth"]["reveal_mask"]
            if descriptor.get("kind") != "labels":
                raise LiveRoomError("Quiz reveal-mask descriptor is unsupported")
            labels = [int(value) for value in descriptor.get("source_labels") or []]
            if pack["ground_truth"]["lesion_present"] and not labels:
                raise LiveRoomError("Quiz reveal-mask labels are missing")
            if labels:
                selected = np.isin(source, labels)
                if pack["ground_truth"]["lesion_present"] and not np.any(selected):
                    raise LiveRoomError("Quiz reveal mask is empty")
                output_data[selected] = int(descriptor.get("output_label", 1))
        output = nib.Nifti1Image(output_data, image.affine, header=image.header.copy())
        output.set_data_dtype(np.uint8)
        temporary = room_dir / f".{path.name}.tmp.nii.gz"
        nib.save(output, str(temporary))
        os.replace(temporary, path)
        path.chmod(0o600)
        return path

    def get_mask_snapshot(self, room_id: str, room_key: str) -> tuple[Path, int]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            if metadata.get("mode") == "quiz":
                return self._quiz_mask_locked(room_dir, metadata), 0
            if int(metadata["latest_seq"]) == 0:
                return Path(metadata["base_mask_path"]), 0
            path = self._materialize_mask_locked(room_dir, metadata)
            return path, int(metadata["mask_snapshot_seq"])

    def get_quiz_reveal_segmentation(self, room_id: str, room_key: str) -> bytes:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            self._require_quiz(metadata)
            quiz = self._load_quiz_locked(room_dir)
            pack = self._pack_locked(room_dir, metadata)
            final_revealed = (
                int(quiz.get("question_index", -1)) == len(pack["questions"]) - 1
                and quiz.get("phase") in {"question_revealed", "completed"}
            )
            if not final_revealed:
                error = LiveRoomError("The lesion overlay is available only after the final reveal")
                error.status_code = 403
                error.code = "quiz_reveal_locked"
                raise error
            try:
                return self._quiz_mask_locked(room_dir, metadata, revealed=True).read_bytes()
            except Exception:
                self.quiz_telemetry.record(
                    "reveal_failure",
                    pack_id=pack["pack_id"],
                    case_id=pack["case_id"],
                    mode="race",
                )
                raise

    def _event_is_current(self, state: dict[str, Any], event: dict[str, Any]) -> bool:
        payload = event["payload"]
        if event["type"] == "measurement.upsert":
            current = state["measurements"].get(payload["measurement"]["id"])
            return bool(current and int(current.get("_seq", -1)) == int(event["seq"]))
        if event["type"] == "measurement.delete":
            return payload["id"] not in state["measurements"]
        if event["type"] == "note.upsert":
            current = state["notes"].get(payload["note"]["id"])
            return bool(current and int(current.get("_seq", -1)) == int(event["seq"]))
        if event["type"] == "note.delete":
            return payload["id"] not in state["notes"]
        return True

    def undo_latest(
        self, room_id: str, room_key: str, participant_id: str, name: str
    ) -> dict[str, Any]:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            if metadata.get("mode") == "quiz":
                quiz = self._load_quiz_locked(room_dir)
                if quiz.get("phase") in {"question_open", "question_closed"}:
                    raise LiveRoomError("Collaboration is locked until the question is revealed")
            self._ensure_event_capacity_locked(room_dir, metadata)
            state = self._read_json(room_dir / "state.json")
            undone = set(state.get("undone_event_ids", []))
            all_events = list(self._iter_events(room_dir))
            candidates = [
                event
                for event in all_events
                if event.get("participant_id") == participant_id
                and event.get("event_id") not in undone
                and event.get("type") != "chat.add"
            ]
            target = candidates[-1] if candidates else None
            if target is None:
                return {"ok": False, "reason": "Nothing to undo"}

            def touched_later(object_kind: str, object_id: str) -> bool:
                upsert_type = f"{object_kind}.upsert"
                delete_type = f"{object_kind}.delete"
                for later in all_events:
                    if int(later["seq"]) <= int(target["seq"]):
                        continue
                    later_payload = later.get("payload") or {}
                    later_id = (
                        (later_payload.get(object_kind) or {}).get("id")
                        if later.get("type") == upsert_type
                        else later_payload.get("id") if later.get("type") == delete_type else None
                    )
                    if later_id == object_id:
                        return True
                return False

            seq = int(metadata["latest_seq"]) + 1
            inverse_type = target["type"]
            inverse_payload: dict[str, Any]
            partial = False
            reverted = 0
            total = 0
            if inverse_type == "mask.patch":
                nvoxels = int(np.prod(metadata["dimensions"], dtype=np.int64))
                writers = self._writer_map(room_dir, nvoxels)
                current_values = self._mask_value_map(room_dir, metadata)
                historical_values = self._mask_values_at_seq_locked(
                    room_dir, metadata, int(target["seq"]) - 1
                )
                inverse_ranges: list[dict[str, int]] = []
                for item in target["payload"]["ranges"]:
                    start, length = int(item["start"]), int(item["length"])
                    total += length
                    matches = np.asarray(writers[start : start + length] == int(target["seq"]), dtype=np.bool_)
                    prior = historical_values[start : start + length]
                    current = np.asarray(current_values[start : start + length], dtype=np.uint8)
                    changes = np.flatnonzero(
                        (matches[1:] != matches[:-1]) | (prior[1:] != prior[:-1]) | (current[1:] != current[:-1])
                    ) + 1
                    boundaries = np.concatenate((
                        np.array([0], dtype=np.int64),
                        changes,
                        np.array([length], dtype=np.int64),
                    ))
                    for run_start, run_end in zip(boundaries[:-1], boundaries[1:]):
                        if not matches[int(run_start)]:
                            continue
                        authoritative_before = int(current[int(run_start)])
                        authoritative_after = int(prior[int(run_start)])
                        if authoritative_before == authoritative_after:
                            continue
                        run_length = int(run_end - run_start)
                        inverse_ranges.append(
                            {
                                "start": start + int(run_start),
                                "length": run_length,
                                "before": authoritative_before,
                                "after": authoritative_after,
                            }
                        )
                        reverted += run_length
                        if len(inverse_ranges) > MAX_MASK_RANGES_PER_EVENT:
                            del writers
                            del current_values
                            return {"ok": False, "reason": "Undo is too fragmented to apply safely", "partial": True}
                del writers
                del current_values
                if not inverse_ranges:
                    state["undone_event_ids"].append(target["event_id"])
                    _atomic_json(room_dir / "state.json", state)
                    return {"ok": False, "reason": "Later edits replaced every affected voxel", "partial": True}
                partial = reverted < total
                inverse_payload = {
                    "operation_id": str(uuid.uuid4()),
                    "geometry_hash": metadata["geometry_hash"],
                    "resolution": metadata["resolution"],
                    "segment_label": target["payload"]["segment_label"],
                    "ranges": inverse_ranges,
                }
            elif inverse_type == "measurement.upsert":
                measurement_id = target["payload"]["measurement"]["id"]
                if not self._event_is_current(state, target):
                    return {"ok": False, "reason": "Measurement changed after this operation"}
                before = target.get("before")
                if before:
                    before = dict(before)
                    before["_seq"] = seq
                    state["measurements"][measurement_id] = before
                    inverse_payload = {"measurement": before}
                else:
                    state["measurements"].pop(measurement_id, None)
                    inverse_type = "measurement.delete"
                    inverse_payload = {"id": measurement_id}
            elif inverse_type == "measurement.delete":
                before = target.get("before")
                target_id = target["payload"]["id"]
                if not before or touched_later("measurement", target_id) or not self._event_is_current(state, target):
                    return {"ok": False, "reason": "Measurement cannot be restored"}
                before = dict(before)
                before["_seq"] = seq
                state["measurements"][before["id"]] = before
                inverse_type = "measurement.upsert"
                inverse_payload = {"measurement": before}
            elif inverse_type == "note.upsert":
                note_id = target["payload"]["note"]["id"]
                if not self._event_is_current(state, target):
                    return {"ok": False, "reason": "Note changed after this operation"}
                before = target.get("before")
                if before:
                    before = dict(before)
                    before["_seq"] = seq
                    state["notes"][note_id] = before
                    inverse_payload = {"note": before}
                else:
                    state["notes"].pop(note_id, None)
                    inverse_type = "note.delete"
                    inverse_payload = {"id": note_id}
            elif inverse_type == "note.delete":
                before = target.get("before")
                target_id = target["payload"]["id"]
                if not before or touched_later("note", target_id) or not self._event_is_current(state, target):
                    return {"ok": False, "reason": "Note cannot be restored"}
                before = dict(before)
                before["_seq"] = seq
                state["notes"][before["id"]] = before
                inverse_type = "note.upsert"
                inverse_payload = {"note": before}
            else:
                return {"ok": False, "reason": "Operation cannot be undone"}

            inverse = {
                "seq": seq,
                "event_id": self._new_event_id_locked(room_dir, metadata),
                "type": inverse_type,
                "participant_id": participant_id,
                "name": _clean_text(name, MAX_NAME),
                "created_at": isoformat(self._now()),
                "payload": inverse_payload,
                "undo_of": target["event_id"],
            }
            state["undone_event_ids"].append(target["event_id"])
            rebuild_mask = inverse_type == "mask.patch"
            after_append = None
            if rebuild_mask:
                metadata["mask_events_since_snapshot"] += 1
                metadata["mask_values_seq"] = seq
                after_append = lambda: self._apply_mask_caches(
                    room_dir, metadata, inverse_payload, seq
                )
            self._append_event_locked(
                room_dir,
                metadata,
                inverse,
                derived={"state.json": state},
                rebuild_mask=rebuild_mask,
                after_append=after_append,
            )
            if inverse_type == "mask.patch" and self._snapshot_due(metadata):
                self._materialize_mask_locked(room_dir, metadata)
            return {
                "ok": True,
                "partial": partial,
                "reverted": reverted if total else None,
                "total": total if total else None,
                "event": inverse,
            }

    def _build_report(self, room_dir: Path, metadata: dict[str, Any], state: dict[str, Any]) -> tuple[Path, Path]:
        report_html = room_dir / "report.html"
        report_pdf = room_dir / "report.pdf"
        measurements = list(state["measurements"].values())
        notes = list(state["notes"].values())
        chat = list(state["chat"])
        rows = "".join(
            f"<tr><td>{html.escape(item['tool'])}</td><td>{html.escape(item.get('label', ''))}</td>"
            f"<td>{html.escape(str(item.get('text', '')))}</td></tr>"
            for item in measurements
        ) or "<tr><td colspan='3'>None</td></tr>"
        note_rows = "".join(
            f"<li><strong>{html.escape(item['author'])}</strong>: {html.escape(item['text'])}</li>"
            for item in notes
        ) or "<li>None</li>"
        chat_rows = "".join(
            f"<li><strong>{html.escape(item['author'])}</strong>: {html.escape(item['text'])}</li>"
            for item in chat
        ) or "<li>None</li>"
        quiz_summary = self._quiz_export_summary_locked(room_dir, metadata)
        quiz_html = ""
        if quiz_summary:
            leaderboard_rows = "".join(
                f"<tr><td>{int(item['rank'])}</td><td>{html.escape(item['name'])}</td>"
                f"<td>{int(item['score'])}/{int(item['max_score'])}</td>"
                f"<td>{html.escape(item['consistency']['status'])}</td></tr>"
                for item in quiz_summary["leaderboard"]
            ) or "<tr><td colspan='4'>No revealed results yet</td></tr>"
            quiz_html = (
                f"<h2>Live Quiz</h2><p>Pack {html.escape(quiz_summary['pack_id'])} "
                f"version {int(quiz_summary['pack_version'])} · phase {html.escape(quiz_summary['phase'])}</p>"
                "<table><thead><tr><th>Rank</th><th>Participant</th><th>Score</th>"
                f"<th>Consistency</th></tr></thead><tbody>{leaderboard_rows}</tbody></table>"
            )
        html_text = f"""<!doctype html><html><head><meta charset='utf-8'><title>BodyMaps Live Room report</title>
<style>body{{font:14px Arial,sans-serif;margin:40px;color:#18212a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px;text-align:left}}small{{color:#53606b}}</style></head><body>
<h1>BodyMaps Live Room report</h1><p>Case {html.escape(metadata['case_id'])} · {html.escape(metadata['resolution'])} resolution</p>
<p>Created {html.escape(metadata['created_at'])} · Expires {html.escape(metadata['expires_at'])}</p>
<h2>Measurements</h2><table><thead><tr><th>Tool</th><th>Label</th><th>Text</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Pinned notes</h2><ul>{note_rows}</ul><h2>Chat</h2><ul>{chat_rows}</ul>
{quiz_html}
<p><small>For research and education use only. Not for diagnostic use.</small></p></body></html>"""
        report_html.write_text(html_text, encoding="utf-8")
        report_html.chmod(0o600)

        pdf = canvas.Canvas(str(report_pdf), pagesize=letter)
        width, height = letter
        y = height - 54
        pdf.setTitle("BodyMaps Live Room report")
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(54, y, "BodyMaps Live Room report")
        y -= 28
        pdf.setFont("Helvetica", 10)
        for line in (
            f"Case {metadata['case_id']} | {metadata['resolution']} resolution",
            f"Created {metadata['created_at']}",
            f"Expires {metadata['expires_at']}",
            f"Measurements: {len(measurements)} | Notes: {len(notes)} | Chat messages: {len(chat)}",
        ):
            pdf.drawString(54, y, line)
            y -= 16
        if quiz_summary:
            y -= 4
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(54, y, "Live Quiz")
            y -= 18
            pdf.setFont("Helvetica", 9)
            pdf.drawString(62, y, f"Pack {quiz_summary['pack_id']} v{quiz_summary['pack_version']} | {quiz_summary['phase']}")
            y -= 14
            for item in quiz_summary["leaderboard"][:20]:
                pdf.drawString(
                    62,
                    y,
                    f"#{item['rank']} {item['name']}: {item['score']}/{item['max_score']} | {item['consistency']['status']}",
                )
                y -= 13
        y -= 8
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(54, y, "Pinned notes")
        y -= 18
        pdf.setFont("Helvetica", 9)
        for item in notes[:40]:
            text = f"{item['author']}: {item['text']}"[:110]
            pdf.drawString(62, y, text)
            y -= 14
            if y < 72:
                pdf.showPage()
                y = height - 54
                pdf.setFont("Helvetica", 9)
        pdf.setFont("Helvetica-Oblique", 9)
        pdf.drawString(54, 36, "For research and education use only. Not for diagnostic use.")
        pdf.save()
        report_pdf.chmod(0o600)
        return report_html, report_pdf

    def _quiz_export_summary_locked(self, room_dir: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
        if metadata.get("mode", "review") != "quiz":
            return None
        quiz = self._load_quiz_locked(room_dir)
        pack = self._pack_locked(room_dir, metadata)
        return {
            "pack_id": pack["pack_id"],
            "pack_version": pack["version"],
            "case_id": pack["case_id"],
            "timer_seconds": metadata.get("quiz_timer_seconds"),
            "phase": quiz.get("phase"),
            "revealed_distributions": {
                question_id: reveal.get("distribution", {})
                for question_id, reveal in (quiz.get("reveals") or {}).items()
            },
            "leaderboard": quiz.get("leaderboard") or [],
            "consistency_summary": quiz.get("consistency_summary") or {},
            "question_elapsed_ms": quiz.get("question_elapsed_ms") or {},
            "disclaimer": "For research and education use only. Not for diagnostic use.",
        }

    def _require_quiz_export_ready_locked(self, room_dir: Path, metadata: dict[str, Any]) -> None:
        if metadata.get("mode", "review") != "quiz":
            return
        quiz = self._load_quiz_locked(room_dir)
        pack = self._pack_locked(room_dir, metadata)
        final_revealed = (
            int(quiz.get("question_index", -1)) == len(pack["questions"]) - 1
            and quiz.get("phase") in {"question_revealed", "completed"}
        )
        if not final_revealed:
            error = LiveRoomError("Quiz exports are available only after the final reveal")
            error.status_code = 403
            error.code = "quiz_export_locked"
            raise error

    @staticmethod
    def _chat_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            event["payload"]["message"]
            for event in events
            if event.get("type") == "chat.add" and (event.get("payload") or {}).get("message")
        ]

    def build_export(self, room_id: str, room_key: str) -> Path:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            self._require_quiz_export_ready_locked(room_dir, metadata)
            export_path = room_dir / "export.zip"
            if export_path.exists():
                return export_path
            state = self._read_json(room_dir / "state.json")
            events = list(self._iter_events(room_dir))
            export_state = {**state, "chat": self._chat_from_events(events)}
            if metadata.get("mode") == "quiz":
                quiz = self._load_quiz_locked(room_dir)
                pack = self._pack_locked(room_dir, metadata)
                final_revealed = (
                    int(quiz.get("question_index", -1)) == len(pack["questions"]) - 1
                    and quiz.get("phase") in {"question_revealed", "completed"}
                )
                mask_path = self._quiz_mask_locked(room_dir, metadata, revealed=final_revealed)
            else:
                mask_path = self._materialize_mask_locked(room_dir, metadata)
            report_html, report_pdf = self._build_report(room_dir, metadata, export_state)
            measurements = list(state["measurements"].values())
            measurements_json = json.dumps(measurements, indent=2, ensure_ascii=False)
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=["id", "tool", "label", "text", "revision", "frame_of_reference"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(
                {key: _csv_safe(value) for key, value in measurement.items()}
                for measurement in measurements
            )
            manifest = {
                **self.public_metadata(metadata),
                "exported_at": isoformat(self._now()),
                "participant_nicknames": sorted({event["name"] for event in events}),
                "artifacts": [
                    "edited_labelmap.nii.gz",
                    "measurements.csv",
                    "measurements.json",
                    "notes.json",
                    "chat.json",
                    "events.json",
                    "report.html",
                    "report.pdf",
                ],
                "disclaimer": "For research and education use only. Not for diagnostic use.",
            }
            quiz_summary = self._quiz_export_summary_locked(room_dir, metadata)
            if quiz_summary:
                manifest["artifacts"].append("quiz-summary.json")
            temp_path = room_dir / ".export.tmp.zip"
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(mask_path, "edited_labelmap.nii.gz")
                archive.writestr("measurements.csv", csv_buffer.getvalue())
                archive.writestr("measurements.json", measurements_json)
                archive.writestr("notes.json", json.dumps(list(state["notes"].values()), indent=2, ensure_ascii=False))
                archive.writestr("chat.json", json.dumps(export_state["chat"], indent=2, ensure_ascii=False))
                archive.writestr("events.json", json.dumps(events, indent=2, ensure_ascii=False))
                archive.write(report_html, "report.html")
                archive.write(report_pdf, "report.pdf")
                if quiz_summary:
                    archive.writestr("quiz-summary.json", json.dumps(quiz_summary, indent=2, ensure_ascii=False))
                archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            if temp_path.stat().st_size > MAX_EXPORT_BYTES:
                temp_path.unlink(missing_ok=True)
                raise RoomFull("Room export exceeds the 4 GiB artifact limit")
            os.replace(temp_path, export_path)
            export_path.chmod(0o600)
            return export_path

    def get_report(self, room_id: str, room_key: str) -> Path:
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            if not hmac.compare_digest(hash_room_key(room_key), metadata.get("key_hash", "")):
                raise RoomUnauthorized("Invalid room key")
            self._require_quiz_export_ready_locked(room_dir, metadata)
            state = self._read_json(room_dir / "state.json")
            events = list(self._iter_events(room_dir))
            report_state = {**state, "chat": self._chat_from_events(events)}
            _, report_pdf = self._build_report(room_dir, metadata, report_state)
            return report_pdf

    def open_artifact(self, room_id: str, room_key: str, path: Path):
        """Open a previously authorized artifact while the room lock prevents replacement."""
        with self._locked(room_id) as room_dir:
            metadata = self._load_metadata(room_dir)
            self._authorize_locked(metadata, room_key)
            resolved = path.resolve(strict=True)
            room_resolved = room_dir.resolve()
            allowed_external = {Path(metadata["base_mask_path"]).resolve()}
            if resolved.parent != room_resolved and resolved not in allowed_external:
                raise RoomUnauthorized("Artifact does not belong to this room")
            if path.is_symlink() or not resolved.is_file():
                raise LiveRoomError("Room artifact is not a regular file")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            return os.fdopen(os.open(resolved, flags), "rb")

    def cleanup_expired(self) -> list[str]:
        removed: list[str] = []
        for child in list(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                room_id = self.validate_room_id(child.name)
                with self._locked(room_id) as room_dir:
                    metadata = self._load_metadata(room_dir, allow_expired=True)
                    if parse_time(metadata["expires_at"]) > self._now():
                        continue
                shutil.rmtree(child, ignore_errors=True)
                removed.append(room_id)
            except (LiveRoomError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipped corrupt Live Room during cleanup room=%s error=%s", child.name, exc)
                continue
        return removed
