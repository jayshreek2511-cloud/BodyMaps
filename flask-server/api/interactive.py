"""Feature-flagged, authenticated HTTP boundary for remote nnInteractive."""

from __future__ import annotations

import atexit
import os
import re
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
from flask import Blueprint, jsonify, request

from api.auth import current_user
from constants import Constants
from services import plan_store
from services.interactive_sessions import (
    InteractiveCapacity, InteractiveError, InteractiveExpired,
    InteractiveNotFound, InteractiveSessionManager, InteractiveTimeout,
    decode_crop, encode_region,
)

interactive_blueprint = Blueprint("interactive", __name__)
manager = InteractiveSessionManager()
_quota_lock = threading.RLock()
atexit.register(manager.close_all)


def _enabled() -> bool:
    return os.environ.get("NNINTERACTIVE_ENABLED", "false").lower() == "true"


def _json_body() -> dict:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise InteractiveError("Request body must be a JSON object")
    return value


def _require_feature_and_user():
    if not _enabled():
        return jsonify({"error": "Interactive segmentation is not enabled", "code": "feature_disabled"}), 404
    user = current_user()
    if user is None:
        return jsonify({"error": "Authentication required", "code": "authentication_required"}), 401
    if not plan_store.is_verified_researcher(user["id"]):
        return jsonify({"error": "A verified research account is required", "code": "verification_required"}), 403
    return user


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InteractiveError(f"{name} must be an integer")
    return int(value)


def _bbox(value, shape: tuple[int, int, int], *, require_slice: bool = True) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 3:
        raise InteractiveError("bbox must contain three [start, end] pairs")
    result = []
    for axis, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise InteractiveError("bbox must contain three [start, end] pairs")
        start, end = _integer(pair[0], "bbox start"), _integer(pair[1], "bbox end")
        if start < 0 or end > shape[axis] or start >= end:
            raise InteractiveError("bbox is outside the volume or empty")
        result.append([start, end])
    if require_slice and sum((b - a) == 1 for a, b in result) != 1:
        raise InteractiveError("bbox must lie on one 2D slice")
    return result


def _volume_path(dataset: str, case_id: str) -> Path:
    dataset_key = dataset.lower()
    if dataset_key in {"pants", "pant-s", "pan_ts"}:
        if not re.fullmatch(r"\d{1,8}|PanTS_\d{8}", case_id):
            raise InteractiveError("Invalid PanTS case ID")
        numeric = case_id[-8:] if case_id.startswith("PanTS_") else case_id
        root = Constants.PANTS_PATH
        folder = f"PanTS_{int(numeric):08d}"
    elif dataset_key in {"cancerverse", "cv"}:
        if not re.fullmatch(r"CV_?\d{1,8}", case_id, re.IGNORECASE):
            raise InteractiveError("Invalid CancerVerse case ID")
        root = Constants.CANCERVERSE_PATH
        folder = "CV_" + case_id.upper().removeprefix("CV_").removeprefix("CV").zfill(8)
    else:
        raise InteractiveError("Dataset must be PanTS or CancerVerse")
    if not root:
        raise InteractiveError(f"{dataset} dataset is not configured on this server")
    base = (Path(root) / "image_only" / folder / "ct.nii.gz").resolve()
    allowed_root = Path(root).resolve()
    if allowed_root not in base.parents:
        raise InteractiveError("Invalid dataset path")
    if not base.is_file():
        raise InteractiveError("Full-resolution CT was not found for this case")
    return base


def _load_original_ct(dataset: str, case_id: str) -> np.ndarray:
    path = _volume_path(dataset, case_id)
    image = nib.load(str(path), mmap=True)
    if len(image.shape) != 3 or any(int(size) <= 0 for size in image.shape):
        raise InteractiveError("CT must be a three-dimensional NIfTI volume")
    if int(np.prod(image.shape, dtype=np.int64)) > 512_000_000:
        raise InteractiveError("CT volume exceeds the interactive segmentation size limit")
    # ArrayProxy applies NIfTI slope/intercept while preserving raw CT intensity
    # values; there is no windowing, normalization, or uint8 conversion.
    volume = np.asanyarray(image.dataobj)
    if not np.issubdtype(volume.dtype, np.number) or not np.isfinite(volume).all():
        raise InteractiveError("CT contains invalid voxel values")
    return np.ascontiguousarray(volume)


def _session():
    user = _require_feature_and_user()
    if not isinstance(user, dict):
        return None, user
    body = _json_body()
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or len(session_id) > 64:
        raise InteractiveError("session_id is required")
    return manager.get(session_id, str(user["id"])), user


@interactive_blueprint.errorhandler(InteractiveError)
def _interactive_error(error: InteractiveError):
    response = jsonify({"error": str(error), "code": error.code})
    if isinstance(error, InteractiveCapacity):
        response.headers["Retry-After"] = "30"
    return response, error.status_code


@interactive_blueprint.get("/interactive/config")
def interactive_config():
    if not _enabled():
        return jsonify({"enabled": False}), 200
    return jsonify({
        "enabled": True,
        "license": "CC BY-NC-SA 4.0",
        "research_use_only": True,
        "mock": os.environ.get("NNINTERACTIVE_MOCK", "false").lower() == "true",
    }), 200


@interactive_blueprint.post("/interactive/start")
def interactive_start():
    user = _require_feature_and_user()
    if not isinstance(user, dict):
        return user
    body = _json_body()
    dataset, case_id = str(body.get("dataset", "PanTS")), str(body.get("case_id", ""))
    quota = max(0, int(os.environ.get("NNINTERACTIVE_DAILY_QUOTA", "5")))
    active_max = max(1, int(os.environ.get("NNINTERACTIVE_MAX_SESSIONS_PER_USER", "2")))
    with _quota_lock:
        if plan_store.count_interactive_sessions(str(user["id"])) >= quota:
            return jsonify({"error": "Daily interactive session limit reached", "code": "daily_quota_exceeded",
                            "limit": quota}), 429
        if manager.active_count(str(user["id"])) >= active_max:
            return jsonify({"error": "Too many active interactive sessions", "code": "active_session_limit",
                            "limit": active_max}), 429
        image = _load_original_ct(dataset, case_id)
        item = manager.create(str(user["id"]), dataset, case_id, image)
        plan_store.record_interactive_session(str(user["id"]), item.session_id)
    return jsonify({"session_id": item.session_id, "shape": list(item.image.shape),
                    "dtype": str(item.image.dtype), "dataset": dataset, "case_id": case_id}), 201


def _do_prompt(kind: str):
    item, user = _session()
    if item is None:
        return user
    body = _json_body()
    shape = tuple(int(size) for size in item.target.shape)
    include = body.get("include", True)
    if not isinstance(include, bool):
        raise InteractiveError("include must be true or false")
    slice_axis = _integer(body.get("slice_axis"), "slice_axis")
    slice_index = _integer(body.get("slice_index"), "slice_index")
    if slice_axis not in range(3) or slice_index < 0 or slice_index >= shape[slice_axis]:
        raise InteractiveError("slice axis or index is outside the volume")
    prompt = {"type": kind, "include": include, "slice_axis": slice_axis, "slice_index": slice_index}
    if kind == "point":
        coordinates = body.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 3:
            raise InteractiveError("coordinates must contain three voxel indices")
        coords = [_integer(value, "coordinate") for value in coordinates]
        if any(value < 0 or value >= shape[axis] for axis, value in enumerate(coords)):
            raise InteractiveError("point is outside the volume")
        if coords[slice_axis] != slice_index:
            raise InteractiveError("point must lie on the selected slice")
        prompt["coordinates"] = coords
    elif kind == "bbox":
        prompt["bounds"] = _bbox(body.get("bounds"), shape)
        if prompt["bounds"][slice_axis][1] - prompt["bounds"][slice_axis][0] != 1 or prompt["bounds"][slice_axis][0] != slice_index:
            raise InteractiveError("bbox must lie on the selected slice")
    else:
        bounds = _bbox(body.get("interaction_bbox"), shape)
        if bounds[slice_axis][1] - bounds[slice_axis][0] != 1 or bounds[slice_axis][0] != slice_index:
            raise InteractiveError("interaction_bbox must lie on the selected slice")
        prompt["interaction_bbox"] = bounds
        prompt["crop"] = decode_crop(body.get("crop"), bounds, shape)
    with item.lock:
        changed = manager.apply(item, kind, prompt)
        region = encode_region(item.target, changed)
    return jsonify({"session_id": item.session_id, "changed_bbox": changed, "delta": region,
                    "status": "predicted"}), 200


@interactive_blueprint.post("/interactive/point")
def interactive_point():
    return _do_prompt("point")


@interactive_blueprint.post("/interactive/bbox")
def interactive_bbox():
    return _do_prompt("bbox")


@interactive_blueprint.post("/interactive/scribble")
def interactive_scribble():
    return _do_prompt("scribble")


@interactive_blueprint.post("/interactive/lasso")
def interactive_lasso():
    return _do_prompt("lasso")


@interactive_blueprint.post("/interactive/undo")
def interactive_undo():
    item, user = _session()
    if item is None:
        return user
    with item.lock:
        changed = manager.undo(item)
        region = encode_region(item.target, changed)
    return jsonify({"changed_bbox": changed, "delta": region, "undone": changed is not None}), 200


@interactive_blueprint.post("/interactive/reset")
def interactive_reset():
    item, user = _session()
    if item is None:
        return user
    with item.lock:
        manager.reset(item)
    return jsonify({"reset": True}), 200


@interactive_blueprint.post("/interactive/commit")
def interactive_commit():
    item, user = _session()
    if item is None:
        return user
    body = _json_body()
    label_name = body.get("label_name")
    label_id = body.get("label_id")
    if not isinstance(label_name, str) or not label_name.strip() or len(label_name) > 128:
        raise InteractiveError("label_name is required and must be at most 128 characters")
    if isinstance(label_id, bool) or not isinstance(label_id, int) or label_id < 1 or label_id > 65535:
        raise InteractiveError("label_id must be an integer between 1 and 65535")
    with item.lock:
        bbox = _mask_bbox(item.target)
        delta = encode_region(item.target, bbox)
    if delta is None:
        raise InteractiveError("There is no preview mask to accept")
    return jsonify({"accepted": True, "label_name": label_name.strip(), "label_id": label_id,
                    "bbox": bbox, "mask": delta}), 200


def _mask_bbox(mask: np.ndarray):
    coords = np.argwhere(mask != 0)
    if not len(coords):
        return None
    return [[int(coords[:, i].min()), int(coords[:, i].max()) + 1] for i in range(3)]


@interactive_blueprint.post("/interactive/close")
def interactive_close():
    item, user = _session()
    if item is None:
        return user
    manager.close(item.session_id, str(user["id"]))
    return jsonify({"closed": True}), 200


@interactive_blueprint.get("/interactive/health")
def interactive_health():
    if not _enabled():
        return jsonify({"error": "Interactive segmentation is not enabled", "code": "feature_disabled"}), 404
    from services.interactive_sessions import make_remote_session
    remote = None
    try:
        remote = make_remote_session()
        healthy = bool(remote.ping(timeout=5.0))
        if not healthy:
            return jsonify({"status": "unavailable", "license": getattr(remote, "license", "CC BY-NC-SA 4.0")}), 503
        return jsonify({"status": "ok", "license": getattr(remote, "license", "CC BY-NC-SA 4.0"),
                        "research_use_only": True}), 200
    except Exception as exc:
        return jsonify({"status": "unavailable", "error": type(exc).__name__}), 503
    finally:
        if remote is not None:
            try:
                remote.close()
            except Exception:
                pass

