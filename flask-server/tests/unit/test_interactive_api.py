import base64
import zlib

import nibabel as nib
import numpy as np
import pytest
from flask import Flask

from api import interactive
from services.interactive_sessions import InteractiveCapacity, InteractiveSessionManager
from services import interactive_sessions


@pytest.fixture
def client(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    ct_dir = dataset / "image_only" / "PanTS_00000017"
    ct_dir.mkdir(parents=True)
    ct_path = ct_dir / "ct.nii.gz"
    image = np.arange(12 * 14 * 8, dtype=np.int16).reshape((12, 14, 8)) - 1000
    nib.save(nib.Nifti1Image(image, np.eye(4)), ct_path)
    label_dir = dataset / "mask_only" / "PanTS_00000017"
    label_dir.mkdir(parents=True)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[4:7, 5:8, 2:4] = 23
    label_path = label_dir / "combined_labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), label_path)
    original = ct_path.read_bytes()
    label_original = label_path.read_bytes()
    monkeypatch.setattr(interactive.Constants, "PANTS_PATH", str(dataset))
    monkeypatch.setattr(interactive, "manager", InteractiveSessionManager())
    monkeypatch.setenv("NNINTERACTIVE_ENABLED", "true")
    monkeypatch.setenv("NNINTERACTIVE_MOCK", "true")
    monkeypatch.setenv("NNINTERACTIVE_DAILY_QUOTA", "5")
    monkeypatch.setattr(interactive, "current_user", lambda: {"id": "researcher"})
    monkeypatch.setattr(interactive.plan_store, "is_verified_researcher", lambda _uid: True)
    charges = []
    monkeypatch.setattr(interactive.plan_store, "count_interactive_sessions", lambda _uid: len(charges))
    monkeypatch.setattr(interactive.plan_store, "record_interactive_session", lambda _uid, sid: charges.append(sid))
    app = Flask(__name__)
    app.register_blueprint(interactive.interactive_blueprint, url_prefix="/api")
    app.testing = True
    yield app.test_client(), charges, ct_path, original, label_path, label_original
    interactive.manager.close_all()


def _start(client):
    response = client.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17"})
    assert response.status_code == 201, response.json
    return response.json["session_id"]


def _region_bytes(delta):
    return zlib.decompress(base64.b64decode(delta["data"]))


def test_feature_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NNINTERACTIVE_ENABLED", raising=False)
    app = Flask(__name__)
    app.register_blueprint(interactive.interactive_blueprint, url_prefix="/api")
    assert app.test_client().post("/api/interactive/start", json={}).status_code == 404


def test_start_prompts_preview_undo_reset_and_accept_do_not_write_dataset(client):
    http, _charges, path, original, label_path, label_original = client
    session_id = _start(http)
    start_session = {"session_id": session_id}

    point = http.post("/api/interactive/point", json={**start_session, "slice_axis": 2, "slice_index": 4,
                                                       "coordinates": [6, 7, 4]})
    assert point.status_code == 200
    assert point.json["delta"]["encoding"] == "zlib-base64-uint8"
    assert len(_region_bytes(point.json["delta"])) == int(np.prod(point.json["delta"]["shape"]))

    negative = http.post("/api/interactive/point", json={**start_session, "slice_axis": 2, "slice_index": 4,
                                                          "coordinates": [6, 7, 4], "include": False})
    assert negative.status_code == 200
    undo = http.post("/api/interactive/undo", json=start_session)
    assert undo.status_code == 200 and undo.json["undone"]

    bbox = http.post("/api/interactive/bbox", json={**start_session, "slice_axis": 2, "slice_index": 3,
        "bounds": [[3, 8], [4, 10], [3, 4]]})
    assert bbox.status_code == 200
    scribble = http.post("/api/interactive/scribble", json={**start_session, "slice_axis": 2, "slice_index": 2,
        "interaction_bbox": [[2, 5], [2, 6], [2, 3]],
        "crop": [[[1], [1], [0], [0]], [[0], [1], [1], [0]], [[0], [0], [1], [0]]]})
    assert scribble.status_code == 200, scribble.json
    lasso = http.post("/api/interactive/lasso", json={**start_session, "slice_axis": 2, "slice_index": 1,
        "interaction_bbox": [[2, 5], [2, 6], [1, 2]],
        "crop": [[[1], [1], [0], [0]], [[0], [1], [1], [0]], [[0], [0], [1], [0]]]})
    assert lasso.status_code == 200, lasso.json

    accepted = http.post("/api/interactive/commit", json={**start_session, "label_name": "Test organ", "label_id": 23})
    assert accepted.status_code == 200
    reset = http.post("/api/interactive/reset", json=start_session)
    assert reset.status_code == 200
    assert path.read_bytes() == original
    assert label_path.read_bytes() == label_original


def test_ownership_verification_and_daily_quota(client, monkeypatch):
    http, charges, _path, _original, _labels, _label_original = client
    session_id = _start(http)
    monkeypatch.setattr(interactive, "current_user", lambda: {"id": "another-user"})
    response = http.post("/api/interactive/reset", json={"session_id": session_id})
    assert response.status_code == 404
    assert http.post("/api/interactive/close", json={"session_id": session_id}).status_code == 404
    assert http.post("/api/interactive/commit", json={"session_id": session_id, "label_name": "x", "label_id": 1}).status_code == 404
    monkeypatch.setattr(interactive, "current_user", lambda: None)
    assert http.post("/api/interactive/start", json={}).status_code == 401
    monkeypatch.setattr(interactive, "current_user", lambda: {"id": "researcher"})
    monkeypatch.setattr(interactive.plan_store, "is_verified_researcher", lambda _uid: False)
    assert http.post("/api/interactive/start", json={}).status_code == 403
    monkeypatch.setattr(interactive.plan_store, "is_verified_researcher", lambda _uid: True)
    monkeypatch.setattr(interactive.plan_store, "count_interactive_sessions", lambda _uid: 5)
    blocked = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17"})
    assert blocked.status_code == 429 and blocked.json["code"] == "daily_quota_exceeded"
    assert len(charges) == 1


def test_capacity_returns_retry_after(client, monkeypatch):
    http, _charges, _path, _original, _labels, _label_original = client
    monkeypatch.setattr(interactive.manager, "create", lambda *_args, **_kwargs: (_ for _ in ()).throw(InteractiveCapacity("full")))
    response = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"


def test_start_timeout_is_a_clear_gateway_timeout(client, monkeypatch):
    class ReadTimeout(Exception):
        pass

    monkeypatch.setattr(interactive_sessions, "make_remote_session", lambda: (_ for _ in ()).throw(ReadTimeout()))
    http, *_rest = client
    response = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17"})
    assert response.status_code == 504
    assert response.json["code"] == "server_timeout"
    assert "timed out" in response.json["error"]


@pytest.mark.parametrize("failure, expected_status, expected_code", [
    ("capacity", 503, "server_at_capacity"),
    ("timeout", 504, "server_timeout"),
])
def test_prompt_capacity_and_timeout_are_clear_http_errors(client, failure, expected_status, expected_code):
    http, *_rest = client
    sid = _start(http)
    item = interactive.manager.get(sid, "researcher")
    error = type("ServerAtCapacityError", (Exception,), {}) if failure == "capacity" else type("ReadTimeout", (Exception,), {})
    item.remote.add_point_interaction = lambda *_args, **_kwargs: (_ for _ in ()).throw(error())
    response = http.post("/api/interactive/point", json={"session_id": sid, "slice_axis": 2,
        "slice_index": 4, "coordinates": [6, 7, 4]})
    assert response.status_code == expected_status
    assert response.json["code"] == expected_code
    if failure == "capacity":
        assert response.headers["Retry-After"] == "30"


def test_mock_requires_explicit_env_and_real_client_is_default(monkeypatch):
    import sys
    import types

    captured = {}
    package = types.ModuleType("nnInteractive")
    package.__path__ = []
    inference = types.ModuleType("nnInteractive.inference")
    inference.__path__ = []
    remote_module = types.ModuleType("nnInteractive.inference.remote")
    class RealRemote:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    remote_module.nnInteractiveRemoteInferenceSession = RealRemote
    monkeypatch.setitem(sys.modules, "nnInteractive", package)
    monkeypatch.setitem(sys.modules, "nnInteractive.inference", inference)
    monkeypatch.setitem(sys.modules, "nnInteractive.inference.remote", remote_module)
    monkeypatch.setenv("NNINTERACTIVE_MOCK", "false")
    monkeypatch.setenv("NNINTERACTIVE_SERVER_URL", "https://gpu.example")
    monkeypatch.setenv("NNINTERACTIVE_API_KEY", "server-only-secret")
    remote = interactive_sessions.make_remote_session()
    assert isinstance(remote, RealRemote)
    assert captured == {"server_url": "https://gpu.example", "api_key": "server-only-secret"}
    monkeypatch.setenv("NNINTERACTIVE_MOCK", "true")
    assert isinstance(interactive_sessions.make_remote_session(), __import__("services.interactive_mock", fromlist=["FakeInteractiveSession"]).FakeInteractiveSession)


def test_api_key_is_not_returned_to_browser(client, monkeypatch):
    http, *_rest = client
    monkeypatch.setenv("NNINTERACTIVE_API_KEY", "never-send-this")
    response = http.get("/api/interactive/config")
    assert response.status_code == 200
    assert b"never-send-this" not in response.data


def test_live_room_reuses_one_session_across_verified_editors(client, monkeypatch):
    http, charges, _path, _original, _labels, _label_original = client
    monkeypatch.setattr(interactive, "_room_context", lambda _room_id: {
        "room_id": "room-1", "case_id": "PanTS_00000017", "mode": "review"
    })
    first = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17", "room_id": "room-1"},
                      headers={"X-Room-Key": "valid"})
    assert first.status_code == 201
    monkeypatch.setattr(interactive, "current_user", lambda: {"id": "collaborator"})
    second = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17", "room_id": "room-1"},
                       headers={"X-Room-Key": "valid"})
    assert second.status_code == 200
    assert second.json["session_id"] == first.json["session_id"]
    assert len(charges) == 2
    point = http.post("/api/interactive/point", json={"session_id": first.json["session_id"], "room_id": "room-1",
        "slice_axis": 2, "slice_index": 4, "coordinates": [6, 7, 4]}, headers={"X-Room-Key": "valid"})
    assert point.status_code == 200
    without_room = http.post("/api/interactive/point", json={"session_id": first.json["session_id"],
        "slice_axis": 2, "slice_index": 4, "coordinates": [6, 7, 4]}, headers={"X-Room-Key": "valid"})
    assert without_room.status_code == 404


def test_invalid_geometry_is_rejected(client):
    http, _charges, _path, _original, _labels, _label_original = client
    session_id = _start(http)
    response = http.post("/api/interactive/bbox", json={"session_id": session_id, "slice_axis": 2,
        "slice_index": 0, "bounds": [[1, 3], [1, 4], [0, 2]]})
    assert response.status_code == 400


def test_existing_organ_mask_is_used_as_initial_seed_and_switchable(client):
    http, _charges, _ct_path, _ct_original, _labels, _label_original = client
    started = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17", "initial_mask_label": 23})
    assert started.status_code == 201, started.json
    seeded = np.frombuffer(_region_bytes(started.json["seeded_mask"]), dtype=np.uint8)
    assert seeded.sum() == 18
    switched = http.post("/api/interactive/select-organ", json={"session_id": started.json["session_id"], "label_id": 23})
    assert switched.status_code == 200
    assert np.frombuffer(_region_bytes(switched.json["seeded_mask"]), dtype=np.uint8).sum() == 18


def test_reaccept_region_includes_removed_seed_voxels(client):
    http, _charges, *_rest = client
    started = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17", "initial_mask_label": 23})
    sid = started.json["session_id"]
    changed = http.post("/api/interactive/bbox", json={"session_id": sid, "slice_axis": 2, "slice_index": 2,
        "bounds": [[4, 6], [5, 7], [2, 3]], "include": False})
    assert changed.status_code == 200
    accepted = http.post("/api/interactive/commit", json={"session_id": sid, "label_name": "organ", "label_id": 23})
    assert accepted.status_code == 200
    mask = accepted.json["mask"]
    values = np.frombuffer(_region_bytes(mask), dtype=np.uint8).reshape(mask["shape"])
    assert mask["bbox"] == [[4, 7], [5, 8], [2, 4]]
    assert values[0, 0, 0] == 0
    assert values.sum() == 14
    item = interactive.manager.get(sid, "researcher")
    assert int(item.seed_target.sum()) == 14
    assert int(item.target.sum()) == 14


def test_expired_remote_session_is_recreated_and_prompts_are_replayed(monkeypatch):
    class SessionExpiredError(Exception):
        pass

    class Remote:
        created = 0

        def __init__(self):
            type(self).created += 1
            self.inner = __import__("services.interactive_mock", fromlist=["FakeInteractiveSession"]).FakeInteractiveSession()
            self.point_calls = 0
            self.fail_on_point = 2 if type(self).created == 1 else None

        def set_image(self, image):
            self.inner.set_image(image)

        def set_target_buffer(self, target):
            self.inner.set_target_buffer(target)

        def add_point_interaction(self, coordinates, include_interaction=True, **kwargs):
            self.point_calls += 1
            if self.point_calls == self.fail_on_point:
                raise SessionExpiredError()
            return self.inner.add_point_interaction(coordinates, include_interaction, **kwargs)

        def add_bbox_interaction(self, *args, **kwargs):
            return self.inner.add_bbox_interaction(*args, **kwargs)

        def add_scribble_interaction(self, *args, **kwargs):
            return self.inner.add_scribble_interaction(*args, **kwargs)

        def add_lasso_interaction(self, *args, **kwargs):
            return self.inner.add_lasso_interaction(*args, **kwargs)

        def close(self):
            self.inner.close()

        def undo(self):
            return self.inner.undo()

        def reset_interactions(self):
            return self.inner.reset_interactions()

    Remote.created = 0
    monkeypatch.setattr(interactive_sessions, "make_remote_session", Remote)
    manager = InteractiveSessionManager()
    image = np.zeros((20, 20, 20), dtype=np.int16)
    item = manager.create("owner", "PanTS", "17", image)
    first_prompt = {"type": "point", "coordinates": [8, 8, 8], "include": True}
    second_prompt = {"type": "point", "coordinates": [10, 10, 10], "include": True}
    manager.apply(item, "point", first_prompt)
    manager.apply(item, "point", second_prompt)
    assert Remote.created == 2
    assert len(item.prompts) == 2
    assert len(item.remote.inner._prompts) == 2
    assert int(item.target.sum()) > 0
    manager.close_all()
