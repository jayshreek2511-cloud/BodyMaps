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
    original = ct_path.read_bytes()
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
    yield app.test_client(), charges, ct_path, original
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
    http, _charges, path, original = client
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


def test_ownership_verification_and_daily_quota(client, monkeypatch):
    http, charges, _path, _original = client
    session_id = _start(http)
    monkeypatch.setattr(interactive, "current_user", lambda: {"id": "another-user"})
    response = http.post("/api/interactive/reset", json={"session_id": session_id})
    assert response.status_code == 404
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
    http, _charges, _path, _original = client
    monkeypatch.setattr(interactive.manager, "create", lambda *_args: (_ for _ in ()).throw(InteractiveCapacity("full")))
    response = http.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "17"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"


def test_invalid_geometry_is_rejected(client):
    http, _charges, _path, _original = client
    session_id = _start(http)
    response = http.post("/api/interactive/bbox", json={"session_id": session_id, "slice_axis": 2,
        "slice_index": 0, "bounds": [[1, 3], [1, 4], [0, 2]]})
    assert response.status_code == 400


def test_expired_remote_session_is_recreated_and_prompts_are_replayed(monkeypatch):
    class SessionExpiredError(Exception):
        pass

    class Remote:
        created = 0

        def __init__(self):
            type(self).created += 1
            self.inner = __import__("services.interactive_mock", fromlist=["FakeInteractiveSession"]).FakeInteractiveSession()
            self.fail_next_point = type(self).created == 1

        def set_image(self, image):
            self.inner.set_image(image)

        def set_target_buffer(self, target):
            self.inner.set_target_buffer(target)

        def add_point_interaction(self, coordinates, include_interaction=True, **kwargs):
            if self.fail_next_point:
                self.fail_next_point = False
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
    prompt = {"type": "point", "coordinates": [10, 10, 10], "include": True}
    manager.apply(item, "point", prompt)
    assert Remote.created == 2
    assert len(item.prompts) == 1
    assert int(item.target.sum()) > 0
    manager.close_all()
