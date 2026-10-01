import numpy as np
import pytest
from flask import Flask

from api import interactive
from services.interactive_sessions import InteractiveNotFound, InteractiveSessionManager


class FakeRemoteSession:
    def set_image(self, _image):
        pass

    def set_target_buffer(self, _target):
        pass

    def close(self):
        pass


@pytest.fixture
def replacement_client(monkeypatch):
    manager = InteractiveSessionManager()
    monkeypatch.setattr(interactive, "manager", manager)
    interactive._user_sessions.clear()
    monkeypatch.setattr(interactive, "current_user", lambda: {"id": "researcher"})
    monkeypatch.setattr(interactive.plan_store, "is_verified_researcher", lambda _uid: True)
    monkeypatch.setattr(interactive.plan_store, "count_interactive_sessions", lambda _uid: 0)
    monkeypatch.setattr(interactive.plan_store, "record_interactive_session", lambda *_args: None)
    monkeypatch.setattr(interactive, "_load_original_ct", lambda *_args: np.zeros((4, 5, 6), dtype=np.int16))
    monkeypatch.setattr("services.interactive_sessions.make_remote_session", FakeRemoteSession)
    monkeypatch.setenv("NNINTERACTIVE_ENABLED", "true")
    monkeypatch.setenv("NNINTERACTIVE_DAILY_QUOTA", "5")
    monkeypatch.setenv("NNINTERACTIVE_MAX_SESSIONS_PER_USER", "1")
    app = Flask(__name__)
    app.register_blueprint(interactive.interactive_blueprint, url_prefix="/api")
    app.testing = True
    try:
        yield app.test_client(), manager
    finally:
        manager.close_all()
        interactive._user_sessions.clear()


def test_starting_new_personal_session_closes_previous(replacement_client):
    client, manager = replacement_client
    first = client.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "1"})
    assert first.status_code == 201
    first_id = first.json["session_id"]

    second = client.post("/api/interactive/start", json={"dataset": "PanTS", "case_id": "2"})
    assert second.status_code == 201
    assert second.json["session_id"] != first_id
    assert manager.active_count("researcher") == 1
    with pytest.raises(InteractiveNotFound):
        manager.get(first_id, "researcher")
